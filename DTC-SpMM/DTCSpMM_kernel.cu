#include "config.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <fstream>
#include <mma.h>
#include <sputnik/spmm/cuda_spmm.h>
#include <sputnik/sputnik.h>
#include <sstream>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <torch/extension.h>
#include <vector>
#define WPB 8
#define EXE_TIME 1000
#define NUM_SM_GPU 128 // 4090
#define USE_SPUTNIK
using namespace nvcuda;

struct GpuTimer {
  cudaEvent_t start;
  cudaEvent_t stop;
  GpuTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start() { cudaEventRecord(start); }

  void Stop() { cudaEventRecord(stop); }

  float Elapsed() {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

// From (https://github.com/xxcclong/GNN-Computing)
typedef uint64_t clocktype;
struct Dur {
  clocktype begin;
  clocktype end;
  int smid = -1;
  Dur(clocktype x, clocktype y, int outsm) {
    begin = x;
    end = y;
    smid = outsm;
  }
};

bool cmp(Dur x, Dur y) { return (x.end > y.end); }
static __device__ inline uint64_t GlobalTimer64(void) {
  volatile uint64_t first_reading;
  volatile uint32_t second_reading;
  uint32_t high_bits_first;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(first_reading));
  high_bits_first = first_reading >> 32;
  asm volatile("mov.u32 %0, %%globaltimer_hi;" : "=r"(second_reading));
  if (high_bits_first == second_reading) {
    return first_reading;
  }
  // Return the value with the updated high bits, but the low bits set to 0.
  return ((uint64_t)second_reading) << 32;
}
__device__ inline uint getSMId() {
  uint smid;
  asm("mov.u32 %0, %smid;" : "=r"(smid));
  return smid;
}

//////////////////////////////////////////////////////////////////////
/// Preprocessing
//////////////////////////////////////////////////////////////////////
__global__ void roundup_to_multiple_of_eight(int *input, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size) {
    int rounded_value = ((input[tid] + 7) / 8) * 8;
    input[tid] = rounded_value;
  }
}

__global__ void get_padding_tileid_kernel(int *ori_offset, uint8_t *ori_tileid,
                                          int *padded_offset,
                                          uint8_t *padded_tileid, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    int s = ori_offset[tid];
    int e = ori_offset[tid + 1];
    int s1 = padded_offset[tid];
    for (int i = 0; i < e - s; i++) {
      padded_tileid[s1 + i] = ori_tileid[s + i];
    }
  }
}

__global__ void fill_edgeToRow(int *edgeToRow, int *nodePointer,
                               int num_nodes) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int nid = tid / 32;
  int laneid = tid % 32;
  // check a valid node range.
  if (nid < num_nodes) {
#pragma unroll
    for (int eid = nodePointer[nid] + laneid; eid < nodePointer[nid + 1];
         eid += 32) {
      edgeToRow[eid] = nid;
    }
  }
}
/*Generate segment*/
__global__ void fill_segment(int *nodePointer, int *seg_out, int blockSize_h,
                             int blockSize_w, int num_nodes) {
  int tid = threadIdx.x;
  int winId = blockIdx.x; // each warp one window
  unsigned block_start = nodePointer[winId * blockSize_h];
  unsigned block_end =
      nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
  unsigned num_window_edges = block_end - block_start;
  const unsigned threadPerBlock = blockDim.x * blockDim.y;
  for (unsigned idx = tid; idx < num_window_edges; idx += threadPerBlock) {
    seg_out[block_start + idx] = winId;
  }
}
void fill_segment_cuda(int *nodePointer, int *seg_out, int blockSize_h,
                       int blockSize_w, int num_nodes) {
  int block_size = 512;
  int window_count = (num_nodes + blockSize_h - 1) / blockSize_h;
  fill_segment<<<window_count, block_size>>>(nodePointer, seg_out, blockSize_h,
                                             blockSize_w, num_nodes);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

/*Generate TCblock_rowid*/
__global__ void generate_tcblock_rowid(int *rowwindow_offset,
                                       int *tcblock_rowid,
                                       int num_row_windows) {
  int tid = threadIdx.x;
  int winId = blockIdx.x; // each warp one window
  unsigned block_start = rowwindow_offset[winId];
  unsigned block_end = rowwindow_offset[min(winId + 1, num_row_windows)];
  unsigned num_blocks = block_end - block_start;
  const unsigned threadPerBlock = blockDim.x * blockDim.y;
  for (unsigned idx = tid; idx < num_blocks; idx += threadPerBlock) {
    tcblock_rowid[block_start + idx] = winId;
  }
}
void generate_tcblock_rowid_cuda(int *rowwindow_offset, int *tcblock_rowid,
                                 int num_row_windows) {
  int block_size = 512;
  int window_count = num_row_windows;
  generate_tcblock_rowid<<<window_count, block_size>>>(
      rowwindow_offset, tcblock_rowid, num_row_windows);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

/* Generate edge2column*/
__device__ __forceinline__ int binarysearch(int *arr, int size, int target) {
  int left = 0;
  int right = size - 1;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (arr[mid] == target) {
      while (mid > 0 && arr[mid - 1] == target) {
        mid--;
      }
      return mid;
    } else if (arr[mid] < target) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return -1;
}
__device__ __forceinline__ void inplace_deduplication(int *array, int length,
                                                      int *loc) {
  int cur = 1;
  while (cur < length) {
    if (array[cur] != array[cur - 1]) {
      (*loc)++;
      array[(*loc)] = array[cur];
    }
    cur++;
  }
}
__global__ void generate_edgetocolumn(int *nodePointer, int *edgelist,
                                      int *edgelist_sort, int *edgetocol,
                                      int *blockpartition, int *blocknum,
                                      int blockSize_h, int blockSize_w,
                                      int num_nodes) {
  int winId = blockIdx.x; // each warp one window
  unsigned block_start = nodePointer[winId * blockSize_h];
  unsigned block_end =
      nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
  unsigned num_window_edges = block_end - block_start;
  if (num_window_edges == 0)
    return;
  const unsigned threadPerBlock = blockDim.x * blockDim.y;
  int *start = edgelist_sort + block_start;
  int size = 0;
  inplace_deduplication(start, num_window_edges, &size);
  int num = (size + blockSize_w) / blockSize_w;
  atomicAdd(blocknum, num);
  blockpartition[winId] = num;
  for (unsigned idx = block_start; idx < block_end; idx += 1) {
    int index = binarysearch(start, size + 1, edgelist[idx]);
    edgetocol[idx] = index;
  }
}
void generate_edgetocolumn_cuda(int *nodePointer, int *edgelist,
                                int *edgelist_sort, int *edgetocol,
                                int *blockpartition, int *blocknum,
                                int blockSize_h, int blockSize_w,
                                int num_nodes) {
  int block_size = 1;
  int window_count = (num_nodes + blockSize_h - 1) / blockSize_h;
  int block_size1 = 128;
  int block_count1 = (window_count + 127) / 128;
  generate_edgetocolumn<<<window_count, block_size>>>(
      nodePointer, edgelist, edgelist_sort, edgetocol, blockpartition, blocknum,
      blockSize_h, blockSize_w, num_nodes);
  // generate_edgetocolumn_v1<<< window_count, block_size >>> (nodePointer,
  // edgelist, edgelist_sort, edgetocol, blockpartition, blocknum, blockSize_h,
  // blockSize_w, num_nodes);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

/*Generate TC offset, tileid and AtoB*/
__global__ void generate_tcoffset_id_atob(
    int *nodePointer, int *rowwindow_offset, int *edgeToColumn, int *edgeToRow,
    int *edgeList, int *tcblock_offset, uint8_t *tcblocktile_id,
    int *sparseatob, int max_block, int num_nodes, int blockSize_h,
    int blockSize_w, int num_row_windows) {
  extern __shared__ int pos_ptr[];
  int tid = threadIdx.x;
  int winId = blockIdx.x; // each warp one window
  unsigned block_start = rowwindow_offset[winId];
  unsigned block_end = rowwindow_offset[min(winId + 1, num_row_windows)];
  unsigned num_blocks = block_end - block_start;
  if (num_blocks == 0) {
    return;
  }
  int *tcblock_offset_ptr = pos_ptr + num_blocks;
  int *tcblock_offset_global_ptr = tcblock_offset + block_start;
  int *tcblock_nnz_ptr = pos_ptr + num_blocks + 1;
  unsigned element_start = nodePointer[winId * blockSize_h];
  unsigned element_end =
      nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
  unsigned num_window_edges = element_end - element_start;
  if (num_window_edges == 0) {
    return;
  }
  for (int i = 0; i < 2 * num_blocks + 1; i++) {
    pos_ptr[i] = 0;
  }
  for (unsigned e_index = element_start; e_index < element_end; e_index++) {
    unsigned col = edgeToColumn[e_index]; // new col
    tcblock_nnz_ptr[col / blockSize_w]++;
  }
  for (int i = 0; i < num_blocks; i++) {
    tcblock_offset_global_ptr[i] = tcblock_nnz_ptr[i];
  }
  auto tileid = tcblocktile_id + element_start;
  auto sparse_AToB = sparseatob + block_start * blockSize_w;
  for (int i = 0; i < num_blocks; i++) {
    tcblock_nnz_ptr[i] += tcblock_nnz_ptr[i - 1];
  }
  for (unsigned e_index = element_start; e_index < element_end; e_index++) {
    unsigned col = edgeToColumn[e_index]; // new col
    unsigned tcblock_id = col / blockSize_w;
    unsigned row_local = edgeToRow[e_index] % blockSize_h;
    unsigned col_local = col % blockSize_w;
    tileid[tcblock_offset_ptr[tcblock_id] + pos_ptr[tcblock_id]] =
        (uint8_t)(row_local * blockSize_w + col_local);
    sparse_AToB[tcblock_id * blockSize_w + col_local] = edgeList[e_index];
    pos_ptr[tcblock_id]++;
  }
}
void generate_tcoffset_id_atob_cuda(int *nodePointer, int *rowwindow_offset,
                                    int *edgeToColumn, int *edgeToRow,
                                    int *edgeList, int *tcblock_offset,
                                    uint8_t *tcblock_tileid, int *sparseatob,
                                    int max_block, int num_nodes,
                                    int blockSize_h, int blockSize_w,
                                    int num_row_windows) {
  int block_size = 1;
  int window_count = num_row_windows;
  const int dynamic_shared_size = (2 * max_block + 1) * sizeof(int);
  std::cout << "dynamic_shared_size: " << dynamic_shared_size << std::endl;
  if (dynamic_shared_size > 98304) {
    int maxbytes = 131072; // 96 KB
    cudaFuncSetAttribute(generate_tcoffset_id_atob,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
  } else if (dynamic_shared_size > 65536) {
    int maxbytes = 98304; // 96 KB
    cudaFuncSetAttribute(generate_tcoffset_id_atob,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
  } else if (dynamic_shared_size > 32768) {
    int maxbytes = 65536; // 128 KB
    cudaFuncSetAttribute(generate_tcoffset_id_atob,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
  }
  generate_tcoffset_id_atob<<<window_count, block_size, dynamic_shared_size>>>(
      nodePointer, rowwindow_offset, edgeToColumn, edgeToRow, edgeList,
      tcblock_offset, tcblock_tileid, sparseatob, max_block, num_nodes,
      blockSize_h, blockSize_w, num_row_windows);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}
void padding_up_8(int *input, int size) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  roundup_to_multiple_of_eight<<<blocksPerGrid, threadsPerBlock>>>(input, size);
}
void get_padding_tileid(int *ori_offset, uint8_t *ori_tileid,
                        int *padded_offset, uint8_t *padded_tileid, int size) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  get_padding_tileid_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      ori_offset, ori_tileid, padded_offset, padded_tileid, size);
}
/*main function*/
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int>
seg_sort_dequ(int *seg, int *edgeLists, int *nodepointer, int *edgetocol,
              int *edgetorow, int *blockpartition, int *block_num,
              int *rowwindow_offset, int blockSize_h, int blockSize_w,
              int num_nodes, int num_edges, int rowwindow_num) {
  thrust::device_ptr<int> Seg = thrust::device_pointer_cast(seg);
  thrust::device_vector<int> deviceSeg(Seg, Seg + num_edges);
  thrust::device_ptr<int> EL = thrust::device_pointer_cast(edgeLists);
  thrust::device_vector<int> deviceEL(EL, EL + num_edges);
  auto begin = thrust::make_zip_iterator(
      thrust::make_tuple(deviceSeg.begin(), deviceEL.begin()));
  auto end = thrust::make_zip_iterator(
      thrust::make_tuple(deviceSeg.end(), deviceEL.end()));
  thrust::sort(thrust::device, begin, end);
  generate_edgetocolumn_cuda(
      nodepointer, edgeLists, thrust::raw_pointer_cast(&deviceEL[0]), edgetocol,
      blockpartition, block_num, blockSize_h, blockSize_w, num_nodes);
  thrust::device_ptr<int> blockpartition_ptr =
      thrust::device_pointer_cast(blockpartition);
  thrust::device_ptr<int> rowwindow_offset_ptr =
      thrust::device_pointer_cast(rowwindow_offset + 1);
  thrust::device_vector<int> blockpartition_vector(
      blockpartition_ptr, blockpartition_ptr + rowwindow_num);
  thrust::inclusive_scan(blockpartition_vector.begin(),
                         blockpartition_vector.end(), rowwindow_offset_ptr);
  auto options_gpu =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  auto options_gpu_unit8 =
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
  thrust::device_ptr<int> bnum_ptr = thrust::device_pointer_cast(block_num);
  thrust::host_vector<int> bnum_vector(bnum_ptr, bnum_ptr + 1);
  int block_counter = bnum_vector[0];
  auto tcblock_rowid_tensor = torch::zeros({block_counter}, options_gpu);
  auto tcblock_rowid = tcblock_rowid_tensor.data<int>();
  generate_tcblock_rowid_cuda(rowwindow_offset, tcblock_rowid, rowwindow_num);
  auto max_element =
      thrust::max_element(thrust::device, blockpartition_vector.begin(),
                          blockpartition_vector.end());
  int max_blocks = *max_element;
  auto tcblocktile_id_tensor = torch::zeros({num_edges}, options_gpu_unit8);
  auto tcblock_offset_tensor = torch::zeros({block_counter + 1}, options_gpu);
  auto sparse_AToX_index_tensor =
      torch::zeros({block_counter * blockSize_w}, options_gpu);
  auto tcblock_offset = tcblock_offset_tensor.data<int>();
  auto sparse_AToX_index = sparse_AToX_index_tensor.data<int>();
  auto tcblocktile_id = tcblocktile_id_tensor.data<uint8_t>();
  generate_tcoffset_id_atob_cuda(
      nodepointer, rowwindow_offset, edgetocol, edgetorow, edgeLists,
      tcblock_offset + 1, tcblocktile_id, sparse_AToX_index, max_blocks,
      num_nodes, blockSize_h, blockSize_w, rowwindow_num);
  thrust::device_ptr<int> tcblock_offset_ptr =
      thrust::device_pointer_cast(tcblock_offset);
  thrust::inclusive_scan(tcblock_offset_ptr,
                         tcblock_offset_ptr + block_counter + 1,
                         tcblock_offset_ptr);
  return std::make_tuple(tcblock_offset_tensor, tcblock_rowid_tensor,
                         tcblocktile_id_tensor, sparse_AToX_index_tensor,
                         block_counter);
}
void fill_edgeToRow_cuda(int *edgeToRow, int *nodePointer, int num_nodes) {
  int wrap_size = 32;
  int block_size = 1024;
  int grid_size = (num_nodes * wrap_size + block_size - 1) / block_size;
  fill_edgeToRow<<<grid_size, block_size>>>(edgeToRow, nodePointer, num_nodes);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

//////////////////////
/// SPMM forward (GCN, GraphSAGE)
//////////////////////
__global__ void spmm_forward_cuda_kernel(
	const int * __restrict__ nodePointer,		// node pointer.
	const int *__restrict__ edgeList,			// edge list.
	const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
	const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
	const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ in_mat,		    // input feature matrix.
	float *out_mat							    // aggreAGNNed output feature matrix.
);
__global__ void spmm_forward_cuda_kernel_clock(
	const int * __restrict__ nodePointer,		// node pointer.
	const int *__restrict__ edgeList,			// edge list.
	const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
	const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
	const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output,							    // aggreAGNNed output feature matrix.
	clocktype* timer
);
__global__ void spmm_forward_cuda_kernel_improved(
	const int *__restrict__ Rowwindow_offset, 		// rowid of each TC block.
	const uint8_t *__restrict__ TCblocktile_id, 		// rowid of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
);
__global__ void spmm_forward_cuda_kernel_improved_with_value(
	const int *__restrict__ Rowwindow_offset, 		// rowid of each TC block.
	const uint8_t *__restrict__ TCblocktile_id, 		// rowid of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA,
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
);
__global__ void spmm_forward_cuda_kernel_improved_with_value_clock(
	const int *__restrict__ Rowwindow_offset, 		// rowid of each TC block.
	const uint8_t *__restrict__ TCblocktile_id, 		// rowid of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA,
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output,							    // output feature matrix.
	clocktype* timer
);

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8(
	const int *__restrict__ Rowwindow_offset, 		// rowid of each TC block.
	const uint8_t *__restrict__ TCblocktile_id, 		// rowid of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
);
__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1(
	const int *__restrict__ Rowwindow_offset, 		// rowid of each TC block.
	const uint8_t *__restrict__ TCblocktile_id, 		// rowid of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
);
__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value(
	const int *__restrict__ Rowwindow_offset, 		// rowid of each TC block.
	const uint8_t *__restrict__ TCblocktile_id, 		// rowid of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA, 
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
);

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer(
	const int *__restrict__ Rowwindow_offset, 		// rowid of each TC block.
	const uint8_t *__restrict__ TCblocktile_id, 		// rowid of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA, 
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
);
__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_double_buffer(
	const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
);
__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float2(
	const int *__restrict__ Rowwindow_offset, 		// rowid of each TC block.
	const uint8_t *__restrict__ TCblocktile_id, 		// rowid of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA, 
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
);
__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_double_buffer_float2(
	const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.		
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	float *input,		    // input feature matrix.
	float *output							    // output feature matrix.
);
__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_double_buffer_float2_split(
	const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.		
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	float *input,		    // input feature matrix.
	float *output							    // output feature matrix.
);

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float2_split(
	const int *__restrict__ Rowwindow_offset, 		// rowid of each TC block.
	const uint8_t *__restrict__ TCblocktile_id, 		// rowid of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA, 
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
);

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float4(
	const int *__restrict__ Rowwindow_offset, 		// rowid of each TC block.
	const uint8_t *__restrict__ TCblocktile_id, 		// rowid of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA, 
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
);
__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float4_split(
	const int *__restrict__ Rowwindow_offset, 		// rowid of each TC block.
	const uint8_t *__restrict__ TCblocktile_id, 		// rowid of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA, 
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
);
__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_double_buffer_float4_split(
	const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.	
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	float *input,		    // input feature matrix.
	float *output							    // output feature matrix.
);
__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_split(
	const int *__restrict__ Rowwindow_offset, 		// rowid of each TC block.
	const uint8_t *__restrict__ TCblocktile_id, 		// rowid of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA, 
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
);

__global__ void spmm_forward_cuda_kernel_improved_ptx_uint8_v1_strict_balance(
	const int *__restrict__ TCblock_rowid, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const int tc_count,
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
);
__global__ void spmm_forward_cuda_kernel_improved_ptx_uint8_v1_strict_balance_withv(
	const int *__restrict__ TCblock_rowid, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA, 		// colid of each TC block nonzero element.
	const int tc_count,
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
);
__global__ void spmm_forward_cuda_kernel_improved_ptx_uint8_v1_strict_balance_withv_clock(
	const int *__restrict__ TCblock_rowid, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA, 		// colid of each TC block nonzero element.
	const int tc_count,
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output,							    // output feature matrix.
	clocktype* timer
);
__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float4_split_balance(
	const int *__restrict__ TCblock_rowid, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA, 		
	const int tc_count,
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
);





////////////////////////////////////////////
//
// SPMM Foward Pass  (GCN, GraphSAGE), from TC-GNN
//
////////////////////////////////////////////
// #define SMCLOCK
std::vector<torch::Tensor> spmm_forward_cuda(
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow,
              int num_nodes,
              int num_edges,
              int embedding_dim,
    torch::Tensor input
) 
{
    auto output = torch::zeros_like(input);
    const int num_row_windows = blockPartition.size(0);
    // const int WARPperBlock = WPB;
	const int WARPperBlock = (embedding_dim + BLK_H - 1) / BLK_H;
    dim3 grid(num_row_windows, 1, 1);
    dim3 block(WARP_SIZE, WARPperBlock, 1);
    const int dimTileNum = (embedding_dim + BLK_H - 1) / BLK_H;
	const int dynamic_shared_size = 2 * dimTileNum * BLK_W * BLK_H * sizeof(float); // dynamic shared memory.
#ifndef SMCLOCK
    GpuTimer timer;
    timer.Start();
	for(int i = 0; i < EXE_TIME; i++){
		spmm_forward_cuda_kernel<<<grid, block, dynamic_shared_size>>>(
			nodePointer.data<int>(), 
			edgeList.data<int>(),
			blockPartition.data<int>(), 
			edgeToColumn.data<int>(), 
			edgeToRow.data<int>(), 
			num_nodes,
			num_edges,
			embedding_dim,
			input.data<float>(), 
			output.data<float>()
		);
	}
	timer.Stop();
	float origin_tcgnn_time = timer.Elapsed() / EXE_TIME;
	std::ofstream res_file;
	res_file.open("TCGNNSpMM_exe_time_and_throughput.csv", std::ios::app);
	float spmm_flop = float(num_edges) * float(embedding_dim) * 2.0;
	float origin_tcgnn_throughput_ = (float(spmm_flop * 1000.))/(origin_tcgnn_time * 1000. * 1000. * 1000.);
	res_file << "TCGNNSpMM:" << "," << origin_tcgnn_time << "," << origin_tcgnn_throughput_ << ",";
	res_file.close();
#else
	clocktype* thetimer;
	cudaMalloc((void**)&thetimer, sizeof(clocktype) * 3 * num_row_windows);
	cudaMemset(thetimer, 0, 3 * num_row_windows * sizeof(clocktype));
    spmm_forward_cuda_kernel_clock<<<grid, block, dynamic_shared_size>>>(
                                                                    nodePointer.data<int>(), 
                                                                    edgeList.data<int>(),
                                                                    blockPartition.data<int>(), 
                                                                    edgeToColumn.data<int>(), 
                                                                    edgeToRow.data<int>(), 
                                                                    num_nodes,
                                                                    num_edges,
                                                                    embedding_dim,
                                                                    input.data<float>(), 
                                                                    output.data<float>(),
																	thetimer
                                                                );
    clocktype* cpu_timer = new clocktype[3 * num_row_windows];
    memset(cpu_timer, 0, 3 * num_row_windows * sizeof(clocktype));
    cudaMemcpy(cpu_timer, thetimer, sizeof(clocktype) * 3 * num_row_windows, cudaMemcpyDeviceToHost);
	std::vector<Dur> v;
    for(int j = 0; j < num_row_windows; ++j) {
        v.push_back(Dur(cpu_timer[j * 3], cpu_timer[j * 3 + 1], (int)(cpu_timer[j * 3 + 2])));
    }
	clocktype* start_time_sm = new clocktype[NUM_SM_GPU];
	for (int i = 0; i < NUM_SM_GPU; i++) {
		start_time_sm[i] = LONG_LONG_MAX;
	}
	clocktype* end_time_sm = new clocktype[NUM_SM_GPU];
	for (int i = 0; i < NUM_SM_GPU; i++) {
		end_time_sm[i] = 0;
	}
	for(auto item : v) {
       if (item.begin < start_time_sm[item.smid]) {
		 start_time_sm[item.smid] = item.begin;
	   }
	   if (item.end > end_time_sm[item.smid]) {
		end_time_sm[item.smid] = item.end;
	  }
	}
	std::ofstream out("block_time.csv");
	if (out.is_open()) {
	  for (int i = 0; i < NUM_SM_GPU; i++) {
		out << (double)(end_time_sm[i] - start_time_sm[i])/1e6 << std::endl;
	  }
    }
	out.close();
	cudaFree(thetimer);
	delete[] cpu_timer;
	delete[] start_time_sm;
	delete[] end_time_sm;
#endif
    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    return {output};
}


std::vector<torch::Tensor> spmm_forward_improved_ptx_uint8_cuda(
	torch::Tensor Rowwindow_offset,
	torch::Tensor TCblocktile_id,
	torch::Tensor TCblock_offset,
	torch::Tensor sparse_AToX_idx,
              int num_nodes,
              int num_edges,
              int embedding_dim,
    torch::Tensor input,
	std::string exeplan
) 
{
	auto output = torch::zeros_like(input);
	const int num_row_windows = Rowwindow_offset.size(0) - 1;
	const int WARPperBlock = embedding_dim / BLK_H;
	const int WARPperBlock1 = embedding_dim / 32;
    dim3 grid(num_row_windows, 1, 1);
    dim3 block(WARP_SIZE, WARPperBlock, 1);
	dim3 grid_split(num_row_windows, WARPperBlock / 4, 1);
    dim3 block_split(WARP_SIZE, 4, 1);
	dim3 grid_float4(num_row_windows, 1, 1);
    dim3 block_float4(WARP_SIZE, WARPperBlock1, 1);
	dim3 grid_float4_split(num_row_windows, WARPperBlock1 / 4, 1);
    dim3 block_float4_split(WARP_SIZE, 4, 1);
    const int dimTileNum = (embedding_dim + BLK_H - 1) / BLK_H;
	auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
	auto val = torch::ones({TCblocktile_id.size(0)}, options);
	float *valuesA = val.data<float>();
	GpuTimer timer;
	if (exeplan == "float_nonsplit") {
		timer.Start();
		for(int i = 0; i < EXE_TIME; i++) {
		  spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer<<<grid, block>>>(
																		Rowwindow_offset.data<int>(), 
																		TCblocktile_id.data<uint8_t>(), 
																		TCblock_offset.data<int>(), 
																		sparse_AToX_idx.data<int>(), 
																		valuesA,
																		num_nodes,
																		num_edges,
																		embedding_dim,
																		input.data<float>(), 
																		output.data<float>()
																	);
		}
		timer.Stop();
	} else if (exeplan == "float2_nonsplit") {
		timer.Start();
		for(int i = 0; i < EXE_TIME; i++) {
		  spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float2<<<grid, block>>>(
																		Rowwindow_offset.data<int>(), 
																		TCblocktile_id.data<uint8_t>(), 
																		TCblock_offset.data<int>(), 
																		sparse_AToX_idx.data<int>(), 
																		valuesA,
																		num_nodes,
																		num_edges,
																		embedding_dim,
																		input.data<float>(), 
																		output.data<float>()
																	);
		}
		timer.Stop();
	} else if (exeplan == "float2_split") {
		timer.Start();
		if (embedding_dim >= 64)
		for(int i = 0; i < EXE_TIME; i++) {
		  spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float2_split<<<grid_split, block_split>>>(
																		Rowwindow_offset.data<int>(), 
																		TCblocktile_id.data<uint8_t>(), 
																		TCblock_offset.data<int>(), 
																		sparse_AToX_idx.data<int>(), 
																		valuesA,
																		num_nodes,
																		num_edges,
																		embedding_dim,
																		input.data<float>(), 
																		output.data<float>()
																	);
		}
		timer.Stop();
	} else if (exeplan == "float4_nonsplit") {
		timer.Start();
		for(int i = 0; i < EXE_TIME; i++) {
		  spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float4<<<grid_float4, block_float4>>>(
																		Rowwindow_offset.data<int>(), 
																		TCblocktile_id.data<uint8_t>(), 
																		TCblock_offset.data<int>(), 
																		sparse_AToX_idx.data<int>(), 
																		valuesA,
																		num_nodes,
																		num_edges,
																		embedding_dim,
																		input.data<float>(), 
																		output.data<float>()
																	);
		}
		timer.Stop();
	} else if (exeplan == "float4_split") {
		timer.Start();
		if (embedding_dim >= 128)
		for(int i = 0; i < EXE_TIME; i++) {
		  spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float4_split<<<grid_float4_split, block_float4_split>>>(
																		Rowwindow_offset.data<int>(), 
																		TCblocktile_id.data<uint8_t>(), 
																		TCblock_offset.data<int>(), 
																		sparse_AToX_idx.data<int>(), 
																		valuesA,
																		num_nodes,
																		num_edges,
																		embedding_dim,
																		input.data<float>(), 
																		output.data<float>()
																	);
		}
		timer.Stop();
	} else {
		std::cout << "Not supported exe plan!";
		exit(-1);
	}
    float dtc_time = timer.Elapsed() / EXE_TIME;
	std::ofstream res_file;
	res_file.open("DTCSpMM_exe_time_and_throughput.csv", std::ios::app);
	float spmm_flop = float(num_edges) * float(embedding_dim) * 2.0;
	float DTC_SpMM_throughput_ = (float(spmm_flop * 1000.))/(dtc_time * 1000. * 1000. * 1000.);
	res_file << "DTCSpMM:" << "," << dtc_time << "," << DTC_SpMM_throughput_ << ",";
	res_file.close();
	cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    return {output};
}




std::vector<torch::Tensor> spmm_forward_cuda_origin_clock(
	torch::Tensor Rowwindow_offset,
	torch::Tensor TCblocktile_id,
	torch::Tensor TCblock_offset,
	torch::Tensor sparse_AToX_idx,
              int num_nodes,
              int num_edges,
              int embedding_dim,
    torch::Tensor input
) {
	auto output = torch::zeros_like(input);
	const int num_row_windows = Rowwindow_offset.size(0) - 1;
	const int WARPperBlock = embedding_dim / BLK_H;
	const int WARPperBlock1 = embedding_dim / 32;
    dim3 grid(num_row_windows, 1, 1);
    dim3 block(WARP_SIZE, WARPperBlock, 1);
    const int dimTileNum = (embedding_dim + BLK_H - 1) / BLK_H;
	auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
	auto val = torch::ones({TCblocktile_id.size(0)}, options);
	float *valuesA = val.data<float>();
	clocktype* thetimer;
	cudaMalloc((void**)&thetimer, sizeof(clocktype) * 3 * num_row_windows);
	cudaMemset(thetimer, 0, 3 * num_row_windows * sizeof(clocktype));
	// BASE
	const int dynamic_shared_size = 2 * dimTileNum * BLK_W * BLK_H * sizeof(float); // dynamic shared memory.
	for(int i = 0; i < 1; i++)
	  spmm_forward_cuda_kernel_improved_with_value_clock<<<grid, block, dynamic_shared_size>>>(
                                                                    Rowwindow_offset.data<int>(), 
                                                                    TCblocktile_id.data<uint8_t>(), 
                                                                    TCblock_offset.data<int>(), 
																	sparse_AToX_idx.data<int>(), 
																	valuesA,
                                                                    num_nodes,
                                                                    num_edges,
                                                                    embedding_dim,
                                                                    input.data<float>(), 
                                                                    output.data<float>(),
																	thetimer
                                                                );
    
	clocktype* cpu_timer = new clocktype[3 * num_row_windows];
    memset(cpu_timer, 0, 3 * num_row_windows * sizeof(clocktype));
    cudaMemcpy(cpu_timer, thetimer, sizeof(clocktype) * 3 * num_row_windows, cudaMemcpyDeviceToHost);
	std::vector<Dur> v;
    for(int j = 0; j < num_row_windows; ++j) {
        v.push_back(Dur(cpu_timer[j * 3], cpu_timer[j * 3 + 1], (int)(cpu_timer[j * 3 + 2])));
    }
	clocktype* start_time_sm = new clocktype[NUM_SM_GPU];
	for (int i = 0; i < NUM_SM_GPU; i++) {
		start_time_sm[i] = LONG_LONG_MAX;
	}
	clocktype* end_time_sm = new clocktype[NUM_SM_GPU];
	for (int i = 0; i < NUM_SM_GPU; i++) {
		end_time_sm[i] = 0;
	}
	for(auto item : v) {
       if (item.begin < start_time_sm[item.smid]) {
		 start_time_sm[item.smid] = item.begin;
	   }
	   if (item.end > end_time_sm[item.smid]) {
		end_time_sm[item.smid] = item.end;
	  }
	}
	std::ofstream out("block_time_ours_before_balance.csv");
	if (out.is_open()) {
	  for (int i = 0; i < NUM_SM_GPU; i++) {
		out << (double)(end_time_sm[i] - start_time_sm[i])/1e6 << std::endl;
	  }
    }
	out.close();

	cudaFree(thetimer);
	delete[] cpu_timer;
	delete[] start_time_sm;
	delete[] end_time_sm;
	cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    return {output};
}

std::vector<torch::Tensor> spmm_forward_improved_ptx_uint8_cuda_dtc_for_gcn(
	torch::Tensor Rowwindow_offset,
	torch::Tensor TCblocktile_id,
	torch::Tensor TCblock_offset,
	torch::Tensor sparse_AToX_idx,
              int num_nodes,
              int num_edges,
              int embedding_dim,
    torch::Tensor input
) 
{
	auto output = torch::zeros_like(input);
	const int num_row_windows = Rowwindow_offset.size(0) - 1;
	if (embedding_dim >= 128 && embedding_dim % 4 == 0) {
	  const int WARPperBlock1 = embedding_dim / 32;
	  dim3 grid_float4_split(num_row_windows, WARPperBlock1 / 4, 1);
      dim3 block_float4_split(WARP_SIZE, 4, 1);
	  spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_double_buffer_float4_split<<<grid_float4_split, block_float4_split>>>(
																  Rowwindow_offset.data<int>(), 
																  TCblocktile_id.data<uint8_t>(), 
																  TCblock_offset.data<int>(), 
																  sparse_AToX_idx.data<int>(), 
																  num_nodes,
																  num_edges,
																  embedding_dim,
																  input.data<float>(), 
																  output.data<float>()
															  );
    } else if (embedding_dim >= 64 && embedding_dim % 2 == 0) {
	   const int WARPperBlock = embedding_dim / BLK_H;
	   dim3 grid_split(num_row_windows, WARPperBlock / 4, 1);
	   dim3 block_split(WARP_SIZE, 4, 1);
	   spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_double_buffer_float2_split<<<grid_split, block_split>>>(
                                                                    Rowwindow_offset.data<int>(), 
                                                                    TCblocktile_id.data<uint8_t>(), 
                                                                    TCblock_offset.data<int>(), 
																	sparse_AToX_idx.data<int>(), 
                                                                    num_nodes,
                                                                    num_edges,
                                                                    embedding_dim,
                                                                    input.data<float>(), 
                                                                    output.data<float>()
                                                                );
	}
    else if (embedding_dim % 2 == 0) {
	  const int WARPperBlock = embedding_dim / BLK_H;
	  dim3 grid(num_row_windows, 1, 1);
	  dim3 block(WARP_SIZE, WARPperBlock, 1);
	  spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_double_buffer_float2<<<grid, block>>>(
                                                                    Rowwindow_offset.data<int>(), 
                                                                    TCblocktile_id.data<uint8_t>(), 
                                                                    TCblock_offset.data<int>(), 
																	sparse_AToX_idx.data<int>(), 
                                                                    num_nodes,
                                                                    num_edges,
                                                                    embedding_dim,
                                                                    input.data<float>(), 
                                                                    output.data<float>()
                                                                );
	}
	else {
	  const int WARPperBlock = embedding_dim / BLK_H;
	  dim3 grid(num_row_windows, 1, 1);
	  dim3 block(WARP_SIZE, WARPperBlock, 1);
	  spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_double_buffer<<<grid, block>>>(
                                                                    Rowwindow_offset.data<int>(), 
                                                                    TCblocktile_id.data<uint8_t>(), 
                                                                    TCblock_offset.data<int>(), 
																	sparse_AToX_idx.data<int>(), 
                                                                    num_nodes,
                                                                    num_edges,
                                                                    embedding_dim,
                                                                    input.data<float>(), 
                                                                    output.data<float>()
                                                                );
	}
	cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    return {output};
}

std::vector<torch::Tensor> spmm_balance_forward_cuda_ptx_unit8_prefetch(
	torch::Tensor TCblock_rowid,
	torch::Tensor TCblocktile_id,
	torch::Tensor TCblock_offset,
	torch::Tensor sparse_AToX_idx,
	          int tc_count,
              int num_nodes,
              int num_edges,
              int embedding_dim,
    torch::Tensor input,
	std::string exeplan
) {
	auto output = torch::zeros_like(input);
	const int WARPperBlock = embedding_dim / BLK_H;
	const int WARPperBlock1 = embedding_dim / 32;
    dim3 grid((tc_count + TCBLOCK_PER_WARP - 1) / TCBLOCK_PER_WARP, 1, 1);
    dim3 block(WARP_SIZE, WARPperBlock, 1);
	dim3 grid_float4_split((tc_count + TCBLOCK_PER_WARP - 1) / TCBLOCK_PER_WARP, WARPperBlock1 / 4, 1);
    dim3 block_float4_split(WARP_SIZE, 4, 1);

    const int dimTileNum = (embedding_dim + BLK_H - 1) / BLK_H;
	auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
	auto val = torch::ones({num_edges}, options);
	float *valuesA = val.data<float>();
	GpuTimer timer;
	if (exeplan == "float_nonsplit") {
		timer.Start();
		for(int i = 0; i < EXE_TIME; i++) {
		  spmm_forward_cuda_kernel_improved_ptx_uint8_v1_strict_balance_withv<<<grid, block>>>(
																		TCblock_rowid.data<int>(), 
																		TCblocktile_id.data<uint8_t>(), 
																		TCblock_offset.data<int>(), 
																		sparse_AToX_idx.data<int>(), 
																		valuesA,
																		tc_count, 
																		num_nodes,
																		num_edges,
																		embedding_dim,
																		input.data<float>(), 
																		output.data<float>()
																	);
		}
		timer.Stop();
	} else if (exeplan == "float4_split") {
		timer.Start();
		if (embedding_dim >= 128)
		for (int i = 0; i < EXE_TIME; i++) {
			spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float4_split_balance<<<grid_float4_split, block_float4_split>>>(
																		TCblock_rowid.data<int>(), 
																		TCblocktile_id.data<uint8_t>(), 
																		TCblock_offset.data<int>(), 
																		sparse_AToX_idx.data<int>(), 
																		valuesA,
																		tc_count, 
																		num_nodes,
																		num_edges,
																		embedding_dim,
																		input.data<float>(), 
																		output.data<float>()
																	);
		}
		timer.Stop();
	} else {
		std::cout << "Not supported exe plan!";
		exit(-1);
	} 

	float DTC_time = timer.Elapsed() / EXE_TIME;
	std::ofstream res_file;
	res_file.open("DTCSpMM_exe_time_and_throughput.csv", std::ios::app);
	float spmm_flop = float(num_edges) * float(embedding_dim) * 2.0;
	float DTC_throughput = (float(spmm_flop * 1000.))/(DTC_time * 1000. * 1000. * 1000.);
	res_file << "DTCSpMM:" << "," << DTC_time << "," << DTC_throughput;
	res_file.close();
	// check for exe error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    return {output};
}


std::vector<torch::Tensor> spmm_balance_clock (
	torch::Tensor TCblock_rowid,
	torch::Tensor TCblocktile_id,
	torch::Tensor TCblock_offset,
	torch::Tensor sparse_AToX_idx,
	          int tc_count,
              int num_nodes,
              int num_edges,
              int embedding_dim,
    torch::Tensor input
) {
	auto output = torch::zeros_like(input);
	const int WARPperBlock = embedding_dim / BLK_H;
	const int WARPperBlock1 = embedding_dim / 32;
    dim3 grid((tc_count + TCBLOCK_PER_WARP - 1) / TCBLOCK_PER_WARP, 1, 1);
	std::cout << "TCBLOCK_PER_WARP:" << TCBLOCK_PER_WARP << std::endl;
    dim3 block(WARP_SIZE, WARPperBlock, 1);


    const int dimTileNum = (embedding_dim + BLK_H - 1) / BLK_H;
	auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
	auto val = torch::ones({num_edges}, options);
	float *valuesA = val.data<float>();
    

	int num_row_windows = (tc_count + TCBLOCK_PER_WARP - 1) / TCBLOCK_PER_WARP;
	clocktype* thetimer;
	cudaMalloc((void**)&thetimer, sizeof(clocktype) * 3 * num_row_windows);
	cudaMemset(thetimer, 0, 3 * num_row_windows * sizeof(clocktype));


	for(int i = 0; i < 10; i++) {
	  spmm_forward_cuda_kernel_improved_ptx_uint8_v1_strict_balance_withv_clock<<<grid, block>>>(
																	TCblock_rowid.data<int>(), 
                                                                    TCblocktile_id.data<uint8_t>(), 
                                                                    TCblock_offset.data<int>(), 
																	sparse_AToX_idx.data<int>(), 
																	valuesA,
																	tc_count, 
                                                                    num_nodes,
                                                                    num_edges,
                                                                    embedding_dim,
                                                                    input.data<float>(), 
                                                                    output.data<float>(),
																	thetimer
                                                                );
	}
	clocktype* cpu_timer = new clocktype[3 * num_row_windows];
    memset(cpu_timer, 0, 3 * num_row_windows * sizeof(clocktype));
    cudaMemcpy(cpu_timer, thetimer, sizeof(clocktype) * 3 * num_row_windows, cudaMemcpyDeviceToHost);
	std::vector<Dur> v;
    for(int j = 0; j < num_row_windows; ++j) {
        v.push_back(Dur(cpu_timer[j * 3], cpu_timer[j * 3 + 1], (int)(cpu_timer[j * 3 + 2])));
    }
	clocktype* start_time_sm = new clocktype[NUM_SM_GPU];
	for (int i = 0; i < NUM_SM_GPU; i++) {
		start_time_sm[i] = LONG_LONG_MAX;
	}
	clocktype* end_time_sm = new clocktype[NUM_SM_GPU];
	for (int i = 0; i < NUM_SM_GPU; i++) {
		end_time_sm[i] = 0;
	}
	for(auto item : v) {
       if (item.begin < start_time_sm[item.smid]) {
		 start_time_sm[item.smid] = item.begin;
	   }
	   if (item.end > end_time_sm[item.smid]) {
		end_time_sm[item.smid] = item.end;
	  }
	}
	std::ofstream out("block_time_ours_after_balance.csv");
	if (out.is_open()) {
	  for (int i = 0; i < NUM_SM_GPU; i++) {
		out << (double)(end_time_sm[i] - start_time_sm[i])/1e6 << std::endl;
	  }
    }
	out.close();
	cudaFree(thetimer);
	delete[] cpu_timer;
	delete[] start_time_sm;
	delete[] end_time_sm;

	// check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    return {output};
}


// The codes are from sputnik (https://github.com/google-research/sputnik)
  void SortedRowSwizzle(int rows, const int *row_offsets, int *row_indices) {
	std::vector<int> swizzle_staging(rows);
	std::iota(swizzle_staging.begin(), swizzle_staging.end(), 0);
	std::sort(swizzle_staging.begin(), swizzle_staging.end(),
			  [&row_offsets](int idx_a, int idx_b) {
				int length_a = row_offsets[idx_a + 1] - row_offsets[idx_a];
				int length_b = row_offsets[idx_b + 1] - row_offsets[idx_b];
				return length_a > length_b;
			  });
	std::memcpy(row_indices, swizzle_staging.data(), sizeof(int) * rows);
  }
  void IdentityRowSwizzle(int rows, const int * /* unused */, int *row_indices) {
	std::iota(row_indices, row_indices + rows, 0);
  }

  std::vector<torch::Tensor> spmm_forward_sputnik(torch::Tensor rowoffset, torch::Tensor colind, torch::Tensor input, int num_nodes, int num_edges, int embedding_dim) {
	auto output = torch::zeros_like(input);
	#ifdef USE_SPUTNIK
	GpuTimer timer;
	const int m = num_nodes;
	const int k = num_nodes;
	const int n = embedding_dim;
	using scalar_t = float;
	int nnzA = num_edges;
	int *rowindA_csr = rowoffset.data<int>();
	int *colindA = colind.data<int>();
	auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
	auto val = torch::ones({num_edges}, options);
	scalar_t *valuesA = val.data<scalar_t>();
	scalar_t *B = input.data<scalar_t>();
	scalar_t *C = output.data<scalar_t>();
	auto order = torch::zeros({m},  torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
	for(int i = 0; i < m; i++){
	  order.data<int>()[i] = i;
	}
	//  May harm spmm's performance. thus we comment it out.
    //  SortedRowSwizzle(m, rowoffset.to(torch::kCPU).data<int>(), order.data<int>()); 
	auto order1 = order.to(torch::kCUDA);
	timer.Start();
	for (int i = 0; i < EXE_TIME; i++) {
	sputnik::CudaSpmm(
		m, k, n, nnzA,
		order1.data<int>(), valuesA, rowindA_csr, 
		colindA,
		B, C, 0);
	}
	timer.Stop();
	float sputnik_time = timer.Elapsed() / EXE_TIME;
	std::ofstream res_file;
	res_file.open("Sputnik_exe_time_and_throughput.csv", std::ios::app);
	float spmm_flop = float(num_edges) * float(embedding_dim) * 2.0;
	float sput_throughput_ = (float(spmm_flop * 1000.))/(sputnik_time * 1000. * 1000. * 1000.);
	res_file << "Sputnik:" << "," << sputnik_time << "," << sput_throughput_ << ",";
	res_file.close();	
	#endif
	return {output};
  }


std::vector<torch::Tensor> spmm_forward_cusparse(torch::Tensor rowoffset, torch::Tensor colind, torch::Tensor input, int num_nodes, int num_edges, int embedding_dim, int algid) {
	GpuTimer timer;
	const int m = num_nodes;
	const int k = num_nodes;
	const int n = embedding_dim;
	using scalar_t = float;
	scalar_t alpha = 1.0;
	scalar_t beta = 0.0;
	int nnzA = num_edges;
	int *rowindA_csr = rowoffset.data<int>();
	int *colindA = colind.data<int>();
	auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
	auto val = torch::ones({num_edges}, options);
	scalar_t *valuesA = val.data<scalar_t>();
	scalar_t *B = input.data<scalar_t>();
	auto output = torch::zeros_like(input);
	scalar_t *C = output.data<scalar_t>();
	int ldb = n;
	int ldc = m;
   // cuda handle
	cusparseHandle_t cusparse_handle = 0;
	cusparseCreate(&cusparse_handle);
  #if CUDART_VERSION < 11000
	cusparseMatDescr_t descr = 0;
	cusparseCreateMatDescr(&descr);
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
	// kernel
	for (int i = 0; i < EXE_TIME; i++) {
	cusparseScsrmm2(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
	  CUSPARSE_OPERATION_TRANSPOSE, m, n, k,
	  nnzA, &alpha, descr, valuesA, rowindA_csr, colindA,
	  B, ldb, &beta, C, ldc);
	}
	timer.Start();
	for (int i = 0; i < EXE_TIME; i++) {
	cusparseScsrmm2(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
	  CUSPARSE_OPERATION_TRANSPOSE, m, n, k,
	  nnzA, &alpha, descr, valuesA, rowindA_csr, colindA,
	  B, ldb, &beta, C, ldc);
	}
	timer.Stop();
	time =  (float)timer.Elapsed()/EXE_TIME;
	std::cout << "cuSPARSE 101 time = " << time << " ms" << std::endl;
  #else
	cusparseSpMatDescr_t matA;
	cusparseDnMatDescr_t matB, matC;
	cusparseCreateCsr(&matA,
		m, k, nnzA,
		rowindA_csr,
		colindA,
		valuesA,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
		CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
	cusparseCreateDnMat(&matB,
		k, n, n,
		B, CUDA_R_32F, CUSPARSE_ORDER_ROW);
	cusparseCreateDnMat(&matC,
		m, n, n,
		C, CUDA_R_32F, CUSPARSE_ORDER_ROW);
  
	auto transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
	auto transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
	size_t workspace_size;
	cusparseSpMMAlg_t alg = CUSPARSE_SPMM_CSR_ALG2;
	if(algid == -1){
	  alg = CUSPARSE_SPMM_ALG_DEFAULT;
	} else if(algid == 2){
	  alg = CUSPARSE_SPMM_CSR_ALG2;
	} else if (algid == 3) {
	  alg = CUSPARSE_SPMM_CSR_ALG3;
	}
		for (int i = 0; i < EXE_TIME; i++) {
		  cusparseSpMM_bufferSize(
			cusparse_handle, transA, transB,
			&alpha, matA, matB, &beta, matC,
			CUDA_R_32F, alg,
			&workspace_size);
		}
		timer.Start();
		for (int i = 0; i < EXE_TIME; i++) {
		  cusparseSpMM_bufferSize(
			cusparse_handle, transA, transB,
			&alpha, matA, matB, &beta, matC,
			CUDA_R_32F, alg,
			&workspace_size);
		}
		timer.Stop();
		std::cout << "cusparse csr buffer time: " <<  timer.Elapsed() / EXE_TIME << " ms " << std::endl;
		void* workspace=NULL;
		cudaMalloc(&workspace, workspace_size);
		for (int i = 0; i < EXE_TIME; i++) {
		  cusparseSpMM(
			cusparse_handle, transA, transB,
			&alpha, matA, matB, &beta, matC,
			CUDA_R_32F,alg,
			workspace);
		}
		timer.Start();
		for (int i = 0; i < EXE_TIME; i++) {
		  cusparseSpMM(
			  cusparse_handle, transA, transB,
			  &alpha, matA, matB, &beta, matC,
			  CUDA_R_32F, alg,
			  workspace);
		}
		timer.Stop();  
		float cus_time = timer.Elapsed() / EXE_TIME;
		std::ofstream res_file;
		res_file.open("cuSPARSE_exe_time_and_throughput.csv", std::ios::app);
		float spmm_flop = float(num_edges) * float(embedding_dim) * 2.0;
		float cus_throughput_ = (float(spmm_flop * 1000.))/(cus_time * 1000. * 1000. * 1000.);
		res_file << "cusparse" + std::to_string(algid) + ":" << "," << cus_time << "," << cus_throughput_ << ",";
		res_file.close();	
		cudaFree(workspace);
	cusparseDestroySpMat(matA);
	cusparseDestroyDnMat(matB);
	cusparseDestroyDnMat(matC);
  #endif 
  return {output};
  }

  std::vector<torch::Tensor> spmm_forward_cusparse_blocked_ellpack(torch::Tensor ell_colind, torch::Tensor input, int num_nodes, int block_size, int ell_columns, int embedding_dim) {
	GpuTimer timer;
    long long int   A_num_rows      = num_nodes;
    int   A_num_cols      = num_nodes;
    int   A_ell_blocksize = block_size;
    long long   A_ell_cols      = ell_columns;
	std::cout << A_num_rows << " " << A_num_cols << " " << A_ell_blocksize << " " << A_ell_cols << std::endl;
    int   A_num_blocks    = A_ell_cols * A_num_rows /
                           (A_ell_blocksize * A_ell_blocksize);
    int   B_num_rows      = A_num_cols;
    int   B_num_cols      = embedding_dim;
    int   ldb             = B_num_rows;
    int   ldc             = A_num_rows;
    int   B_size          = ldb * B_num_cols;
    int   C_size          = ldc * B_num_cols;
	using scalar_t = float;
	auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
	auto output = torch::zeros_like(input);
	auto val_size = A_num_rows*A_ell_cols;
	if (val_size * sizeof(float) > 21474836480) {
		std::ofstream res_file;
		res_file.open("cusparse_ell_exe_time.csv", std::ios::app);
		res_file << "cusparse(bell-" + std::to_string(block_size) + "):" << "," << "OOM" << ",";
		res_file.close();
		return {output};
	}
	auto val = torch::ones({val_size}, options);
	scalar_t *dB = input.data<scalar_t>();
	scalar_t *dC = output.data<scalar_t>();
	scalar_t *dA_values = val.data<scalar_t>();
	int *dA_columns = ell_colind.data<int>();


   // cuda handle
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
	float alpha           = 1.0f;
    float beta            = 0.0f;
    cusparseCreate(&handle);
    // Create sparse matrix A in blocked ELL format
    cusparseCreateBlockedEll(
                                      &matA,
                                      A_num_rows, A_num_cols, A_ell_blocksize,
                                      A_ell_cols, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    // Create dense matrix B
    cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL);
    // Create dense matrix C
    cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL);
    // allocate an external buffer if needed
    cusparseSpMM_bufferSize(
                        	handle,
                        	CUSPARSE_OPERATION_NON_TRANSPOSE,
                        	CUSPARSE_OPERATION_NON_TRANSPOSE,
                        	&alpha, matA, matB, &beta, matC, CUDA_R_32F,
                        	CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);
	timer.Start();
	for (int i = 0; i < EXE_TIME; i++) {
    // execute SpMM
      cusparseSpMM(handle,
                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                   &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                   CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
	}
	timer.Stop();
	float cus_ell_time = timer.Elapsed() / EXE_TIME;
	std::cout << "time:" <<  cus_ell_time << std::endl;
	std::ofstream res_file;
	res_file.open("cusparse_ell_exe_time.csv", std::ios::app);
	res_file << "cusparse(bell-" + std::to_string(block_size) + "):" << "," << cus_ell_time << ",";
	res_file.close();
	cudaFree(dBuffer);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(handle);
    return {output};
  }


// The codes are from TC-GNN ()
__global__ void spmm_forward_cuda_kernel(
	const int * __restrict__ nodePointer,		// node pointer.
	const int *__restrict__ edgeList,			// edge list.
	const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
	const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
	const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // aggreAGNNed output feature matrix.
) {
    const unsigned bid = blockIdx.x;								// block_index == row_window_index
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16.
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block.
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.

	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned nIdx_start = bid * BLK_H;					    // starting nodeIdx of current row_window.
	const unsigned nIdx_end = min((bid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.
	
	const unsigned eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
	const unsigned eIdx_end = nodePointer[nIdx_end];				// ending edgeIdx of the current row_window.
	const unsigned num_TC_blocks = blockPartition[bid]; 			// number of TC_blocks of the current row_window.
	const unsigned dense_bound = numNodes * embedding_dim;

	__shared__ float sparse_A[BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[BLK_W];					// TC_block col to dense_tile row.
	// __shared__ float dense_X[dimTileNum * BLK_W * BLK_H];	// column-major dense tile [dimTileNum*BLK_H, BLK_W]
	extern __shared__ float dense_X[];

	wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
	wmma::fill_fragment(acc_frag, 0.0f);

	// Processing TC_blocks along the column dimension of Sparse A.
	for (unsigned i = 0; i < num_TC_blocks; i++){

		// Init A_colToX_row with dummy values.
		if (tid < BLK_W){
			sparse_AToX_index[tid] = numNodes + 1;
		}

		__syncthreads();

		// Init sparse_A with zero values.
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock){
			sparse_A[idx] = 0;
		}

		// Init dense_X with zero values.
		#pragma unroll
		for (unsigned idx = tid; idx < dimTileNum * BLK_W * BLK_H; idx += threadPerBlock){
			dense_X[idx] = 0;
		}

		// Initialize sparse_A by using BLK_H (16) threads from the warp-0.
		// currently fetch all neighbors of the current nodes.
		// then to see whether it can fit into current TC_block frame of column.		

		// FAN: important to understand. all threads check all data to check if they are in current tc block
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock){
			unsigned col = edgeToColumn[eIdx];
			if (i * BLK_W <= col && col < (i + 1) * BLK_W){			// if the edge in the current TC_block frame of column.
				unsigned row_local = edgeToRow[eIdx] % BLK_H;
				unsigned col_local = col % BLK_W;
				sparse_A[row_local * BLK_W + col_local] = 1;		// set the edge of the sparse_A.
				sparse_AToX_index[col_local] = edgeList[eIdx];		// record the mapping from sparse_A colId to rowId of dense_X.
			}		
		}

		__syncthreads();

		// Initialize dense_X by column-major store,
		// Threads of a warp for fetching a dense_X.
		// each warp identify by wid.
		if (wid < dimTileNum)
			#pragma unroll
			for (unsigned idx = laneid; idx < BLK_W * BLK_H; idx += warpSize){
				unsigned dense_rowIdx = sparse_AToX_index[idx % BLK_W];						// TC_block_col to dense_tile_row.
				unsigned dense_dimIdx = idx / BLK_W;										// dimIndex of the dense tile.
				unsigned source_idx = dense_rowIdx * embedding_dim + wid * BLK_H + dense_dimIdx;
				unsigned target_idx = wid * BLK_W * BLK_H + idx;
				// boundary test.
				if (source_idx >= dense_bound)
					dense_X[target_idx] = 0;
				else
					dense_X[target_idx] = input[source_idx];
			}

		__syncthreads();

		if (wid < dimTileNum)
		{
			wmma::load_matrix_sync(a_frag, sparse_A, BLK_W);   // BANK CONFLICT
			wmma::load_matrix_sync(b_frag, dense_X + wid * BLK_W * BLK_H, BLK_W);     // [dimTileNum*BLK_H, BLK_W] 
			#pragma unroll
			for (unsigned t = 0; t < a_frag.num_elements; t++) {
				a_frag.x[t] =  wmma::__float_to_tf32(a_frag.x[t]);
			}

			#pragma unroll
			for (unsigned t = 0; t < b_frag.num_elements; t++) {
				b_frag.x[t] =  wmma::__float_to_tf32(b_frag.x[t]);
			}
			// Perform the matrix multiplication.
			wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
		}
	}

	if (wid < dimTileNum)
		// Store the matrix to output matrix.
		// * Note * embedding dimension should be padded divisible by BLK_H for output correctness.
		wmma::store_matrix_sync(output + bid * BLK_H * embedding_dim + wid * BLK_H, acc_frag, embedding_dim, wmma::mem_row_major);
}


__global__ void spmm_forward_cuda_kernel_clock(
	const int * __restrict__ nodePointer,		// node pointer.
	const int *__restrict__ edgeList,			// edge list.
	const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
	const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
	const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output,							    // aggreAGNNed output feature matrix.
	clocktype* timer
) {
	 
	int smid = getSMId();
    clocktype tt = GlobalTimer64();
    const unsigned bid = blockIdx.x;								// block_index == row_window_index
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16.
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block.
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.

	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned nIdx_start = bid * BLK_H;					    // starting nodeIdx of current row_window.
	const unsigned nIdx_end = min((bid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.
	
	const unsigned eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
	const unsigned eIdx_end = nodePointer[nIdx_end];				// ending edgeIdx of the current row_window.
	const unsigned num_TC_blocks = blockPartition[bid]; 			// number of TC_blocks of the current row_window.
	const unsigned dense_bound = numNodes * embedding_dim;

	__shared__ float sparse_A[BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[BLK_W];					// TC_block col to dense_tile row.
	// __shared__ float dense_X[dimTileNum * BLK_W * BLK_H];	// column-major dense tile [dimTileNum*BLK_H, BLK_W]
	extern __shared__ float dense_X[];

	wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
	wmma::fill_fragment(acc_frag, 0.0f);

	// Processing TC_blocks along the column dimension of Sparse A.
	for (unsigned i = 0; i < num_TC_blocks; i++){

		// Init A_colToX_row with dummy values.
		if (tid < BLK_W){
			sparse_AToX_index[tid] = numNodes + 1;
		}

		__syncthreads();

		// Init sparse_A with zero values.
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock){
			sparse_A[idx] = 0;
		}

		// Init dense_X with zero values.
		#pragma unroll
		for (unsigned idx = tid; idx < dimTileNum * BLK_W * BLK_H; idx += threadPerBlock){
			dense_X[idx] = 0;
		}

		// Initialize sparse_A by using BLK_H (16) threads from the warp-0.
		// currently fetch all neighbors of the current nodes.
		// then to see whether it can fit into current TC_block frame of column.		

		// FAN: important to understand. all threads check all data to check if they are in current tc block
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock){
			unsigned col = edgeToColumn[eIdx];
			if (i * BLK_W <= col && col < (i + 1) * BLK_W){			// if the edge in the current TC_block frame of column.
				unsigned row_local = edgeToRow[eIdx] % BLK_H;
				unsigned col_local = col % BLK_W;
				sparse_A[row_local * BLK_W + col_local] = 1;		// set the edge of the sparse_A.
				sparse_AToX_index[col_local] = edgeList[eIdx];		// record the mapping from sparse_A colId to rowId of dense_X.
			}		
		}

		__syncthreads();

		// Initialize dense_X by column-major store,
		// Threads of a warp for fetching a dense_X.
		// each warp identify by wid.
		if (wid < dimTileNum)
			#pragma unroll
			for (unsigned idx = laneid; idx < BLK_W * BLK_H; idx += warpSize){
				unsigned dense_rowIdx = sparse_AToX_index[idx % BLK_W];						// TC_block_col to dense_tile_row.
				unsigned dense_dimIdx = idx / BLK_W;										// dimIndex of the dense tile.
				unsigned source_idx = dense_rowIdx * embedding_dim + wid * BLK_H + dense_dimIdx;
				unsigned target_idx = wid * BLK_W * BLK_H + idx;
				// boundary test.
				if (source_idx >= dense_bound)
					dense_X[target_idx] = 0;
				else
					dense_X[target_idx] = input[source_idx];
			}

		__syncthreads();

		if (wid < dimTileNum)
		{
			wmma::load_matrix_sync(a_frag, sparse_A, BLK_W);   // BANK CONFLICT
			wmma::load_matrix_sync(b_frag, dense_X + wid * BLK_W * BLK_H, BLK_W);     // [dimTileNum*BLK_H, BLK_W] 

			#pragma unroll
			for (unsigned t = 0; t < a_frag.num_elements; t++) {
				a_frag.x[t] =  wmma::__float_to_tf32(a_frag.x[t]);
			}

			#pragma unroll
			for (unsigned t = 0; t < b_frag.num_elements; t++) {
				b_frag.x[t] =  wmma::__float_to_tf32(b_frag.x[t]);
			}
			// Perform the matrix multiplication.
			wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
		}
	}
	if (wid < dimTileNum)
		// Store the matrix to output matrix.
		// * Note * embedding dimension should be padded divisible by BLK_H for output correctness.
		wmma::store_matrix_sync(output + bid * BLK_H * embedding_dim + wid * BLK_H, acc_frag, embedding_dim, wmma::mem_row_major);
	
	clocktype tt2 = GlobalTimer64();
    if(threadIdx.x == 0) {
        timer[3 * blockIdx.x] = tt;
        timer[3 * blockIdx.x + 1] = tt2;
        timer[3 * blockIdx.x + 2] = (clocktype)(smid);
    }
}

__global__ void spmm_forward_cuda_kernel_improved(
	const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 	// rowid of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
) {
    int bid = blockIdx.x;								// block_index == row_window_index
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__syncthreads();
	__shared__ float sparse_A[BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[BLK_W];					// TC_block col to dense_tile row.
	extern __shared__ float dense_X[];
	wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
	wmma::fill_fragment(acc_frag, 0.0f);
	unsigned off = wid * BLK_W * BLK_H;
	unsigned dense_rowIdx_off = (laneid % BLK_W);
	unsigned target_idx = off + laneid;
	unsigned target_idx1 = target_idx + 32;
	unsigned target_idx2 = target_idx1 + 32;
	unsigned target_idx3 = target_idx2 + 32;
	unsigned dense_dimIdx = laneid / BLK_W + wid * BLK_H;
	unsigned dense_dimIdx1 = dense_dimIdx + 4;
	unsigned dense_dimIdx2 = dense_dimIdx1 + 4;
	unsigned dense_dimIdx3 = dense_dimIdx2 + 4;
	for (unsigned i = lb; i < hb; i++) {
	  unsigned eIdx_start = TCblock_offset[i];			
	  unsigned eIdx_end = TCblock_offset[i+1];
	  unsigned sparse_AToX_idx_start = i * BLK_W;	   
	  if (tid < BLK_W) {
		sparse_AToX_index[tid] = (numNodes + 1) * embedding_dim;
	  }
	  __syncthreads();
	  #pragma unroll
	  for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock){
		sparse_A[idx] = 0;
	  }
	  #pragma unroll
	  for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock){
		  int id_local = (int)TCblocktile_id[eIdx];
		  sparse_A[id_local] = 1;		// set the edge of the sparse_A.	
	  }
	  #pragma unroll
	  for (unsigned eIdx = sparse_AToX_idx_start + tid; eIdx < sparse_AToX_idx_start + BLK_W; eIdx += threadPerBlock){
		sparse_AToX_index[tid] = sparse_AToX_idx[eIdx] * embedding_dim;	
	  }
	  __syncthreads();
	  if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[dense_rowIdx_off];						// TC_block_col to dense_tile_row.
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		// boundary test.
		if (source_idx >= dense_bound)
		  dense_X[target_idx] = 0;
		else
		  dense_X[target_idx] = input[source_idx];

		source_idx = dense_rowIdx + dense_dimIdx1;
		// boundary test.
		if (source_idx >= dense_bound)
		  dense_X[target_idx1] = 0;
		else
		  dense_X[target_idx1] = input[source_idx];
		
	    source_idx = dense_rowIdx + dense_dimIdx2;
		// boundary test.
		if (source_idx >= dense_bound)
		  dense_X[target_idx2] = 0;
		else
		  dense_X[target_idx2] = input[source_idx];
		
		source_idx = dense_rowIdx + dense_dimIdx3;
		// boundary test.
		if (source_idx >= dense_bound)
		  dense_X[target_idx3] = 0;
		else
		  dense_X[target_idx3] = input[source_idx];
	  }
	  __syncthreads();
	  if (wid < dimTileNum) {
			wmma::load_matrix_sync(a_frag, sparse_A, BLK_W);
			wmma::load_matrix_sync(b_frag, dense_X + off, BLK_W);
			#pragma unroll
			for (unsigned t = 0; t < a_frag.num_elements; t++) {
				a_frag.x[t] =  wmma::__float_to_tf32(a_frag.x[t]);
			}
			#pragma unroll
			for (unsigned t = 0; t < b_frag.num_elements; t++) {
				b_frag.x[t] =  wmma::__float_to_tf32(b_frag.x[t]);
			}
			wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
	  }
	}
	if (wid < dimTileNum)
	  wmma::store_matrix_sync(output + bid * BLK_H * embedding_dim + wid * BLK_H, acc_frag, embedding_dim, wmma::mem_row_major);
}


__global__ void spmm_forward_cuda_kernel_improved_with_value(
	const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// rowid of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA,
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
) {
    int bid = blockIdx.x;								// block_index == row_window_index
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__syncthreads();
	__shared__ float sparse_A[BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[BLK_W];					// TC_block col to dense_tile row.
	extern __shared__ float dense_X[];
	wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
	wmma::fill_fragment(acc_frag, 0.0f);
	unsigned off = wid * BLK_W * BLK_H;
	unsigned dense_rowIdx_off = (laneid % BLK_W);
	unsigned target_idx = off + laneid;
	unsigned target_idx1 = target_idx + 32;
	unsigned target_idx2 = target_idx1 + 32;
	unsigned target_idx3 = target_idx2 + 32;
	unsigned dense_dimIdx = laneid / BLK_W + wid * BLK_H;
	unsigned dense_dimIdx1 = dense_dimIdx + 4;
	unsigned dense_dimIdx2 = dense_dimIdx1 + 4;
	unsigned dense_dimIdx3 = dense_dimIdx2 + 4;
	for (unsigned i = lb; i < hb; i++) {
	  unsigned eIdx_start = TCblock_offset[i];			
	  unsigned eIdx_end = TCblock_offset[i+1];
	  unsigned sparse_AToX_idx_start = i * BLK_W;	   
	  if (tid < BLK_W) {
		sparse_AToX_index[tid] = (numNodes + 1) * embedding_dim;
	  }
	  __syncthreads();
	  #pragma unroll
	  for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock){
		sparse_A[idx] = 0;
	  }
	  #pragma unroll
	  for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock){
		  int id_local = (int)TCblocktile_id[eIdx];
		  sparse_A[id_local] = valuesA[eIdx];		// set the edge of the sparse_A.	
	  }
	  #pragma unroll
	  for (unsigned eIdx = sparse_AToX_idx_start + tid; eIdx < sparse_AToX_idx_start + BLK_W; eIdx += threadPerBlock){
		sparse_AToX_index[tid] = sparse_AToX_idx[eIdx] * embedding_dim;	
	  }
	  __syncthreads();
	  if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[dense_rowIdx_off];						// TC_block_col to dense_tile_row.
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		// boundary test.
		if (source_idx >= dense_bound)
		  dense_X[target_idx] = 0;
		else
		  dense_X[target_idx] = input[source_idx];

		source_idx = dense_rowIdx + dense_dimIdx1;
		// boundary test.
		if (source_idx >= dense_bound)
		  dense_X[target_idx1] = 0;
		else
		  dense_X[target_idx1] = input[source_idx];
		
	    source_idx = dense_rowIdx + dense_dimIdx2;
		// boundary test.
		if (source_idx >= dense_bound)
		  dense_X[target_idx2] = 0;
		else
		  dense_X[target_idx2] = input[source_idx];
		
		source_idx = dense_rowIdx + dense_dimIdx3;
		// boundary test.
		if (source_idx >= dense_bound)
		  dense_X[target_idx3] = 0;
		else
		  dense_X[target_idx3] = input[source_idx];
	  }
	  __syncthreads();
	  if (wid < dimTileNum) {
			wmma::load_matrix_sync(a_frag, sparse_A, BLK_W);
			wmma::load_matrix_sync(b_frag, dense_X + off, BLK_W);
			#pragma unroll
			for (unsigned t = 0; t < a_frag.num_elements; t++) {
				a_frag.x[t] =  wmma::__float_to_tf32(a_frag.x[t]);
			}
			#pragma unroll
			for (unsigned t = 0; t < b_frag.num_elements; t++) {
				b_frag.x[t] =  wmma::__float_to_tf32(b_frag.x[t]);
			}
			wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
	  }
	}
	if (wid < dimTileNum)
	  wmma::store_matrix_sync(output + bid * BLK_H * embedding_dim + wid * BLK_H, acc_frag, embedding_dim, wmma::mem_row_major);
}


__global__ void spmm_forward_cuda_kernel_improved_with_value_clock(
	const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// rowid of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA,
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output,							    // output feature matrix.
	clocktype* timer
) {
	int smid = getSMId();
    clocktype tt = GlobalTimer64();
    int bid = blockIdx.x;								// block_index == row_window_index
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__syncthreads();
	__shared__ float sparse_A[BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[BLK_W];					// TC_block col to dense_tile row.
	extern __shared__ float dense_X[];
	wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
	wmma::fill_fragment(acc_frag, 0.0f);
	unsigned off = wid * BLK_W * BLK_H;
	unsigned dense_rowIdx_off = (laneid % BLK_W);
	unsigned target_idx = off + laneid;
	unsigned target_idx1 = target_idx + 32;
	unsigned target_idx2 = target_idx1 + 32;
	unsigned target_idx3 = target_idx2 + 32;
	unsigned dense_dimIdx = laneid / BLK_W + wid * BLK_H;
	unsigned dense_dimIdx1 = dense_dimIdx + 4;
	unsigned dense_dimIdx2 = dense_dimIdx1 + 4;
	unsigned dense_dimIdx3 = dense_dimIdx2 + 4;
	for (unsigned i = lb; i < hb; i++) {
	  unsigned eIdx_start = TCblock_offset[i];			
	  unsigned eIdx_end = TCblock_offset[i+1];
	  unsigned sparse_AToX_idx_start = i * BLK_W;	   
	  if (tid < BLK_W) {
		sparse_AToX_index[tid] = (numNodes + 1) * embedding_dim;
	  }
	  __syncthreads();
	  #pragma unroll
	  for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock){
		sparse_A[idx] = 0;
	  }
	  #pragma unroll
	  for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock){
		  int id_local = (int)TCblocktile_id[eIdx];
		  sparse_A[id_local] = valuesA[eIdx];		// set the edge of the sparse_A.	
	  }
	  #pragma unroll
	  for (unsigned eIdx = sparse_AToX_idx_start + tid; eIdx < sparse_AToX_idx_start + BLK_W; eIdx += threadPerBlock){
		sparse_AToX_index[tid] = sparse_AToX_idx[eIdx] * embedding_dim;	
	  }
	  __syncthreads();
	  if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[dense_rowIdx_off];						// TC_block_col to dense_tile_row.
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		// boundary test.
		if (source_idx >= dense_bound)
		  dense_X[target_idx] = 0;
		else
		  dense_X[target_idx] = input[source_idx];

		source_idx = dense_rowIdx + dense_dimIdx1;
		// boundary test.
		if (source_idx >= dense_bound)
		  dense_X[target_idx1] = 0;
		else
		  dense_X[target_idx1] = input[source_idx];
		
	    source_idx = dense_rowIdx + dense_dimIdx2;
		// boundary test.
		if (source_idx >= dense_bound)
		  dense_X[target_idx2] = 0;
		else
		  dense_X[target_idx2] = input[source_idx];
		
		source_idx = dense_rowIdx + dense_dimIdx3;
		// boundary test.
		if (source_idx >= dense_bound)
		  dense_X[target_idx3] = 0;
		else
		  dense_X[target_idx3] = input[source_idx];
	  }
	  __syncthreads();
	  if (wid < dimTileNum) {
			wmma::load_matrix_sync(a_frag, sparse_A, BLK_W);
			wmma::load_matrix_sync(b_frag, dense_X + off, BLK_W);
			#pragma unroll
			for (unsigned t = 0; t < a_frag.num_elements; t++) {
				a_frag.x[t] =  wmma::__float_to_tf32(a_frag.x[t]);
			}
			#pragma unroll
			for (unsigned t = 0; t < b_frag.num_elements; t++) {
				b_frag.x[t] =  wmma::__float_to_tf32(b_frag.x[t]);
			}
			wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
	  }
	}
	if (wid < dimTileNum)
	  wmma::store_matrix_sync(output + bid * BLK_H * embedding_dim + wid * BLK_H, acc_frag, embedding_dim, wmma::mem_row_major);
	
	clocktype tt2 = GlobalTimer64();
	if(threadIdx.x == 0) {
		timer[3 * blockIdx.x] = tt;
		timer[3 * blockIdx.x + 1] = tt2;
		timer[3 * blockIdx.x + 2] = (clocktype)(smid);
	}
}

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1(
	const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
) {
    int bid = blockIdx.x;								// block_index == row_window_index
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[BLK_W];					// TC_block col to dense_tile row.
	unsigned off = wid * BLK_W * BLK_H;
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) + wid * BLK_H;
	unsigned dense_dimIdx1 = dense_dimIdx + 8;
	uint32_t group_id = (laneid >> 2);
    uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	int shuffle_idx = (laneid/4) + (laneid%4)*8;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
    uint32_t frag_B[4]; // 8 * 8 * 2  / 32 = 4
	float frag_D[8] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	for (unsigned j = lb; j < hb; j++) {
	  unsigned eIdx_start = TCblock_offset[j];			
	  unsigned eIdx_end = TCblock_offset[j+1];
	  unsigned sparse_AToX_idx_start = j * BLK_W;	   
	  if (tid < BLK_W) {
		sparse_AToX_index[tid] = (numNodes + 1) * embedding_dim;
	  }
	  #pragma unroll
	  for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		sparse_A[idx] = 0.0;
	  }
	  __syncthreads();
	  #pragma unroll
	  for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock){
		  int id_local = (int)TCblocktile_id[eIdx];
		  sparse_A[id_local] = 1.0;		// set the edge of the sparse_A.	
	  }
	  __syncthreads();
	  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[sparse_A_idx]));
	  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[sparse_A_idx1]));
	  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[sparse_A_idx2]));
	  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[sparse_A_idx3]));
	  #pragma unroll
	  for (unsigned eIdx = sparse_AToX_idx_start + tid; eIdx < sparse_AToX_idx_start + BLK_W; eIdx += threadPerBlock) {
		sparse_AToX_index[tid] = sparse_AToX_idx[eIdx] * embedding_dim;	
	  }
	  __syncthreads();
	  if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[dense_rowIdx_off];						// TC_block_col to dense_tile_row.
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(input[source_idx]));
		asm volatile(
			"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
			: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
			: "r"(frag_A[0]), "r"(frag_A[1]), 
			  "r"(frag_B[0]), 
			  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
		);
		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(input[source_idx]));
		asm volatile(
			"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
			: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
			: "r"(frag_A[0]), "r"(frag_A[1]),
			  "r"(frag_B[1]), 
			  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
		  );
		dense_rowIdx = sparse_AToX_index[dense_rowIdx_off1];						// TC_block_col to dense_tile_row.
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(input[source_idx]));
		asm volatile(
			"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
			: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
			: "r"(frag_A[2]), "r"(frag_A[3]), 
			  "r"(frag_B[2]), 
			  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
		);
		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(input[source_idx]));
		asm volatile(
			"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
			: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
			: "r"(frag_A[2]), "r"(frag_A[3]), 
			  "r"(frag_B[3]), 
			  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
		  );
	  }
	}
	uint32_t o_off1 = bid * BLK_H * embedding_dim + wid * BLK_H;
	uint32_t o_off2 = o_off1 + 8;
	if (wid < dimTileNum)
	#pragma unroll
	for(int i = 0; i < 4; i++) {
		uint32_t row_d = 0;
		if( i < 2 ) {
			row_d = group_id;
		} else {
			row_d = group_id + 8;
		}
		uint32_t col_d = (tid_in_group * 2) + (i & 0x1);
		uint32_t off = row_d * embedding_dim + col_d;
		output[o_off1 + off] = frag_D[i];
		output[o_off2 + off] = frag_D[i + 4];
	}
}

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value(
	const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA, 		
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
) {
    int bid = blockIdx.x;								// block_index == row_window_index
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[BLK_W];					// TC_block col to dense_tile row.
	unsigned off = wid * BLK_W * BLK_H;
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) + wid * BLK_H;
	unsigned dense_dimIdx1 = dense_dimIdx + 8;
	uint32_t group_id = (laneid >> 2);
    uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	int shuffle_idx = (laneid/4) + (laneid%4)*8;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
    uint32_t frag_B[4]; // 8 * 8 * 2  / 32 = 4
	float frag_D[8] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int s_ptr = __cvta_generic_to_shared(sparse_A);
	for (unsigned j = lb; j < hb; j++) {
	  unsigned eIdx_start = TCblock_offset[j];			
	  unsigned eIdx_end = TCblock_offset[j+1];
	  unsigned sparse_AToX_idx_start = j * BLK_W;	   
	  if (tid < BLK_W) {
		sparse_AToX_index[tid] = (numNodes + 1) * embedding_dim;
	  }
	  #pragma unroll
	  for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		sparse_A[idx] = 0.0;
	  }
	  __syncthreads();
	  #pragma unroll
	  for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  sparse_A[(int)TCblocktile_id[eIdx]] = valuesA[eIdx];		// set the edge of the sparse_A.	
		//   int id_local = (((int)TCblocktile_id[eIdx])<<2);
		//   asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(s_ptr + id_local), "l"(valuesA+eIdx));	  
	  }
	//   asm ("cp.async.commit_group;\n"::);
	//   asm ("cp.async.wait_group 0;\n" ::);
	  __syncthreads();
	  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[sparse_A_idx]));
	  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[sparse_A_idx1]));
	  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[sparse_A_idx2]));
	  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[sparse_A_idx3]));
	  #pragma unroll
	  for (unsigned eIdx = sparse_AToX_idx_start + tid; eIdx < sparse_AToX_idx_start + BLK_W; eIdx += threadPerBlock) {
		sparse_AToX_index[tid] = sparse_AToX_idx[eIdx] * embedding_dim;	
	  }
	  __syncthreads();
	  if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[dense_rowIdx_off];						// TC_block_col to dense_tile_row.
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(input[source_idx]));
		asm volatile(
			"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
			: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
			: "r"(frag_A[0]), "r"(frag_A[1]), 
			  "r"(frag_B[0]), 
			  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
		);
		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(input[source_idx]));
		asm volatile(
			"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
			: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
			: "r"(frag_A[0]), "r"(frag_A[1]),
			  "r"(frag_B[1]), 
			  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
		  );
		dense_rowIdx = sparse_AToX_index[dense_rowIdx_off1];						// TC_block_col to dense_tile_row.
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(input[source_idx]));
		asm volatile(
			"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
			: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
			: "r"(frag_A[2]), "r"(frag_A[3]), 
			  "r"(frag_B[2]), 
			  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
		);
		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(input[source_idx]));
		asm volatile(
			"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
			: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
			: "r"(frag_A[2]), "r"(frag_A[3]), 
			  "r"(frag_B[3]), 
			  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
		  );
	  }
	}
	uint32_t o_off1 = bid * BLK_H * embedding_dim + wid * BLK_H;
	uint32_t o_off2 = o_off1 + 8;
	if (wid < dimTileNum)
	#pragma unroll
	for(int i = 0; i < 4; i++) {
		uint32_t row_d = 0;
		if( i < 2 ) {
			row_d = group_id;
		} else {
			row_d = group_id + 8;
		}
		uint32_t col_d = (tid_in_group * 2) + (i & 0x1);
		uint32_t off = row_d * embedding_dim + col_d;
		output[o_off1 + off] = frag_D[i];
		output[o_off2 + off] = frag_D[i + 4];
	}
}


__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_split(
	const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA, 		
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
) {
    int bid = blockIdx.x;								// block_index == row_window_index
	int off_y = (blockIdx.y << 7);								
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	unsigned off = wid * BLK_W * BLK_H;
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) + wid * BLK_H + off_y;
	unsigned dense_dimIdx1 = dense_dimIdx + 8;
	uint32_t group_id = (laneid >> 2);
    uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	int shuffle_idx = (laneid/4) + (laneid%4)*8;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
    uint32_t frag_B[4]; // 8 * 8 * 2  / 32 = 4
	float frag_D[8] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
// pre loop
    {
		if (tid < BLK_W) {
		  sparse_AToX_index[tid] = (numNodes + 1) * embedding_dim;
		}
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  sparse_A[TCblocktile_id[eIdx]] = valuesA[eIdx];		// set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
		  sparse_AToX_index[tid] = sparse_AToX_idx[tid] * embedding_dim;	
		}
		__syncthreads();
	}
//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
        int smem_sel_next = ( (j-lb - 1) & 1) ^ 1;

		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off];
			unsigned source_idx = dense_rowIdx + dense_dimIdx;
			if (source_idx >= dense_bound)
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			else
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(input[source_idx]));
	
			source_idx = dense_rowIdx + dense_dimIdx1;
			if (source_idx >= dense_bound)
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			else
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(input[source_idx]));
	
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1];
			source_idx = dense_rowIdx + dense_dimIdx;
			if (source_idx >= dense_bound)
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			else
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(input[source_idx]));
	
			source_idx = dense_rowIdx + dense_dimIdx1;
			if (source_idx >= dense_bound)
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			else
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(input[source_idx]));
		}
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	    eIdx_start = TCblock_offset[j];			
	    eIdx_end = TCblock_offset[j+1];
	    unsigned sparse_AToX_idx_start = j * BLK_W;	   
	    if (tid < BLK_W) {
		  sparse_AToX_index[(smem_sel_next << 3) + tid] = (numNodes + 1) * embedding_dim;
	    }
	    #pragma unroll
	    for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[(smem_sel_next << 7) + idx] = 0.0;
	    }
	    __syncthreads();
	    #pragma unroll
	    for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  int id_local = (((int)TCblocktile_id[eIdx])<<2);
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa_ptr + id_local + (smem_sel_next << 9)), "l"(valuesA+eIdx));	  
	    }
		if (tid < BLK_W) {	
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(sparse_AToX_idx+sparse_AToX_idx_start+tid));	
		}


		// asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		// asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		// asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		// asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[0]), "r"(frag_B[2]), 
            "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[1]), "r"(frag_B[3]), 
            "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
        );

	    asm ("cp.async.commit_group;\n"::);
	    asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}

//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off];
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(input[source_idx]));

		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(input[source_idx]));

		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1];
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(input[source_idx]));

		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(input[source_idx]));
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[0]), "r"(frag_B[2]), 
		  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[1]), "r"(frag_B[3]), 
		  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
	  );




	uint32_t o_off1 = bid * BLK_H * embedding_dim + wid * BLK_H + off_y;
	uint32_t o_off2 = o_off1 + 8;
	if (wid < dimTileNum)
	#pragma unroll
	for(int i = 0; i < 4; i++) {
		uint32_t row_d = 0;
		if( i < 2 ) {
			row_d = group_id;
		} else {
			row_d = group_id + 8;
		}
		uint32_t col_d = (tid_in_group * 2) + (i & 0x1);
		uint32_t off = row_d * embedding_dim + col_d;
		output[o_off1 + off] = frag_D[i];
		output[o_off2 + off] = frag_D[i + 4];
	}
}


__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer(
	const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA, 		
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
) {
    int bid = blockIdx.x;								// block_index == row_window_index
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) + wid * BLK_H;
	unsigned dense_dimIdx1 = dense_dimIdx + 8;
	uint32_t group_id = (laneid >> 2);
    uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	int shuffle_idx = (laneid/4) + (laneid%4)*8;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
    uint32_t frag_B[4]; // 8 * 8 * 2  / 32 = 4
	float frag_D[8] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
// pre loop
    {
		unsigned sparse_AToX_idx_start = lb * BLK_W;	 
		// if (tid < BLK_W) {
		//   sparse_AToX_index[tid] = numNodes + 1;
		// }
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  sparse_A[(int)TCblocktile_id[eIdx]] = valuesA[eIdx];		// set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
		  sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
		}
		__syncthreads();
	}
//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
        int smem_sel_next = ( (j-lb - 1) & 1) ^ 1;

		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;
			unsigned source_idx = dense_rowIdx + dense_dimIdx;
			if (source_idx >= dense_bound)
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			else
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(input[source_idx]));
	
			source_idx = dense_rowIdx + dense_dimIdx1;
			if (source_idx >= dense_bound)
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			else
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(input[source_idx]));
	
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;
			source_idx = dense_rowIdx + dense_dimIdx;
			if (source_idx >= dense_bound)
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			else
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(input[source_idx]));
	
			source_idx = dense_rowIdx + dense_dimIdx1;
			if (source_idx >= dense_bound)
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			else
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(input[source_idx]));
		}
	    eIdx_start = TCblock_offset[j];			
	    eIdx_end = TCblock_offset[j+1];
	    unsigned sparse_AToX_idx_start = j * BLK_W;	   
	    #pragma unroll
	    for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[(smem_sel_next << 7) + idx] = 0.0;
	    }
	    __syncthreads();
	    #pragma unroll
	    for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  int id_local = (((int)TCblocktile_id[eIdx])<<2);
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa_ptr + id_local + (smem_sel_next << 9)), "l"(&valuesA[eIdx]));	  
	    }
		if (tid < BLK_W) {	
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(&sparse_AToX_idx[sparse_AToX_idx_start + tid]));	
		}

		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[0]), "r"(frag_B[2]), 
            "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[1]), "r"(frag_B[3]), 
            "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
        );

	    asm ("cp.async.commit_group;\n"::);
	    asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}

//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(input[source_idx]));

		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(input[source_idx]));

		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(input[source_idx]));

		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(input[source_idx]));
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[0]), "r"(frag_B[2]), 
		  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[1]), "r"(frag_B[3]), 
		  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
	  );


	uint32_t o_off1 = bid * BLK_H * embedding_dim + wid * BLK_H;
	uint32_t o_off2 = o_off1 + 8;
	if (wid < dimTileNum)
	#pragma unroll
	for(int i = 0; i < 4; i++) {
		uint32_t row_d = 0;
		if( i < 2 ) {
			row_d = group_id;
		} else {
			row_d = group_id + 8;
		}
		uint32_t col_d = (tid_in_group * 2) + (i & 0x1);
		uint32_t off = row_d * embedding_dim + col_d;
		output[o_off1 + off] = frag_D[i];
		output[o_off2 + off] = frag_D[i + 4];
	}
}


__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_double_buffer(
	const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
) {
    int bid = blockIdx.x;								// block_index == row_window_index
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) + wid * BLK_H;
	unsigned dense_dimIdx1 = dense_dimIdx + 8;
	uint32_t group_id = (laneid >> 2);
    uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	int shuffle_idx = (laneid/4) + (laneid%4)*8;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
    uint32_t frag_B[4]; // 8 * 8 * 2  / 32 = 4
	float frag_D[8] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
// pre loop
    {
		unsigned sparse_AToX_idx_start = lb * BLK_W;	 
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  sparse_A[(int)TCblocktile_id[eIdx]] = 1.0;		// set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
		  sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
		}
		__syncthreads();
	}
//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
        int smem_sel_next = ( (j-lb - 1) & 1) ^ 1;
		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;
			unsigned source_idx = dense_rowIdx + dense_dimIdx;
			if (source_idx >= dense_bound)
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			else
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(input[source_idx]));
	
			source_idx = dense_rowIdx + dense_dimIdx1;
			if (source_idx >= dense_bound)
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			else
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(input[source_idx]));
	
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;
			source_idx = dense_rowIdx + dense_dimIdx;
			if (source_idx >= dense_bound)
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			else
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(input[source_idx]));
	
			source_idx = dense_rowIdx + dense_dimIdx1;
			if (source_idx >= dense_bound)
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			else
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(input[source_idx]));
		}
	    eIdx_start = TCblock_offset[j];			
	    eIdx_end = TCblock_offset[j+1];
	    unsigned sparse_AToX_idx_start = j * BLK_W;	   
	    #pragma unroll
	    for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[(smem_sel_next << 7) + idx] = 0.0;
	    }
	    __syncthreads();
		if (tid < BLK_W) {	
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(&sparse_AToX_idx[sparse_AToX_idx_start + tid]));	
		}
	    #pragma unroll
	    for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  sparse_A[(smem_sel_next << 7) + TCblocktile_id[eIdx]] = 1.0;	  
	    }
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[0]), "r"(frag_B[2]), 
            "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[1]), "r"(frag_B[3]), 
            "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
        );

	    asm ("cp.async.commit_group;\n"::);
	    asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}

//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(input[source_idx]));

		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(input[source_idx]));

		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(input[source_idx]));

		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(input[source_idx]));
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[0]), "r"(frag_B[2]), 
		  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[1]), "r"(frag_B[3]), 
		  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
	  );


	uint32_t o_off1 = bid * BLK_H * embedding_dim + wid * BLK_H;
	uint32_t o_off2 = o_off1 + 8;
	if (wid < dimTileNum)
	#pragma unroll
	for(int i = 0; i < 4; i++) {
		uint32_t row_d = 0;
		if( i < 2 ) {
			row_d = group_id;
		} else {
			row_d = group_id + 8;
		}
		uint32_t col_d = (tid_in_group * 2) + (i & 0x1);
		uint32_t off = row_d * embedding_dim + col_d;
		output[o_off1 + off] = frag_D[i];
		output[o_off2 + off] = frag_D[i + 4];
	}
}



#define FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float2(
	const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 	// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA, 		
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	float *input,		                        // input feature matrix.
	float *output							    // output feature matrix.
) {
    int bid = blockIdx.x;								// block_index == row_window_index
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	unsigned off = wid * BLK_W * BLK_H;
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) * 2 + wid * BLK_H;
	// unsigned dense_dimIdx1 = dense_dimIdx + 8;
	uint32_t group_id = (laneid >> 2);
    uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	int shuffle_idx = (laneid/4) + (laneid%4)*8;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
    uint32_t frag_B[4]; // 8 * 8 * 2  / 32 = 4
	float frag_D[8] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
// pre loop
    {
		unsigned sparse_AToX_idx_start = lb * BLK_W;	  
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  sparse_A[TCblocktile_id[eIdx]] = valuesA[eIdx];		// set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
		  sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
		}
		__syncthreads();
	}
//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
        int smem_sel_next = ( (j-lb - 1) & 1) ^ 1;
		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;
			unsigned source_idx = dense_rowIdx + dense_dimIdx;
			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			} else {
				float2 t = FLOAT2(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
			}
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;
			source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			} else {
				float2 t = FLOAT2(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.y));
			}
		}

	    eIdx_start = TCblock_offset[j];			
	    eIdx_end = TCblock_offset[j+1];
	    unsigned sparse_AToX_idx_start = j * BLK_W;	   
	    #pragma unroll
	    for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[(smem_sel_next << 7) + idx] = 0.0;
	    }
	    __syncthreads();
	    #pragma unroll
	    for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  int id_local = (((int)TCblocktile_id[eIdx])<<2);
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa_ptr + id_local + (smem_sel_next << 9)), "l"(valuesA+eIdx));	  
	    }
		if (tid < BLK_W) {	
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(sparse_AToX_idx + sparse_AToX_idx_start + tid));	
		}


		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[0]), "r"(frag_B[2]), 
            "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[1]), "r"(frag_B[3]), 
            "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
        );

	    asm ("cp.async.commit_group;\n"::);
	    asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}

//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
		} else {
			float2 t = FLOAT2(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
		}
		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		} else {
			float2 t = FLOAT2(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.y));
		}
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[0]), "r"(frag_B[2]), 
		  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[1]), "r"(frag_B[3]), 
		  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
	  );

	uint32_t o_off = bid * BLK_H * embedding_dim + wid * BLK_H;
	if (wid < dimTileNum)
	#pragma unroll
	for(int i = 0; i < 4; i++) {
		uint32_t row_d = 0;
		if( i < 2 ) {
			row_d = group_id;
		} else {
			row_d = group_id + 8;
		}
		uint32_t col_d = (tid_in_group<<2) + ((i & 0x1)<<1);
		uint32_t off = row_d * embedding_dim + col_d;
		output[o_off + off] = frag_D[i];
		output[o_off + off + 1] = frag_D[i + 4];
	}
}

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_double_buffer_float2(
	const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.		
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	float *input,		    // input feature matrix.
	float *output							    // output feature matrix.
) {
    int bid = blockIdx.x;								// block_index == row_window_index
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	unsigned off = wid * BLK_W * BLK_H;
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) * 2 + wid * BLK_H;
	// unsigned dense_dimIdx1 = dense_dimIdx + 8;
	uint32_t group_id = (laneid >> 2);
    uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	int shuffle_idx = (laneid/4) + (laneid%4)*8;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
    uint32_t frag_B[4]; // 8 * 8 * 2  / 32 = 4
	float frag_D[8] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
// pre loop
    {
		unsigned sparse_AToX_idx_start = lb * BLK_W;	  
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  sparse_A[TCblocktile_id[eIdx]] = 1.0;		// set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
		  sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
		}
		__syncthreads();
	}
//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
        int smem_sel_next = ( (j-lb - 1) & 1) ^ 1;
		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;
			unsigned source_idx = dense_rowIdx + dense_dimIdx;
			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			} else {
				float2 t = FLOAT2(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
			}
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;
			source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			} else {
				float2 t = FLOAT2(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.y));
			}
		}

	    eIdx_start = TCblock_offset[j];			
	    eIdx_end = TCblock_offset[j+1];
	    unsigned sparse_AToX_idx_start = j * BLK_W;	   
	    #pragma unroll
	    for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[(smem_sel_next << 7) + idx] = 0.0;
	    }
	    __syncthreads();
		if (tid < BLK_W) {	
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(sparse_AToX_idx + sparse_AToX_idx_start + tid));	
		}
	    #pragma unroll
	    for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  sparse_A[(smem_sel_next << 7) + TCblocktile_id[eIdx]] = 1.0;	  
	    }
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[0]), "r"(frag_B[2]), 
            "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[1]), "r"(frag_B[3]), 
            "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
        );

	    asm ("cp.async.commit_group;\n"::);
	    asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}

//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
		} else {
			float2 t = FLOAT2(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
		}
		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		} else {
			float2 t = FLOAT2(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.y));
		}
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[0]), "r"(frag_B[2]), 
		  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[1]), "r"(frag_B[3]), 
		  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
	  );

	uint32_t o_off = bid * BLK_H * embedding_dim + wid * BLK_H;
	if (wid < dimTileNum)
	#pragma unroll
	for(int i = 0; i < 4; i++) {
		uint32_t row_d = 0;
		if( i < 2 ) {
			row_d = group_id;
		} else {
			row_d = group_id + 8;
		}
		uint32_t col_d = (tid_in_group<<2) + ((i & 0x1)<<1);
		uint32_t off = row_d * embedding_dim + col_d;
		output[o_off + off] = frag_D[i];
		output[o_off + off + 1] = frag_D[i + 4];
	}
}


__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float2_split(
	const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA, 		
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	float *input,		    // input feature matrix.
	float *output							    // output feature matrix.
) {
    int bid = blockIdx.x;								// block_index == row_window_index
	int off_y = (blockIdx.y << 6);
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	unsigned off = wid * BLK_W * BLK_H;
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) * 2 + wid * BLK_H + off_y;
	// unsigned dense_dimIdx1 = dense_dimIdx + 8;
	uint32_t group_id = (laneid >> 2);
    uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	int shuffle_idx = (laneid/4) + (laneid%4)*8;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
    uint32_t frag_B[4]; // 8 * 8 * 2  / 32 = 4
	float frag_D[8] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
// pre loop
    {
		unsigned sparse_AToX_idx_start = lb * BLK_W;	  
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  sparse_A[TCblocktile_id[eIdx]] = valuesA[eIdx];		// set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
		  sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
		}
		__syncthreads();
	}
//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
        int smem_sel_next = ( (j-lb - 1) & 1) ^ 1;
		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;
			unsigned source_idx = dense_rowIdx + dense_dimIdx;
			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			} else {
				float2 t = FLOAT2(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
			}
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;
			source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			} else {
				float2 t = FLOAT2(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.y));
			}
		}

	    eIdx_start = TCblock_offset[j];			
	    eIdx_end = TCblock_offset[j+1];
	    unsigned sparse_AToX_idx_start = j * BLK_W;	   
	    #pragma unroll
	    for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[(smem_sel_next << 7) + idx] = 0.0;
	    }
	    __syncthreads();
	    #pragma unroll
	    for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  int id_local = (((int)TCblocktile_id[eIdx])<<2);
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa_ptr + id_local + (smem_sel_next << 9)), "l"(valuesA+eIdx));	  
	    }
		if (tid < BLK_W) {	
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(sparse_AToX_idx + sparse_AToX_idx_start + tid));	
		}


		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[0]), "r"(frag_B[2]), 
            "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[1]), "r"(frag_B[3]), 
            "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
        );

	    asm ("cp.async.commit_group;\n"::);
	    asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}

//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;  // TC_block_col to dense_tile_row.
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
		} else {
			float2 t = FLOAT2(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
		}
		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;  // TC_block_col to dense_tile_row.
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		} else {
			float2 t = FLOAT2(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.y));
		}
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[0]), "r"(frag_B[2]), 
		  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[1]), "r"(frag_B[3]), 
		  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
	  );

	uint32_t o_off = bid * BLK_H * embedding_dim + wid * BLK_H + off_y;
	if (wid < dimTileNum)
	#pragma unroll
	for(int i = 0; i < 4; i++) {
		uint32_t row_d = 0;
		if( i < 2 ) {
			row_d = group_id;
		} else {
			row_d = group_id + 8;
		}
		uint32_t col_d = (tid_in_group<<2) + ((i & 0x1)<<1);
		uint32_t off = row_d * embedding_dim + col_d;
		output[o_off + off] = frag_D[i];
		output[o_off + off + 1] = frag_D[i + 4];
	}
}

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_double_buffer_float2_split(
	const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.		
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	float *input,		    // input feature matrix.
	float *output							    // output feature matrix.
) {
    int bid = blockIdx.x;								// block_index == row_window_index
	int off_y = (blockIdx.y << 6);
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	unsigned off = wid * BLK_W * BLK_H;
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) * 2 + wid * BLK_H + off_y;
	// unsigned dense_dimIdx1 = dense_dimIdx + 8;
	uint32_t group_id = (laneid >> 2);
    uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	int shuffle_idx = (laneid/4) + (laneid%4)*8;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
    uint32_t frag_B[4]; // 8 * 8 * 2  / 32 = 4
	float frag_D[8] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
// pre loop
    {
		unsigned sparse_AToX_idx_start = lb * BLK_W;	  
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  sparse_A[TCblocktile_id[eIdx]] = 1.0;		// set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
		  sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
		}
		__syncthreads();
	}
//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
        int smem_sel_next = ( (j-lb - 1) & 1) ^ 1;
		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;  // TC_block_col to dense_tile_row.
			unsigned source_idx = dense_rowIdx + dense_dimIdx;
			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			} else {
				float2 t = FLOAT2(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
			}
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;  // TC_block_col to dense_tile_row.
			source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			} else {
				float2 t = FLOAT2(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.y));
			}
		}

	    eIdx_start = TCblock_offset[j];			
	    eIdx_end = TCblock_offset[j+1];
	    unsigned sparse_AToX_idx_start = j * BLK_W;	   
	    #pragma unroll
	    for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[(smem_sel_next << 7) + idx] = 0.0;
	    }
	    __syncthreads();
		if (tid < BLK_W) {	
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(sparse_AToX_idx + sparse_AToX_idx_start + tid));	
		}
	    #pragma unroll
	    for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  sparse_A[(smem_sel_next << 7) + TCblocktile_id[eIdx]] = 1.0;	  
	    }
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[0]), "r"(frag_B[2]), 
            "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[1]), "r"(frag_B[3]), 
            "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
        );

	    asm ("cp.async.commit_group;\n"::);
	    asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}

//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;  // TC_block_col to dense_tile_row.
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
		} else {
			float2 t = FLOAT2(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
		}
		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;  // TC_block_col to dense_tile_row.
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		} else {
			float2 t = FLOAT2(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.y));
		}
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[0]), "r"(frag_B[2]), 
		  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[1]), "r"(frag_B[3]), 
		  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
	  );

	uint32_t o_off = bid * BLK_H * embedding_dim + wid * BLK_H + off_y;
	if (wid < dimTileNum)
	#pragma unroll
	for(int i = 0; i < 4; i++) {
		uint32_t row_d = 0;
		if( i < 2 ) {
			row_d = group_id;
		} else {
			row_d = group_id + 8;
		}
		uint32_t col_d = (tid_in_group<<2) + ((i & 0x1)<<1);
		uint32_t off = row_d * embedding_dim + col_d;
		output[o_off + off] = frag_D[i];
		output[o_off + off + 1] = frag_D[i + 4];
	}
}


#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float4(
	const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA, 		
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	float *input,		    // input feature matrix.
	float *output							    // output feature matrix.
) {
    int bid = blockIdx.x;								// block_index == row_window_index
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / 32;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) * 4 + wid * 32;
	uint32_t group_id = (laneid >> 2);
    uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
    uint32_t frag_B[8]; // 8 * 8 * 2  / 32 = 4
	float frag_D[16] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
// pre loop
    {
		unsigned sparse_AToX_idx_start = lb * BLK_W;	
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  sparse_A[TCblocktile_id[eIdx]] = valuesA[eIdx];		// set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
		  sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
		}
		__syncthreads();
	}
//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
        int smem_sel_next = ( (j - lb - 1) & 1) ^ 1;
		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;  // TC_block_col to dense_tile_row.
			unsigned source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			} else {
				float4 t = FLOAT4(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.w));
			}
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;  // TC_block_col to dense_tile_row.
			source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(z));
			} else {
				float4 t = FLOAT4(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(t.y));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(t.z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(t.w));
			}

		}

	    eIdx_start = TCblock_offset[j];			
	    eIdx_end = TCblock_offset[j+1];
	    unsigned sparse_AToX_idx_start = j * BLK_W;	   

	    #pragma unroll
	    for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[(smem_sel_next << 7) + idx] = 0.0;
	    }
	    __syncthreads();
	    #pragma unroll
	    for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  int id_local = (((int)TCblocktile_id[eIdx])<<2);
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa_ptr + id_local + (smem_sel_next << 9)), "l"(valuesA+eIdx));	  
	    }
		if (tid < BLK_W) {	
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(sparse_AToX_idx+sparse_AToX_idx_start+tid));	
		}

		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[0]), "r"(frag_B[4]), 
            "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[1]), "r"(frag_B[5]), 
            "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
        );
		asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[8]), "=f"(frag_D[9]), "=f"(frag_D[10]), "=f"(frag_D[11])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[2]), "r"(frag_B[6]), 
            "f"(frag_D[8]), "f"(frag_D[9]), "f"(frag_D[10]), "f"(frag_D[11])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[12]), "=f"(frag_D[13]), "=f"(frag_D[14]), "=f"(frag_D[15])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[3]), "r"(frag_B[7]), 
            "f"(frag_D[12]), "f"(frag_D[13]), "f"(frag_D[14]), "f"(frag_D[15])
        );
	    asm ("cp.async.commit_group;\n"::);
	    asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}

//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;  // TC_block_col to dense_tile_row.
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		} else {
			float4 t = FLOAT4(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.w));
		}
		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;  // TC_block_col to dense_tile_row.
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(z));
		} else {
			float4 t = FLOAT4(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(t.y));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(t.z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(t.w));
		}
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[0]), "r"(frag_B[4]), 
		  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[1]), "r"(frag_B[5]), 
		  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[8]), "=f"(frag_D[9]), "=f"(frag_D[10]), "=f"(frag_D[11])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[2]), "r"(frag_B[6]), 
		  "f"(frag_D[8]), "f"(frag_D[9]), "f"(frag_D[10]), "f"(frag_D[11])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[12]), "=f"(frag_D[13]), "=f"(frag_D[14]), "=f"(frag_D[15])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[3]), "r"(frag_B[7]), 
		  "f"(frag_D[12]), "f"(frag_D[13]), "f"(frag_D[14]), "f"(frag_D[15])
	  );

	uint32_t o_off = bid * BLK_H * embedding_dim + wid * 32;
	if (wid < dimTileNum)
	#pragma unroll
	for(int i = 0; i < 4; i++) {
		uint32_t row_d = 0;
		if( i < 2 ) {
			row_d = group_id;
		} else {
			row_d = group_id + 8;
		}
		uint32_t col_d = (tid_in_group << 3) + ((i & 0x1)<<2);
		uint32_t off = row_d * embedding_dim + col_d;
		uint32_t off_set = o_off + off;
		output[off_set] = frag_D[i];
		output[off_set + 1] = frag_D[i + 4];
		output[off_set + 2] = frag_D[i + 8];
		output[off_set + 3] = frag_D[i + 12];
	}
}



__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float4_split(
	const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA, 		
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	float *input,		    // input feature matrix.
	float *output							    // output feature matrix.
) {
    int bid = blockIdx.x;								// block_index == row_window_index
	int off_y = (blockIdx.y << 7);
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / 32;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) * 4 + wid * 32 + off_y;
	uint32_t group_id = (laneid >> 2);
    uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
    uint32_t frag_B[8]; // 8 * 8 * 2  / 32 = 4
	float frag_D[16] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
// pre loop
    {
		unsigned sparse_AToX_idx_start = lb * BLK_W;	
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  sparse_A[TCblocktile_id[eIdx]] = valuesA[eIdx];		// set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
		  sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
		}
		__syncthreads();
	}
//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
        int smem_sel_next = ( (j - lb - 1) & 1) ^ 1;
		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;  // TC_block_col to dense_tile_row.
			unsigned source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			} else {
				float4 t = FLOAT4(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.w));
			}
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;  // TC_block_col to dense_tile_row.
			source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(z));
			} else {
				float4 t = FLOAT4(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(t.y));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(t.z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(t.w));
			}

		}

	    eIdx_start = TCblock_offset[j];			
	    eIdx_end = TCblock_offset[j+1];
	    unsigned sparse_AToX_idx_start = j * BLK_W;	   

	    #pragma unroll
	    for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[(smem_sel_next << 7) + idx] = 0.0;
	    }
	    __syncthreads();
	    #pragma unroll
	    for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  int id_local = (((int)TCblocktile_id[eIdx])<<2);
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa_ptr + id_local + (smem_sel_next << 9)), "l"(valuesA+eIdx));	  
	    }
		if (tid < BLK_W) {	
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(sparse_AToX_idx+sparse_AToX_idx_start+tid));	
		}

		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[0]), "r"(frag_B[4]), 
            "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[1]), "r"(frag_B[5]), 
            "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
        );
		asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[8]), "=f"(frag_D[9]), "=f"(frag_D[10]), "=f"(frag_D[11])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[2]), "r"(frag_B[6]), 
            "f"(frag_D[8]), "f"(frag_D[9]), "f"(frag_D[10]), "f"(frag_D[11])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[12]), "=f"(frag_D[13]), "=f"(frag_D[14]), "=f"(frag_D[15])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[3]), "r"(frag_B[7]), 
            "f"(frag_D[12]), "f"(frag_D[13]), "f"(frag_D[14]), "f"(frag_D[15])
        );
	    asm ("cp.async.commit_group;\n"::);
	    asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}

//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim; // TC_block_col to dense_tile_row.
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		} else {
			float4 t = FLOAT4(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.w));
		}
		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;  // TC_block_col to dense_tile_row.
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(z));
		} else {
			float4 t = FLOAT4(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(t.y));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(t.z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(t.w));
		}
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[0]), "r"(frag_B[4]), 
		  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[1]), "r"(frag_B[5]), 
		  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[8]), "=f"(frag_D[9]), "=f"(frag_D[10]), "=f"(frag_D[11])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[2]), "r"(frag_B[6]), 
		  "f"(frag_D[8]), "f"(frag_D[9]), "f"(frag_D[10]), "f"(frag_D[11])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[12]), "=f"(frag_D[13]), "=f"(frag_D[14]), "=f"(frag_D[15])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[3]), "r"(frag_B[7]), 
		  "f"(frag_D[12]), "f"(frag_D[13]), "f"(frag_D[14]), "f"(frag_D[15])
	  );

	uint32_t o_off = bid * BLK_H * embedding_dim + wid * 32 + off_y;
	if (wid < dimTileNum)
	#pragma unroll
	for(int i = 0; i < 4; i++) {
		uint32_t row_d = 0;
		if( i < 2 ) {
			row_d = group_id;
		} else {
			row_d = group_id + 8;
		}
		uint32_t col_d = (tid_in_group << 3) + ((i & 0x1)<<2);
		uint32_t off = row_d * embedding_dim + col_d;
		uint32_t off_set = o_off + off;
		output[off_set] = frag_D[i];
		output[off_set + 1] = frag_D[i + 4];
		output[off_set + 2] = frag_D[i + 8];
		output[off_set + 3] = frag_D[i + 12];
	}
}




__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_double_buffer_float4_split(
	const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.	
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	float *input,		    // input feature matrix.
	float *output							    // output feature matrix.
) {
    int bid = blockIdx.x;								// block_index == row_window_index
	int off_y = (blockIdx.y << 7);
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / 32;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) * 4 + wid * 32 + off_y;
	uint32_t group_id = (laneid >> 2);
    uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
    uint32_t frag_B[8]; // 8 * 8 * 2  / 32 = 4
	float frag_D[16] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
// pre loop
    {
		unsigned sparse_AToX_idx_start = lb * BLK_W;	
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  sparse_A[TCblocktile_id[eIdx]] = 1.0;		// set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
		  sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
		}
		__syncthreads();
	}
//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
        int smem_sel_next = ( (j - lb - 1) & 1) ^ 1;
		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;  // TC_block_col to dense_tile_row.
			unsigned source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			} else {
				float4 t = FLOAT4(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.w));
			}
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;  // TC_block_col to dense_tile_row.
			source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(z));
			} else {
				float4 t = FLOAT4(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(t.y));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(t.z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(t.w));
			}

		}

	    eIdx_start = TCblock_offset[j];			
	    eIdx_end = TCblock_offset[j+1];
	    unsigned sparse_AToX_idx_start = j * BLK_W;	   

	    #pragma unroll
	    for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[(smem_sel_next << 7) + idx] = 0.0;
	    }
	    __syncthreads();
		if (tid < BLK_W) {	
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(sparse_AToX_idx+sparse_AToX_idx_start+tid));	
		}
	    #pragma unroll
	    for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  sparse_A[(smem_sel_next << 7) + TCblocktile_id[eIdx]] = 1.0;	  
	    }
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[0]), "r"(frag_B[4]), 
            "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[1]), "r"(frag_B[5]), 
            "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
        );
		asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[8]), "=f"(frag_D[9]), "=f"(frag_D[10]), "=f"(frag_D[11])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[2]), "r"(frag_B[6]), 
            "f"(frag_D[8]), "f"(frag_D[9]), "f"(frag_D[10]), "f"(frag_D[11])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[12]), "=f"(frag_D[13]), "=f"(frag_D[14]), "=f"(frag_D[15])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[3]), "r"(frag_B[7]), 
            "f"(frag_D[12]), "f"(frag_D[13]), "f"(frag_D[14]), "f"(frag_D[15])
        );
	    asm ("cp.async.commit_group;\n"::);
	    asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}

//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;  // TC_block_col to dense_tile_row.
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		} else {
			float4 t = FLOAT4(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.w));
		}
		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;  // TC_block_col to dense_tile_row.
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(z));
		} else {
			float4 t = FLOAT4(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(t.y));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(t.z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(t.w));
		}
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[0]), "r"(frag_B[4]), 
		  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[1]), "r"(frag_B[5]), 
		  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[8]), "=f"(frag_D[9]), "=f"(frag_D[10]), "=f"(frag_D[11])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[2]), "r"(frag_B[6]), 
		  "f"(frag_D[8]), "f"(frag_D[9]), "f"(frag_D[10]), "f"(frag_D[11])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[12]), "=f"(frag_D[13]), "=f"(frag_D[14]), "=f"(frag_D[15])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[3]), "r"(frag_B[7]), 
		  "f"(frag_D[12]), "f"(frag_D[13]), "f"(frag_D[14]), "f"(frag_D[15])
	  );

	uint32_t o_off = bid * BLK_H * embedding_dim + wid * 32 + off_y;
	if (wid < dimTileNum)
	#pragma unroll
	for(int i = 0; i < 4; i++) {
		uint32_t row_d = 0;
		if( i < 2 ) {
			row_d = group_id;
		} else {
			row_d = group_id + 8;
		}
		uint32_t col_d = (tid_in_group << 3) + ((i & 0x1)<<2);
		uint32_t off = row_d * embedding_dim + col_d;
		uint32_t off_set = o_off + off;
		output[off_set] = frag_D[i];
		output[off_set + 1] = frag_D[i + 4];
		output[off_set + 2] = frag_D[i + 8];
		output[off_set + 3] = frag_D[i + 12];
	}
}

__global__ void spmm_forward_cuda_kernel_improved_ptx_uint8_v1_strict_balance(
	const int *__restrict__ TCblock_rowid, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const int tc_count,
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
) {
    int bid = blockIdx.x;								// block_index == row_window_index
	const unsigned lb = bid * TCBLOCK_PER_WARP;
	const unsigned hb = min((bid + 1) * TCBLOCK_PER_WARP, tc_count);
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[BLK_W];					// TC_block col to dense_tile row.
	__shared__ int tc_rowid[TCBLOCK_PER_WARP];
	unsigned wid_BLK_H = wid * BLK_H;
	unsigned off = wid_BLK_H * BLK_H;
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) + wid_BLK_H;
	unsigned dense_dimIdx1 = dense_dimIdx + 8;
	uint32_t group_id = (laneid >> 2);
    uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	int shuffle_idx = (laneid/4) + (laneid%4)*8;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
    uint32_t frag_B[4]; // 8 * 8 * 2  / 32 = 4
	float frag_D[8] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	#pragma unroll
	for (unsigned idx = tid; idx < TCBLOCK_PER_WARP; idx += threadPerBlock) {
	  int ptr = lb + idx;
	  if (ptr < hb) {
		tc_rowid[idx] = __ldg(TCblock_rowid + ptr);
	  }
	}
	__syncthreads();
	unsigned former_row_id = tc_rowid[0];
	unsigned current_rid = former_row_id;
	for (unsigned j = lb; j < hb; j++) {
	  current_rid = tc_rowid[j - lb];
	  unsigned eIdx_start = TCblock_offset[j];			
	  unsigned eIdx_end = TCblock_offset[j + 1];
	  unsigned sparse_AToX_idx_start = j * BLK_W;	 
	  if (current_rid != former_row_id) {
		uint32_t o_off1 = former_row_id * BLK_H * embedding_dim + wid_BLK_H;
		uint32_t o_off2 = o_off1 + 8;
		if (wid < dimTileNum)
		#pragma unroll
		for(int i = 0; i < 4; i++) {
			uint32_t row_d = 0;
			if( i < 2 ) {
				row_d = group_id;
			} else {
				row_d = group_id + 8;
			}
			uint32_t col_d = (tid_in_group * 2) + (i & 0x1);
			uint32_t off = row_d * embedding_dim + col_d;
			atomicAdd(output + o_off1 + off, frag_D[i]);
			atomicAdd(output + o_off2 + off, frag_D[i + 4]);
			frag_D[i] = 0.0;
			frag_D[i + 4] = 0.0;
		}
		former_row_id = current_rid;
	  }
	  #pragma unroll
	  for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		sparse_A[idx] = 0.0;
	  }
	  if (tid < BLK_W) {
		sparse_AToX_index[tid] = (numNodes + 1) * embedding_dim;
	  }
	  __syncthreads();
	  #pragma unroll
	  for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock){
		int id_local = (int)TCblocktile_id[eIdx];
		sparse_A[id_local] = 1.0;		// set the edge of the sparse_A.	
	  }
	  #pragma unroll
	  for (unsigned eIdx = sparse_AToX_idx_start + tid; eIdx < sparse_AToX_idx_start + BLK_W; eIdx += threadPerBlock) {
		sparse_AToX_index[tid] = sparse_AToX_idx[eIdx] * embedding_dim;	
	  }
	  __syncthreads();
	  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[sparse_A_idx]));
	  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[sparse_A_idx1]));
	  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[sparse_A_idx2]));
	  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[sparse_A_idx3]));
	  __syncthreads();
	  if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[dense_rowIdx_off];						// TC_block_col to dense_tile_row.
		unsigned source_idx = dense_rowIdx + dense_dimIdx;

		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(input[source_idx]));
		asm volatile(
			"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
			: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
			: "r"(frag_A[0]), "r"(frag_A[1]), 
			  "r"(frag_B[0]), 
			  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
		);
		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(input[source_idx]));
		asm volatile(
			"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
			: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
			: "r"(frag_A[0]), "r"(frag_A[1]),
			  "r"(frag_B[1]), 
			  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
		);

		dense_rowIdx = sparse_AToX_index[dense_rowIdx_off1];						// TC_block_col to dense_tile_row.
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(input[source_idx]));
		asm volatile(
			"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
			: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
			: "r"(frag_A[2]), "r"(frag_A[3]), 
			  "r"(frag_B[2]), 
			  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
		);
		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(input[source_idx]));
		asm volatile(
			"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
			: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
			: "r"(frag_A[2]), "r"(frag_A[3]), 
			  "r"(frag_B[3]), 
			  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
		  );
	  }
	}
	uint32_t o_off1 = current_rid * BLK_H * embedding_dim + wid_BLK_H;
	uint32_t o_off2 = o_off1 + 8;
	if (wid < dimTileNum)
	#pragma unroll
	for(int i = 0; i < 4; i++) {
		uint32_t row_d = 0;
		if( i < 2 ) {
			row_d = group_id;
		} else {
			row_d = group_id + 8;
		}
		uint32_t col_d = (tid_in_group * 2) + (i & 0x1);
		uint32_t off = row_d * embedding_dim + col_d;
		atomicAdd(output + o_off1 + off, frag_D[i]);
		atomicAdd(output + o_off2 + off, frag_D[i + 4]);
	}
}

__global__ void spmm_forward_cuda_kernel_improved_ptx_uint8_v1_strict_balance_withv(
	const int *__restrict__ TCblock_rowid, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA,
	const int tc_count,
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
) {
    int bid = blockIdx.x;								// block_index == row_window_index
	const unsigned lb = bid * TCBLOCK_PER_WARP;
	const unsigned hb = min((bid + 1) * TCBLOCK_PER_WARP, tc_count);
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[BLK_W];					// TC_block col to dense_tile row.
	__shared__ int tc_rowid[TCBLOCK_PER_WARP];
	unsigned wid_BLK_H = wid * BLK_H;
	unsigned off = wid_BLK_H * BLK_H;
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) + wid_BLK_H;
	unsigned dense_dimIdx1 = dense_dimIdx + 8;
	uint32_t group_id = (laneid >> 2);
    uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	int shuffle_idx = (laneid/4) + (laneid%4)*8;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
    uint32_t frag_B[4]; // 8 * 8 * 2  / 32 = 4
	float frag_D[8] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	#pragma unroll
	for (unsigned idx = tid; idx < TCBLOCK_PER_WARP; idx += threadPerBlock) {
	  int ptr = lb + idx;
	  if (ptr < hb) {
		tc_rowid[idx] = __ldg(TCblock_rowid + ptr);
	  }
	}
	__syncthreads();
	unsigned former_row_id = tc_rowid[0];
	unsigned current_rid = former_row_id;
	for (unsigned j = lb; j < hb; j++) {
	  current_rid = tc_rowid[j - lb];
	  unsigned eIdx_start = TCblock_offset[j];			
	  unsigned eIdx_end = TCblock_offset[j + 1];
	  unsigned sparse_AToX_idx_start = j * BLK_W;	 
	  if (current_rid != former_row_id) {
		uint32_t o_off1 = former_row_id * BLK_H * embedding_dim + wid_BLK_H;
		uint32_t o_off2 = o_off1 + 8;
		if (wid < dimTileNum)
		#pragma unroll
		for(int i = 0; i < 4; i++) {
			uint32_t row_d = 0;
			if( i < 2 ) {
				row_d = group_id;
			} else {
				row_d = group_id + 8;
			}
			uint32_t col_d = (tid_in_group * 2) + (i & 0x1);
			uint32_t off = row_d * embedding_dim + col_d;
			atomicAdd(output + o_off1 + off, frag_D[i]);
			atomicAdd(output + o_off2 + off, frag_D[i + 4]);
			frag_D[i] = 0.0;
			frag_D[i + 4] = 0.0;
		}
		former_row_id = current_rid;
	  }
	  #pragma unroll
	  for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		sparse_A[idx] = 0.0;
	  }
	  if (tid < BLK_W) {
		sparse_AToX_index[tid] = (numNodes + 1) * embedding_dim;
	  }
	  __syncthreads();
	  #pragma unroll
	  for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		sparse_A[(int)TCblocktile_id[eIdx]] = valuesA[eIdx];		// set the edge of the sparse_A.	
	  }
	  #pragma unroll
	  for (unsigned eIdx = sparse_AToX_idx_start + tid; eIdx < sparse_AToX_idx_start + BLK_W; eIdx += threadPerBlock) {
		sparse_AToX_index[tid] = sparse_AToX_idx[eIdx] * embedding_dim;	
	  }
	  __syncthreads();
	  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[sparse_A_idx]));
	  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[sparse_A_idx1]));
	  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[sparse_A_idx2]));
	  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[sparse_A_idx3]));
	  __syncthreads();
	  if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[dense_rowIdx_off];						// TC_block_col to dense_tile_row.
		unsigned source_idx = dense_rowIdx + dense_dimIdx;

		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(input[source_idx]));
		asm volatile(
			"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
			: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
			: "r"(frag_A[0]), "r"(frag_A[1]), 
			  "r"(frag_B[0]), 
			  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
		);
		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(input[source_idx]));
		asm volatile(
			"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
			: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
			: "r"(frag_A[0]), "r"(frag_A[1]),
			  "r"(frag_B[1]), 
			  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
		);

		dense_rowIdx = sparse_AToX_index[dense_rowIdx_off1];						// TC_block_col to dense_tile_row.
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(input[source_idx]));
		asm volatile(
			"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
			: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
			: "r"(frag_A[2]), "r"(frag_A[3]), 
			  "r"(frag_B[2]), 
			  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
		);
		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(input[source_idx]));
		asm volatile(
			"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
			: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
			: "r"(frag_A[2]), "r"(frag_A[3]), 
			  "r"(frag_B[3]), 
			  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
		  );
	  }
	}
	uint32_t o_off1 = current_rid * BLK_H * embedding_dim + wid_BLK_H;
	uint32_t o_off2 = o_off1 + 8;
	if (wid < dimTileNum)
	#pragma unroll
	for(int i = 0; i < 4; i++) {
		uint32_t row_d = 0;
		if( i < 2 ) {
			row_d = group_id;
		} else {
			row_d = group_id + 8;
		}
		uint32_t col_d = (tid_in_group * 2) + (i & 0x1);
		uint32_t off = row_d * embedding_dim + col_d;
		atomicAdd(output + o_off1 + off, frag_D[i]);
		atomicAdd(output + o_off2 + off, frag_D[i + 4]);
	}
}
__global__ void spmm_forward_cuda_kernel_improved_ptx_uint8_v1_strict_balance_withv_clock(
	const int *__restrict__ TCblock_rowid, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA,
	const int tc_count,
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output,							    // output feature matrix.
	clocktype* timer
) {
	int smid = getSMId();
    clocktype tt = GlobalTimer64();
    int bid = blockIdx.x;								// block_index == row_window_index
	const unsigned lb = bid * TCBLOCK_PER_WARP;
	const unsigned hb = min((bid + 1) * TCBLOCK_PER_WARP, tc_count);
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[BLK_W];					// TC_block col to dense_tile row.
	__shared__ int tc_rowid[TCBLOCK_PER_WARP];
	unsigned wid_BLK_H = wid * BLK_H;
	unsigned off = wid_BLK_H * BLK_H;
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) + wid_BLK_H;
	unsigned dense_dimIdx1 = dense_dimIdx + 8;
	uint32_t group_id = (laneid >> 2);
    uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	int shuffle_idx = (laneid/4) + (laneid%4)*8;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
    uint32_t frag_B[4]; // 8 * 8 * 2  / 32 = 4
	float frag_D[8] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	#pragma unroll
	for (unsigned idx = tid; idx < TCBLOCK_PER_WARP; idx += threadPerBlock) {
	  int ptr = lb + idx;
	  if (ptr < hb) {
		tc_rowid[idx] = __ldg(TCblock_rowid + ptr);
	  }
	}
	__syncthreads();
	unsigned former_row_id = tc_rowid[0];
	unsigned current_rid = former_row_id;
	for (unsigned j = lb; j < hb; j++) {
	  current_rid = tc_rowid[j - lb];
	  unsigned eIdx_start = TCblock_offset[j];			
	  unsigned eIdx_end = TCblock_offset[j + 1];
	  unsigned sparse_AToX_idx_start = j * BLK_W;	 
	  if (current_rid != former_row_id) {
		uint32_t o_off1 = former_row_id * BLK_H * embedding_dim + wid_BLK_H;
		uint32_t o_off2 = o_off1 + 8;
		if (wid < dimTileNum)
		#pragma unroll
		for(int i = 0; i < 4; i++) {
			uint32_t row_d = 0;
			if( i < 2 ) {
				row_d = group_id;
			} else {
				row_d = group_id + 8;
			}
			uint32_t col_d = (tid_in_group * 2) + (i & 0x1);
			uint32_t off = row_d * embedding_dim + col_d;
			atomicAdd(output + o_off1 + off, frag_D[i]);
			atomicAdd(output + o_off2 + off, frag_D[i + 4]);
			frag_D[i] = 0.0;
			frag_D[i + 4] = 0.0;
		}
		former_row_id = current_rid;
	  }
	  #pragma unroll
	  for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		sparse_A[idx] = 0.0;
	  }
	  if (tid < BLK_W) {
		sparse_AToX_index[tid] = (numNodes + 1) * embedding_dim;
	  }
	  __syncthreads();
	  #pragma unroll
	  for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		sparse_A[(int)TCblocktile_id[eIdx]] = valuesA[eIdx];		// set the edge of the sparse_A.	
	  }
	  #pragma unroll
	  for (unsigned eIdx = sparse_AToX_idx_start + tid; eIdx < sparse_AToX_idx_start + BLK_W; eIdx += threadPerBlock) {
		sparse_AToX_index[tid] = sparse_AToX_idx[eIdx] * embedding_dim;	
	  }
	  __syncthreads();
	  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[sparse_A_idx]));
	  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[sparse_A_idx1]));
	  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[sparse_A_idx2]));
	  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[sparse_A_idx3]));
	  __syncthreads();
	  if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[dense_rowIdx_off];						// TC_block_col to dense_tile_row.
		unsigned source_idx = dense_rowIdx + dense_dimIdx;

		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(input[source_idx]));
		asm volatile(
			"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
			: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
			: "r"(frag_A[0]), "r"(frag_A[1]), 
			  "r"(frag_B[0]), 
			  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
		);
		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(input[source_idx]));
		asm volatile(
			"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
			: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
			: "r"(frag_A[0]), "r"(frag_A[1]),
			  "r"(frag_B[1]), 
			  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
		);

		dense_rowIdx = sparse_AToX_index[dense_rowIdx_off1];						// TC_block_col to dense_tile_row.
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(input[source_idx]));
		asm volatile(
			"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
			: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
			: "r"(frag_A[2]), "r"(frag_A[3]), 
			  "r"(frag_B[2]), 
			  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
		);
		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(input[source_idx]));
		asm volatile(
			"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
			: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
			: "r"(frag_A[2]), "r"(frag_A[3]), 
			  "r"(frag_B[3]), 
			  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
		  );
	  }
	}
	uint32_t o_off1 = current_rid * BLK_H * embedding_dim + wid_BLK_H;
	uint32_t o_off2 = o_off1 + 8;
	if (wid < dimTileNum)
	#pragma unroll
	for(int i = 0; i < 4; i++) {
		uint32_t row_d = 0;
		if( i < 2 ) {
			row_d = group_id;
		} else {
			row_d = group_id + 8;
		}
		uint32_t col_d = (tid_in_group * 2) + (i & 0x1);
		uint32_t off = row_d * embedding_dim + col_d;
		atomicAdd(output + o_off1 + off, frag_D[i]);
		atomicAdd(output + o_off2 + off, frag_D[i + 4]);
	}

	clocktype tt2 = GlobalTimer64();
	if(threadIdx.x == 0) {
		timer[3 * blockIdx.x] = tt;
		timer[3 * blockIdx.x + 1] = tt2;
		timer[3 * blockIdx.x + 2] = (clocktype)(smid);
	}
}

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float4_split_balance(
	const int *__restrict__ TCblock_rowid, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA, 		
	const int tc_count,
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	float *input,		    // input feature matrix.
	float *output							    // output feature matrix.
) {
    int bid = blockIdx.x;								// block_index == row_window_index
	int off_y = (blockIdx.y << 7);
	const unsigned lb = bid * TCBLOCK_PER_WARP;
	const unsigned hb = min((bid + 1) * TCBLOCK_PER_WARP, tc_count);
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / 32;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	__shared__ int tc_rowid[TCBLOCK_PER_WARP];
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) * 4 + wid * 32 + off_y;
	uint32_t group_id = (laneid >> 2);
    uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
    uint32_t frag_B[8]; // 8 * 8 * 2  / 32 = 4
	float frag_D[16] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	#pragma unroll
	for (unsigned idx = tid; idx < TCBLOCK_PER_WARP; idx += threadPerBlock) {
	  int ptr = lb + idx;
	  if (ptr < hb) {
		tc_rowid[idx] = __ldg(TCblock_rowid + ptr);
	  }
	}
	__syncthreads();
	unsigned former_row_id = tc_rowid[0];
	unsigned current_rid = former_row_id;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
// pre loop
    {
		unsigned sparse_AToX_idx_start = lb * BLK_W;	
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  sparse_A[TCblocktile_id[eIdx]] = valuesA[eIdx];  // set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
		  sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
		}
		__syncthreads();
	}
//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
        int smem_sel_next = ( (j - lb - 1) & 1) ^ 1;
		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim; // TC_block_col to dense_tile_row.
			unsigned source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			} else {
				float4 t = FLOAT4(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.w));
			}
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim; // TC_block_col to dense_tile_row.
			source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(z));
			} else {
				float4 t = FLOAT4(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(t.y));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(t.z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(t.w));
			}

		}

	    eIdx_start = TCblock_offset[j];			
	    eIdx_end = TCblock_offset[j+1];
	    unsigned sparse_AToX_idx_start = j * BLK_W;	   

	    #pragma unroll
	    for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[(smem_sel_next << 7) + idx] = 0.0;
	    }
	    __syncthreads();
	    #pragma unroll
	    for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  int id_local = (((int)TCblocktile_id[eIdx])<<2);
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa_ptr + id_local + (smem_sel_next << 9)), "l"(valuesA+eIdx));	  
	    }
		if (tid < BLK_W) {	
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(sparse_AToX_idx+sparse_AToX_idx_start+tid));	
		}

		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[0]), "r"(frag_B[4]), 
            "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[1]), "r"(frag_B[5]), 
            "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
        );
		asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[8]), "=f"(frag_D[9]), "=f"(frag_D[10]), "=f"(frag_D[11])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[2]), "r"(frag_B[6]), 
            "f"(frag_D[8]), "f"(frag_D[9]), "f"(frag_D[10]), "f"(frag_D[11])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[12]), "=f"(frag_D[13]), "=f"(frag_D[14]), "=f"(frag_D[15])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[3]), "r"(frag_B[7]), 
            "f"(frag_D[12]), "f"(frag_D[13]), "f"(frag_D[14]), "f"(frag_D[15])
        );


		current_rid = tc_rowid[j - lb];
		if (current_rid != former_row_id) {
			uint32_t o_off = former_row_id * BLK_H * embedding_dim + wid * 32 + off_y;
			if (wid < dimTileNum)
			#pragma unroll
			for(int i = 0; i < 4; i++) {
				uint32_t row_d = 0;
				if( i < 2 ) {
					row_d = group_id;
				} else {
					row_d = group_id + 8;
				}
				uint32_t col_d = (tid_in_group << 3) + ((i & 0x1)<<2);
				uint32_t off = row_d * embedding_dim + col_d;
				uint32_t off_set = o_off + off;
				atomicAdd(output + off_set, frag_D[i]);
				atomicAdd(output + off_set + 1, frag_D[i + 4]);
				atomicAdd(output + off_set + 2, frag_D[i + 8]);
				atomicAdd(output + off_set + 3, frag_D[i + 12]);
				frag_D[i] = 0.0;
				frag_D[i + 4] = 0.0;
				frag_D[i + 8] = 0.0;
				frag_D[i + 12] = 0.0;
			}
			former_row_id = current_rid;
		}

	    asm ("cp.async.commit_group;\n"::);
	    asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}
//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;  // TC_block_col to dense_tile_row.
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		} else {
			float4 t = FLOAT4(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.w));
		}
		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;  // TC_block_col to dense_tile_row.
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(z));
		} else {
			float4 t = FLOAT4(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(t.y));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(t.z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(t.w));
		}
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[0]), "r"(frag_B[4]), 
		  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[1]), "r"(frag_B[5]), 
		  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[8]), "=f"(frag_D[9]), "=f"(frag_D[10]), "=f"(frag_D[11])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[2]), "r"(frag_B[6]), 
		  "f"(frag_D[8]), "f"(frag_D[9]), "f"(frag_D[10]), "f"(frag_D[11])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[12]), "=f"(frag_D[13]), "=f"(frag_D[14]), "=f"(frag_D[15])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[3]), "r"(frag_B[7]), 
		  "f"(frag_D[12]), "f"(frag_D[13]), "f"(frag_D[14]), "f"(frag_D[15])
	  );

	uint32_t o_off = current_rid * BLK_H * embedding_dim + wid * 32 + off_y;
	if (wid < dimTileNum)
	#pragma unroll
	for(int i = 0; i < 4; i++) {
		uint32_t row_d = 0;
		if( i < 2 ) {
			row_d = group_id;
		} else {
			row_d = group_id + 8;
		}
		uint32_t col_d = (tid_in_group << 3) + ((i & 0x1)<<2);
		uint32_t off = row_d * embedding_dim + col_d;
		uint32_t off_set = o_off + off;
		atomicAdd(output + off_set, frag_D[i]);
		atomicAdd(output + off_set + 1, frag_D[i + 4]);
		atomicAdd(output + off_set + 2, frag_D[i + 8]);
		atomicAdd(output + off_set + 3, frag_D[i + 12]);
	}
}
