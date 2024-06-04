#include <torch/extension.h>
#include <vector>
#include <string.h>
#include <cstdlib>
#include <map>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/unique.h>
#define min(x, y) (((x) < (y))? (x) : (y))

void fill_edgeToRow_cuda(int *edgeToRow, int *nodePointer, int num_nodes);

void fill_window_cuda(int *edgeToColumn, int *blockPartition, int *nodePointer,
                      int *edgeList, int blockSize_h, int blockSize_w,
                      int num_nodes);

void fill_segment_cuda(int *nodePointer, int *seg_out, int blockSize_h,
                       int blockSize_w, int num_nodes);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int>
seg_sort_dequ(int *seg, int *edgeLists, int *nodepointer, int *edgetocol,
              int *edgetorow, int *blockPartition, int *blocknum,
              int *row_window_offset, int blockSize_h, int blockSize_w,
              int num_nodes, int num_edges, int rowwindow_num);

std::vector<torch::Tensor>
spmm_forward_cuda(torch::Tensor nodePointer, torch::Tensor edgeList,
                  torch::Tensor blockPartition, torch::Tensor edgeToColumn,
                  torch::Tensor edgeToRow, int num_nodes, int num_edges,
                  int embedding_dim, torch::Tensor input);

std::vector<torch::Tensor> spmm_forward_improved_ptx_uint8_cuda(
    torch::Tensor Rowwindow_offset, torch::Tensor TCblocktile_id,
    torch::Tensor TCblock_offset, torch::Tensor sparse_AToX_idx, int num_nodes,
    int num_edges, int embedding_dim, torch::Tensor input, std::string exeplan);

std::vector<torch::Tensor> spmm_forward_cuda_origin_clock(
    torch::Tensor Rowwindow_offset, torch::Tensor TCblocktile_id,
    torch::Tensor TCblock_offset, torch::Tensor sparse_AToX_idx, int num_nodes,
    int num_edges, int embedding_dim, torch::Tensor input);

std::vector<torch::Tensor> spmm_forward_improved_ptx_uint8_cuda_dtc_for_gcn(
    torch::Tensor Rowwindow_offset, torch::Tensor TCblocktile_id,
    torch::Tensor TCblock_offset, torch::Tensor sparse_AToX_idx, int num_nodes,
    int num_edges, int embedding_dim, torch::Tensor input);

std::vector<torch::Tensor>
spmm_forward_improved_ptx_uint8_prefetch_balance_sort_cuda(
    torch::Tensor Rowwindow_offset, torch::Tensor TCblocktile_id,
    torch::Tensor TCblock_offset, torch::Tensor sparse_AToX_idx,
    torch::Tensor sort, int num_nodes, int num_edges, int embedding_dim,
    torch::Tensor input);

std::vector<torch::Tensor> spmm_balance_forward_cuda_ptx_unit8_prefetch(
    torch::Tensor TCblock_rowid, torch::Tensor TCblocktile_id,
    torch::Tensor TCblock_offset, torch::Tensor sparse_AToX_idx, int tc_count,
    int num_nodes, int num_edges, int embedding_dim, torch::Tensor input,
    std::string exeplan);

std::vector<torch::Tensor>
spmm_balance_clock(torch::Tensor TCblock_rowid, torch::Tensor TCblocktile_id,
                   torch::Tensor TCblock_offset, torch::Tensor sparse_AToX_idx,
                   int tc_count, int num_nodes, int num_edges,
                   int embedding_dim, torch::Tensor input);

std::vector<torch::Tensor> spmm_forward_cusparse(torch::Tensor rowoffset,
                                                 torch::Tensor colind,
                                                 torch::Tensor input,
                                                 int num_nodes, int num_edges,
                                                 int embedding_dim, int algid);

std::vector<torch::Tensor> spmm_forward_cusparse_blocked_ellpack(
    torch::Tensor ell_colind, torch::Tensor input, int num_nodes,
    int block_size, int ell_columns, int embedding_dim);

std::vector<torch::Tensor> spmm_forward_sputnik(torch::Tensor rowoffset,
                                                torch::Tensor colind,
                                                torch::Tensor input,
                                                int num_nodes, int num_edges,
                                                int embedding_dim);

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

//////////////////////////////////////////
//
// SPMM Foward Pass (GCN, GraphSAGE) from TC-GNN
//
////////////////////////////////////////////
std::vector<torch::Tensor>
spmm_forward(torch::Tensor input, torch::Tensor nodePointer,
             torch::Tensor edgeList, torch::Tensor blockPartition,
             torch::Tensor edgeToColumn, torch::Tensor edgeToRow) {
  CHECK_INPUT(input);
  CHECK_INPUT(nodePointer);
  CHECK_INPUT(edgeList);
  CHECK_INPUT(blockPartition);
  CHECK_INPUT(edgeToColumn);
  CHECK_INPUT(edgeToRow);

  int num_nodes = nodePointer.size(0) - 1;
  int num_edges = edgeList.size(0);
  int embedding_dim = input.size(1);

  return spmm_forward_cuda(nodePointer, edgeList, blockPartition, edgeToColumn,
                           edgeToRow, num_nodes, num_edges, embedding_dim,
                           input);
}

std::vector<torch::Tensor> spmm_forward_ptx_uint8_improved(
    torch::Tensor input, torch::Tensor Rowwindow_offset,
    torch::Tensor TCblocktile_id, torch::Tensor TCblock_offset,
    torch::Tensor sparse_AToX_idx, int num_nodes, int num_edges,
    std::string exeplan) {
  CHECK_INPUT(input);
  CHECK_INPUT(Rowwindow_offset);
  CHECK_INPUT(TCblocktile_id);
  CHECK_INPUT(TCblock_offset);
  CHECK_INPUT(sparse_AToX_idx);
  int embedding_dim = input.size(1);
  return spmm_forward_improved_ptx_uint8_cuda(
      Rowwindow_offset, TCblocktile_id, TCblock_offset, sparse_AToX_idx,
      num_nodes, num_edges, embedding_dim, input, exeplan);
}

std::vector<torch::Tensor> spmm_forward_origin_clock(
    torch::Tensor input, torch::Tensor Rowwindow_offset,
    torch::Tensor TCblocktile_id, torch::Tensor TCblock_offset,
    torch::Tensor sparse_AToX_idx, int num_nodes, int num_edges) {
  CHECK_INPUT(input);
  CHECK_INPUT(Rowwindow_offset);
  CHECK_INPUT(TCblocktile_id);
  CHECK_INPUT(TCblock_offset);
  CHECK_INPUT(sparse_AToX_idx);
  int embedding_dim = input.size(1);
  return spmm_forward_cuda_origin_clock(
      Rowwindow_offset, TCblocktile_id, TCblock_offset, sparse_AToX_idx,
      num_nodes, num_edges, embedding_dim, input);
}
std::vector<torch::Tensor> spmm_forward_ptx_uint8_improved_for_gcn(
    torch::Tensor input, torch::Tensor Rowwindow_offset,
    torch::Tensor TCblocktile_id, torch::Tensor TCblock_offset,
    torch::Tensor sparse_AToX_idx, int num_nodes, int num_edges) {
  CHECK_INPUT(input);
  CHECK_INPUT(Rowwindow_offset);  
  CHECK_INPUT(TCblocktile_id);
  CHECK_INPUT(TCblock_offset);
  CHECK_INPUT(sparse_AToX_idx);
  int embedding_dim = input.size(1);
  return spmm_forward_improved_ptx_uint8_cuda_dtc_for_gcn(Rowwindow_offset, 
                            TCblocktile_id, TCblock_offset, sparse_AToX_idx,
                            num_nodes, num_edges, embedding_dim,
                            input);
}

std::vector<torch::Tensor> spmm_balance_forward_ptx_uint8_prefetch(
    torch::Tensor input, torch::Tensor TCblock_rowid,
    torch::Tensor TCblocktile_id, torch::Tensor TCblock_offset,
    torch::Tensor sparse_AToX_idx, int num_nodes, std::string exeplan) {
  CHECK_INPUT(input);
  CHECK_INPUT(TCblock_rowid);
  CHECK_INPUT(TCblocktile_id);
  CHECK_INPUT(TCblock_offset);
  CHECK_INPUT(sparse_AToX_idx);
  int num_edges = TCblocktile_id.size(0);
  int embedding_dim = input.size(1);
  int tc_count = TCblock_rowid.size(0);
  return spmm_balance_forward_cuda_ptx_unit8_prefetch(
      TCblock_rowid, TCblocktile_id, TCblock_offset, sparse_AToX_idx, tc_count,
      num_nodes, num_edges, embedding_dim, input, exeplan);
}

std::vector<torch::Tensor>
spmm_balance_forward_clock(torch::Tensor input, torch::Tensor TCblock_rowid,
                           torch::Tensor TCblocktile_id,
                           torch::Tensor TCblock_offset,
                           torch::Tensor sparse_AToX_idx, int num_nodes) {
  CHECK_INPUT(input);
  CHECK_INPUT(TCblock_rowid);
  CHECK_INPUT(TCblocktile_id);
  CHECK_INPUT(TCblock_offset);
  CHECK_INPUT(sparse_AToX_idx);
  int num_edges = TCblocktile_id.size(0);
  int embedding_dim = input.size(1);
  int tc_count = TCblock_rowid.size(0);
  return spmm_balance_clock(TCblock_rowid, TCblocktile_id, TCblock_offset,
                            sparse_AToX_idx, tc_count, num_nodes, num_edges,
                            embedding_dim, input);
}

std::vector<torch::Tensor>
spmm_forward_cusparse_impl(torch::Tensor input, torch::Tensor rowoffset,
                           torch::Tensor colind, int num_nodes, int alg_id) {
  CHECK_INPUT(input);
  CHECK_INPUT(rowoffset);
  CHECK_INPUT(colind);
  int num_edges = colind.size(0);
  int embedding_dim = input.size(1);
  return spmm_forward_cusparse(rowoffset, colind, input, num_nodes, num_edges,
                               embedding_dim, alg_id);
}

std::vector<torch::Tensor> spmm_forward_cusparse_blocked_ellpack_impl(
    torch::Tensor input, torch::Tensor ell_colind, int num_nodes,
    int block_size, int ell_columns) {
  CHECK_INPUT(input);
  CHECK_INPUT(ell_colind);
  int embedding_dim = input.size(1);
  return spmm_forward_cusparse_blocked_ellpack(
      ell_colind, input, num_nodes, block_size, ell_columns, embedding_dim);
}

std::vector<torch::Tensor> spmm_forward_sputnik_impl(torch::Tensor input,
                                                     torch::Tensor rowoffset,
                                                     torch::Tensor colind,
                                                     int num_nodes) {
  CHECK_INPUT(input);
  CHECK_INPUT(rowoffset);
  CHECK_INPUT(colind);
  int num_edges = colind.size(0);
  int embedding_dim = input.size(1);
  return spmm_forward_sputnik(rowoffset, colind, input, num_nodes, num_edges,
                              embedding_dim);
}

// condense an sorted array with duplication: [1,2,2,3,4,5,5]
// after condense, it becomes: [1,2,3,4,5].
// Also, mapping the origin value to the corresponding new location in the new
// array. 1->[0], 2->[1], 3->[2], 4->[3], 5->[4].
std::map<unsigned, unsigned> inplace_deduplication(unsigned *array,
                                                   unsigned length) {
  int loc = 0, cur = 1;
  std::map<unsigned, unsigned> nb2col;
  nb2col[array[0]] = 0;
  while (cur < length) {
    if (array[cur] != array[cur - 1]) {
      loc++;
      array[loc] = array[cur];
      nb2col[array[cur]] = loc; // mapping from eid to TC_block column index.[]
    }
    cur++;
  }
  return nb2col;
}

// From TC-GNN
int preprocess(torch::Tensor edgeList_tensor, 
                torch::Tensor nodePointer_tensor, 
                int num_nodes, 
                int blockSize_h,
                int blockSize_w,
                torch::Tensor blockPartition_tensor, 
                torch::Tensor edgeToColumn_tensor,
                torch::Tensor edgeToRow_tensor
                ){

    // input tensors.
    auto edgeList = edgeList_tensor.accessor<int, 1>();
    auto nodePointer = nodePointer_tensor.accessor<int, 1>();

    // output tensors.
    auto blockPartition = blockPartition_tensor.accessor<int, 1>();
    auto edgeToColumn = edgeToColumn_tensor.accessor<int, 1>();
    auto edgeToRow = edgeToRow_tensor.accessor<int, 1>();
    auto start = std::chrono::high_resolution_clock::now();
    
    unsigned block_counter = 0;
    #pragma omp parallel for 
    for (unsigned nid = 0; nid < num_nodes; nid++){
        for (unsigned eid = nodePointer[nid]; eid < nodePointer[nid+1]; eid++)
            edgeToRow[eid] = nid;
    }
    #pragma omp parallel for reduction(+:block_counter)
    for (unsigned iter = 0; iter < num_nodes + 1; iter +=  blockSize_h){
        unsigned windowId = iter / blockSize_h;
        unsigned block_start = nodePointer[iter];
        unsigned block_end = nodePointer[min(iter + blockSize_h, num_nodes)];
        unsigned num_window_edges = block_end - block_start;
        unsigned *neighbor_window = (unsigned *) malloc (num_window_edges * sizeof(unsigned));
        memcpy(neighbor_window, &edgeList[block_start], num_window_edges * sizeof(unsigned));

        // Step-1: Sort the neighbor id array of a row window.
        thrust::sort(neighbor_window, neighbor_window + num_window_edges);

        // Step-2: Deduplication of the edge id array.
        // printf("Before dedupblication: %d\n", num_window_edges);
        std::map<unsigned, unsigned> clean_edges2col = inplace_deduplication(neighbor_window, num_window_edges);
        
        // generate blockPartition --> number of TC_blcok in each row window.
        blockPartition[windowId] = (clean_edges2col.size() + blockSize_w - 1) /blockSize_w;
        block_counter += blockPartition[windowId];

        // scan the array and generate edge to column mapping. --> edge_id to compressed_column_id of TC_block.
        for (unsigned e_index = block_start; e_index < block_end; e_index++){
            unsigned eid = edgeList[e_index];
            edgeToColumn[e_index] = clean_edges2col[eid];
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "\t CPU  original Preprocess time: " << elapsed_seconds.count() << " seconds\n";
    printf("TC_Blocks:\t%d\nExp_Edges:\t%d\n", block_counter, block_counter * 8 * 16);
    return block_counter;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, int>
preprocess_gpu(torch::Tensor edgeList_tensor, torch::Tensor nodePointer_tensor,
               int num_nodes, int blockSize_h, int blockSize_w,
               torch::Tensor blockPartition_tensor,
               torch::Tensor edgeToColumn_tensor,
               torch::Tensor edgeToRow_tensor) {
  // input tensors.
  auto options_gpu =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  auto options_gpu_unit8 =
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
  auto edgeList = edgeList_tensor.data<int>();
  auto blockPartition = blockPartition_tensor.data<int>();
  auto row_window_offset_tensor =
      torch::zeros({blockPartition_tensor.size(0) + 1}, options_gpu);
  auto row_window_offset = row_window_offset_tensor.data<int>();
  auto edgeToColumn = edgeToColumn_tensor.data<int>();
  auto seg_out_tensor = torch::zeros({edgeList_tensor.size(0)}, options_gpu);
  auto blocknum = torch::zeros({1}, options_gpu);
  auto block_num = blocknum.data<int>();
  auto edgeToRow = edgeToRow_tensor.data<int>();
  auto nodePointer = nodePointer_tensor.data<int>();
  auto seg_out = seg_out_tensor.data<int>();
  auto start = std::chrono::high_resolution_clock::now();
  fill_edgeToRow_cuda(edgeToRow, nodePointer, num_nodes);
  int block_counter = 0;
  fill_segment_cuda(nodePointer, seg_out, blockSize_h, blockSize_w, num_nodes);
  auto tuple_tensor_blockcnt = seg_sort_dequ(
      seg_out, edgeList, nodePointer, edgeToColumn, edgeToRow, blockPartition,
      block_num, row_window_offset, blockSize_h, blockSize_w, num_nodes,
      edgeList_tensor.size(0), blockPartition_tensor.size(0));
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "\t GPU Preprocess time: " << elapsed_seconds.count()
            << " seconds\n";
  auto tcblock_offset_tensor = std::get<0>(tuple_tensor_blockcnt);
  auto tcblock_rowid_tensor = std::get<1>(tuple_tensor_blockcnt);
  auto tcblocktile_id_tensor = std::get<2>(tuple_tensor_blockcnt);
  auto sparse_AToX_index_tensor = std::get<3>(tuple_tensor_blockcnt);
  block_counter = std::get<4>(tuple_tensor_blockcnt);
  printf("TC_Blocks:\t%d\nExp_Edges:\t%d\n", block_counter,
         block_counter * 8 * 16);
  return std::make_tuple(row_window_offset_tensor, tcblock_rowid_tensor,
                         tcblocktile_id_tensor, tcblock_offset_tensor,
                         sparse_AToX_index_tensor, block_counter);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("preprocess", &preprocess, "Preprocess Step on (CPU)");
  m.def("preprocess_gpu", &preprocess_gpu, "Preprocess Step on (CUDA)");

  // forward computation
  m.def("run_TCGNNSpMM", &spmm_forward, "TC-GNN SPMM forward (CUDA)");
  m.def("run_DTCSpMM", &spmm_forward_ptx_uint8_improved,
        "DTC-SPMM forward (CUDA)");
  m.def("run_DTCSpMM_balance", &spmm_balance_forward_ptx_uint8_prefetch,
        "DTC-SpMM balance version forward (CUDA)");
  m.def("forward_origin_clock", &spmm_forward_origin_clock,
        "DTC-SPMM origin clock forward (CUDA)");
  m.def("DTCSpMM_gcn", &spmm_forward_ptx_uint8_improved_for_gcn,
        "DTC-SPMM ptx uint8 improved forward gcn (CUDA)");
  m.def("balance_forward_clock", &spmm_balance_forward_clock,
        "DTC-SpMM balance SPMM ptx uint8 prefetch forward (CUDA)");
  m.def("run_cuSPARSE", &spmm_forward_cusparse_impl,
        "cusparse SPMM forward (CUDA)");
  m.def("run_Sputnik", &spmm_forward_sputnik_impl,
        "sputnik SPMM forward (CUDA)");
  m.def("forward_cusparse_bell", &spmm_forward_cusparse_blocked_ellpack_impl,
        "cusparse SPMM forward with bell(CUDA)");

  // backward computation
  m.def("backward", &spmm_forward, "TC-GNN SPMM backward (CUDA)");
  m.def("backward_ptx_uint8_improved", &spmm_forward_ptx_uint8_improved,
        "DTC-SPMM ptx uint8 improved backward (CUDA)");
  m.def("backward_ptx_uint8_improved_gcn",
        &spmm_forward_ptx_uint8_improved_for_gcn,
        "DTC-SPMM ptx uint8 improved backward gcn (CUDA)");
  m.def("balance_backward_ptx_uint8_prefetch",
        &spmm_balance_forward_ptx_uint8_prefetch,
        "DTC-SpMM balance SPMM ptx uint8 prefetch backward (CUDA)");
  m.def("backward_cusparse", &spmm_forward_cusparse_impl,
        "cusparse SPMM backward (CUDA)");
  m.def("backward_sputnik", &spmm_forward_sputnik_impl,
        "sputnik SPMM backward (CUDA)");
  m.def("backward_cusparse_bell", &spmm_forward_cusparse_blocked_ellpack_impl,
        "cusparse SPMM back with bell(CUDA)");
}