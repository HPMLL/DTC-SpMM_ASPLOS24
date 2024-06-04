#include <stdio.h>
#include <cuda.h>
// Size of array
#define N 32
#define REPEAT_TIMES 10
// Kernel
__global__ void warp_transpose(uint32_t *startClk, uint32_t *stopClk, float *a, float *b)
{
    int i = threadIdx.x;
    int src_lane = (i/4) + (i%4)*8;
    float v1 = a[i];
	// synchronize all threads
	asm volatile ("bar.sync 0;");
	float v2 = 0;
	// start timing
	uint32_t start = 0;
	asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
	for (int j=0 ; j < REPEAT_TIMES ; ++j) {
		v2 = __shfl_sync (0xffffffff, v1, src_lane, 32);
		v2 = __shfl_sync (0xffffffff, v1, src_lane, 32);
		v2 = __shfl_sync (0xffffffff, v1, src_lane, 32);
		v2 = __shfl_sync (0xffffffff, v1, src_lane, 32);
	}
	// synchronize all threads
	asm volatile("bar.sync 0;");

	// stop timing
	uint32_t stop = 0;
	asm volatile("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");

	// write time and data back to memory
	startClk[i] = start;
	stopClk[i] = stop;
	b[i] = v2;
}



// Main program
int main()
{
	// Number of bytes to allocate for N doubles
	size_t bytes = N*sizeof(float);

	// Allocate memory for arrays A, B, and C on host
	float *A = (float*)malloc(bytes);
	float *B = (float*)malloc(bytes);
	uint32_t *startClk = (uint32_t*) malloc(32*sizeof(uint32_t));
	uint32_t *stopClk = (uint32_t*) malloc(32*sizeof(uint32_t));
	// Allocate memory for arrays d_A, d_B, and d_C on device
	float *d_A, *d_B;
	uint32_t *startClk_g;
	uint32_t *stopClk_g;
	cudaMalloc(&d_A, bytes);
	cudaMalloc(&d_B, bytes);
	cudaMalloc(&startClk_g, 32*sizeof(uint32_t));
	cudaMalloc(&stopClk_g, 32*sizeof(uint32_t));
	// Fill host arrays A and B
	for(int i=0; i<N; i++)
	{
		A[i] = float(i);
	}

	cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

	// Set execution configuration parameters
	//		thr_per_blk: number of CUDA threads per grid block
	//		blk_in_grid: number of blocks in grid
	int thr_per_blk = 32;
	int blk_in_grid = ceil( float(N) / thr_per_blk );

	// Launch kernel
	warp_transpose<<<blk_in_grid, thr_per_blk>>>(startClk_g, stopClk_g, d_A, d_B);

	// Copy data from device array d_C to host array C
	cudaMemcpy(B, d_B, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(startClk, startClk_g, 32*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(stopClk, stopClk_g, 32*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	// Verify results
    float tolerance = 1.0e-14;
	for(int i=0; i<N; i++)
	{
        printf("%f ", B[i]);
		// if( fabs(C[i] - 3.0) > tolerance)
		// { 
		// 	printf("\nError: value of C[%d] = %d instead of 3.0\n\n", i, C[i]);
		// 	exit(1);
		// }
	}	
	float latency;
	latency = ((float)(stopClk[0]-startClk[0]))/((float)(REPEAT_TIMES*4));
	printf("\n shfl latency = %f (clk)\n", latency);
	printf("Total Clk number = %u \n", stopClk[0]-startClk[0]);
	// Free CPU memory
	free(A);
	free(B);

	// Free GPU memory
	cudaFree(d_A);
	cudaFree(d_B);

	printf("\n---------------------------\n");
	printf("__SUCCESS__\n");
	printf("---------------------------\n");
	printf("N                 = %d\n", N);
	printf("Threads Per Block = %d\n", thr_per_blk);
	printf("Blocks In Grid    = %d\n", blk_in_grid);
	printf("---------------------------\n\n");

	return 0;
}
