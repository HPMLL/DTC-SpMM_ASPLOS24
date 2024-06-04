nvcc -gencode arch=compute_89,code=sm_89 -ptx bench_shfl.cu -o bench_shfl.ptx
nvcc -gencode arch=compute_89,code=sm_89 bench_shfl.cu -o bench_shfl
nvcc -gencode arch=compute_89,code=sm_89 -ptx bench_mma.cu -o bench_mma.ptx
nvcc -keep -gencode arch=compute_89,code=sm_89 bench_mma.cu -o bench_mma
nvcc -gencode arch=compute_89,code=sm_89 bench_offset.cu -o bench_offset
nvcc -gencode arch=compute_89,code=sm_89 -ptx bench_offset.cu -o bench_offset.ptx

