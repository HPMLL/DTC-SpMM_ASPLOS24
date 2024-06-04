#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "bench_mma.fatbin.c"
extern void __device_stub__Z10mma_ubenchPmS_PfS0_S0_j(uint64_t *, uint64_t *, float *, float *, float *, uint32_t);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z10mma_ubenchPmS_PfS0_S0_j(uint64_t *__par0, uint64_t *__par1, float *__par2, float *__par3, float *__par4, uint32_t __par5){__cudaLaunchPrologue(6);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 24UL);__cudaSetupArgSimple(__par4, 32UL);__cudaSetupArgSimple(__par5, 40UL);__cudaLaunch(((char *)((void ( *)(uint64_t *, uint64_t *, float *, float *, float *, uint32_t))mma_ubench)));}
# 45 "bench_mma.cu"
void mma_ubench( uint64_t *__cuda_0,uint64_t *__cuda_1,float *__cuda_2,float *__cuda_3,float *__cuda_4,uint32_t __cuda_5)
# 46 "bench_mma.cu"
{__device_stub__Z10mma_ubenchPmS_PfS0_S0_j( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5);
# 173 "bench_mma.cu"
}
# 1 "bench_mma.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T3) {  __nv_dummy_param_ref(__T3); __nv_save_fatbinhandle_for_managed_rt(__T3); __cudaRegisterEntry(__T3, ((void ( *)(uint64_t *, uint64_t *, float *, float *, float *, uint32_t))mma_ubench), _Z10mma_ubenchPmS_PfS0_S0_j, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
