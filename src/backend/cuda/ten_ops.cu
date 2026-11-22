#include "./ten_ops.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#define TEN_CUDA_FILL_THREADS 256
#define TEN_CUDA_FILL_MAX_BLOCKS 65535
__global__ void cuda_fill_kernel(unsigned char *dst, uint64_t scalar,
                                 size_t dtype_size, size_t numel) {

  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)blockDim.x * gridDim.x;

  const unsigned char *scalar_bytes = (const unsigned char *)&scalar;

  for (; idx < numel; idx += stride) {
    size_t offset = idx * dtype_size;

    for (size_t j = 0; j < dtype_size; ++j)
      dst[offset + j] = scalar_bytes[j];
  }
}

extern "C" Ten_Status cuda_storage_fill(void *data, const void *host_scalar,
                                        DataType dtype, size_t numel) {

  if (host_scalar == NULL || (numel > 0 && data == NULL))
    return status_error(TT_ERR_INVALID_ARG, "invalid fill arguments");

  size_t dtype_size = T_get_dtype_size(dtype);

  if (dtype_size == 0 || dtype_size > sizeof(uint64_t) ||
      T_is_dtype_complex(dtype))
    return status_error(TT_ERR_DTYPE_UNSUPPORTED, "unsupported fill dtype");

  if (numel == 0)
    return status_ok();

  size_t total_bytes = 0;

  if (T_checked_mul_size(numel, dtype_size, &total_bytes) != RNG_OK)
    return status_error(TT_ERR_OVERFLOW, "fill size overflow");

  uint64_t scalar = 0;
  memcpy(&scalar, host_scalar, dtype_size);

  size_t required_blocks = 1 + (numel - 1) / TEN_CUDA_FILL_THREADS;

  unsigned int blocks =
      (unsigned int)(required_blocks > TEN_CUDA_FILL_MAX_BLOCKS
                         ? TEN_CUDA_FILL_MAX_BLOCKS
                         : required_blocks);

  cuda_fill_kernel<<<blocks, TEN_CUDA_FILL_THREADS>>>(
      (unsigned char *)data, scalar, dtype_size, numel);

  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess)
    return status_error(TT_ERR_INTERNAL, cudaGetErrorString(err));

  err = cudaDeviceSynchronize();

  if (err != cudaSuccess)
    return status_error(TT_ERR_INTERNAL, cudaGetErrorString(err));

  return status_ok();
}
