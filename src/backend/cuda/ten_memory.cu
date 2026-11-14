#include <cuda_runtime_api.h>

#include "ten/ten_status.h"
#include "ten_memory.h"

Ten_Status cuda_allocate(void **ptr, size_t n_bytes) {
  if (ptr == NULL)
    return status_error(TT_ERR_INVALID_ARG, "Output pointer is NULL");

  *ptr = NULL;

  if (n_bytes == 0)
    return status_ok();

  cudaError_t err = cudaMalloc(ptr, n_bytes);

  if (err != cudaSuccess) {
    enum Ten_StatusType code =
        (err == cudaErrorMemoryAllocation) ? TT_ERR_CUDA : TT_ERR_NO_MEMORY;

    return status_error(code, cudaGetErrorString(err));
  }
  return status_ok();
}

Ten_Status cuda_deallocate(void **ptr) {
  if (ptr == NULL)
    return status_error(TT_ERR_INVALID_ARG, "Pointer argument is NULL");

  if (*ptr == NULL)
    return status_ok();

  cudaError_t err = cudaFree(*ptr);

  if (err != cudaSuccess) {
    enum Ten_StatusType code =
        (err == cudaErrorMemoryAllocation) ? TT_ERR_CUDA : TT_ERR_NO_MEMORY;

    return status_error(code, cudaGetErrorString(err));
  }

  *ptr = NULL;

  return status_ok();
}

Ten_Status cuda_copy(void *dst, const void *src, size_t n_bytes,
                     Ten_CopyKind kind) {
  if (n_bytes == 0)
    return status_ok();

  if (dst == NULL || src == NULL)
    return status_error(TT_ERR_INVALID_ARG,
                        "Source or destination pointer is NULL");

  cudaMemcpyKind cuda_kind;

  switch (kind) {
  case TEN_COPY_HOST_TO_HOST:
    cuda_kind = cudaMemcpyHostToHost;
    break;

  case TEN_COPY_HOST_TO_DEVICE:
    cuda_kind = cudaMemcpyHostToDevice;
    break;

  case TEN_COPY_DEVICE_TO_HOST:
    cuda_kind = cudaMemcpyDeviceToHost;
    break;

  case TEN_COPY_DEVICE_TO_DEVICE:
    cuda_kind = cudaMemcpyDeviceToDevice;
    break;

  default:
    return status_error(TT_ERR_INVALID_ARG, "Invalid memory copy direction");
  }

  cudaError_t err = cudaMemcpy(dst, src, n_bytes, cuda_kind);

  if (err != cudaSuccess) {
    enum Ten_StatusType code =
        (err == cudaErrorMemoryAllocation) ? TT_ERR_CUDA : TT_ERR_NO_MEMORY;

    return status_error(code, cudaGetErrorString(err));
  }
  return status_ok();
}
