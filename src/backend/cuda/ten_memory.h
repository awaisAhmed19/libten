#ifndef TEN_CUDA_MEMORY_H
#define TEN_CUDA_MEMORY_H

#include "internal/ten_backend.h"
#include "ten/ten_status.h"
#ifdef __cplusplus
extern "C" {
#endif
Ten_Status cuda_allocate(void **ptr, size_t n_bytes);
Ten_Status cuda_deallocate(void **ptr);
Ten_Status cuda_copy(void *dst, const void *src, size_t n_bytes,
                     Ten_CopyKind kind);
#ifdef __cplusplus
}
#endif

#endif /* TEN_CUDA_MEMORY_H */
