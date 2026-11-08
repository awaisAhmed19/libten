#ifndef TEN_TENSOR_BACKEND_H
#define TEN_TENSOR_BACKEND_H

#include "ten/ten_tensor.h"
typedef enum {
  TEN_COPY_HOST_TO_HOST,
  TEN_COPY_HOST_TO_DEVICE,
  TEN_COPY_DEVICE_TO_HOST,
  TEN_COPY_DEVICE_TO_DEVICE
} Ten_CopyKind;
typedef struct {
  Ten_Status (*allocate)(void **ptr, size_t n_bytes);
  Ten_Status (*deallocate)(void **ptr);
  Ten_Status (*copy)(void *dst, const void *src, size_t n_bytes,
                     Ten_CopyKind kind);
} Ten_MemoryBackend;
typedef struct {
  Ten_MemoryBackend memory;
} Ten_Backend;

const Ten_Backend *T_get_backend(Device device);

/* Existing core storage functions */
extern Ten_Status T_create_empty_storage(TensorStorage **storage,
                                         DataType dtype, Device device,
                                         size_t n_bytes);

extern Ten_Status T_create_tensor_storage(TensorStorage **storage,
                                          const void *input_data,
                                          DataType dtype, Device device,
                                          size_t n_bytes);

extern Ten_Status T_copy_tensor_data(TensorStorage *dst, const Tensor *src);

#endif
