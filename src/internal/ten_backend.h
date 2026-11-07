#ifndef TEN_TENSOR_BACKEND_H
#define TEN_TENSOR_BACKEND_H

#include "ten/ten_tensor.h"

extern Ten_Status T_create_empty_storage(TensorStorage **storage,
                                         DataType dtype, Device device,
                                         size_t n_bytes);

extern Ten_Status T_create_tensor_storage(TensorStorage **ts,
                                          const void *input_data,
                                          DataType dtype, Device dev,
                                          size_t n_bytes);
extern Ten_Status T_copy_tensor_data(TensorStorage *dst, const Tensor *src);
#endif
