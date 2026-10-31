#ifndef TEN_TENSOR_INTERNAL_H
#define TEN_TENSOR_INTERNAL_H

#include "ten/ten_tensor.h"
struct TensorStorage {
  u0 *data; /* may be NULL only when n_bytes == 0 */
  size_t n_bytes;
  u32 ref_count; // TODO:u32 ref_count is not thread-safe. eventual
                 // support for concurrent access, need to consider an atomic
                 // reference count or explicitly document that reference
                 // management is not thread-safe.
  Device device;
  DataType dtype;
};

typedef struct Tensor {
  struct TensorStorage *storage;
  i32 rank;
  i64 *shape;
  i64 *strides;
  i64 offset; /* elements */
  i64 num_el;
} Tensor;
#endif
