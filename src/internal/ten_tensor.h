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

typedef struct TensorView {
  void *data;
  i32 rank;
  DataType dtype;
  i64 shape[TT_MAX_RANK];
  i64 strides[TT_MAX_RANK];
} TensorView;

extern bool T_is_ranks_valid(const i32 ranks);
extern bool T_is_shape_input_valid(const i64 *input_shape, i32 ranks);
extern bool T_allocate_shape(Tensor *t, const i64 *shape, i32 ranks);
extern bool T_allocate_stride(Tensor *t);
extern bool T_calculate_strides(i64 *strides, const i64 *shape, i32 rank);
extern bool T_storage_retain(TensorStorage *storage);
extern void T_storage_release(TensorStorage **storage);
extern Ten_Status T_create_empty_storage(TensorStorage **storage,
                                         DataType dtype, Device device,
                                         size_t n_bytes);
extern bool T_validate_tensor_bounds(const Tensor *t);
extern Ten_Status T_validate_new_tensor(const Tensor *t, DataType dtype,
                                        const i64 *shape, i32 rank, Device dev);
#endif
