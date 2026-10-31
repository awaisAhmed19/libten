

#ifndef TEN_TENSOR_H
#define TEN_TENSOR_H

#include "ten_status.h"
#include "ten_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TT_MAX_RANK 8
typedef struct Tensor Tensor;
typedef struct TensorStorage TensorStorage;

/*
 * Validation
 *  */

/* Returns whether ranks is within [0, TT_MAX_RANK]. */
extern bool T_is_ranks_valid(i32 ranks);

/* Validates shape pointer, dimensions, rank, and element-count overflow. */
extern bool T_is_shape_input_valid(const i64 *input_shape, i32 ranks);

extern Ten_Status T_create_tensor_storage(TensorStorage **ts,
                                          const void *input_data,
                                          DataType dtype, Device dev,
                                          size_t n_bytes);

/*
 * Construction
 *  */

/*
 * Creates a tensor by copying host input data into newly allocated storage.
 *
 * Requirements:
 * - t must point to a zero-initialized Tensor.
 * - input_shape must contain i_ranks dimensions when i_ranks > 0.
 * - input_data must be non-NULL when the tensor requires nonzero bytes.
 *
 * On success, t owns its storage reference and metadata.
 * On failure, t remains empty.
 */
extern Ten_Status T_create_tensor_by_data(Tensor *t, const void *input_data,
                                          DataType dtype, const i64 *shape,
                                          i32 rank, Device device);

/*
 * Creates a tensor sharing an existing storage allocation.
 * The tensor receives a new storage reference and contiguous strides.
 * The storage must have enough capacity for the requested shape.
 *
 * On success, dst owns its storage reference and metadata.
 * On failure, dst remains empty.
 */
extern Ten_Status T_tensor_share_storage(Tensor *dst, TensorStorage *storage,
                                         const i64 *shape, i32 rank);

/* Creates a deep copy with independent storage. */
extern Ten_Status T_clone(Tensor *dst, const Tensor *src);

/*
 * Lifetime
 *  */

/* Destroys tensor metadata and releases its storage reference. */
extern void T_destroy_tensor(Tensor *t);

/*
 * Metadata queries
 *  */

/* Returns the number of logical elements, or false for an invalid tensor. */
extern bool T_get_numel(const i32 rank, const i64 *shape, i64 *out);

/* Returns the tensor rank, or -1 for an invalid tensor. */
extern i32 T_get_rank(const Tensor *t);

/* Copies the tensor shape into caller-provided storage.
 * Returns false if the tensor is invalid or capacity is insufficient.
 */
extern bool T_get_shape(const Tensor *t, i64 *shape_out, i32 capacity);

/* Returns the size of one dimension, or -1 for an invalid dimension. */
extern i64 T_get_dim(const Tensor *t, i32 dim);

/* Returns the tensor's data type. */
extern DataType T_get_dtype(const Tensor *t);

/* Returns the tensor's device. */
extern Device T_get_device(const Tensor *t);

/* Returns the size of the logical tensor data in bytes, or 0 on error. */
extern size_t T_get_nbytes(const Tensor *t);

/*
 * Data access
 *  */

/*
 * Returns a pointer to the first logical element for CPU tensors.
 * Returns NULL for invalid tensors, non-CPU tensors, or unavailable data.
 *
 * The pointer is borrowed; callers must not free it.
 */
extern void *T_data(Tensor *t);
extern const void *T_const_data(const Tensor *t);

/*
 * Utilities
 *  */

/* Allocates rows * cols bytes. Prefer removing this generic helper
 * if it is not used by the public API.
 */
extern u0 *T_allocate_tensor_data(size_t rows, size_t cols);

#ifdef __cplusplus
}
#endif

#endif /* TEN_TENSOR_H */
