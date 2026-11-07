#ifndef TEN_TENSOR_H
#define TEN_TENSOR_H

#include "ten_status.h"
#include "ten_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 1. Opaque public types and constants
 *  */

#define TT_MAX_RANK 8

typedef struct Tensor Tensor;
typedef struct TensorStorage TensorStorage;
typedef struct TensorView TensorView;

/*
 * 2. Construction
 *  */

/*
 * Creates a tensor by copying host input data into newly
 * allocated storage.
 *
 * Requirements:
 * - t must point to a zero-initialized Tensor.
 * - input_shape must contain rank dimensions when rank > 0.
 * - input_data must be non-NULL when the tensor requires
 *   nonzero bytes.
 *
 * On success, t owns its storage reference and metadata.
 * On failure, t remains empty.
 */
extern Ten_Status T_create_tensor_by_data(Tensor *t, const void *input_data,
                                          DataType dtype, const i64 *shape,
                                          i32 rank, Device device);

/*
 * Creates a tensor sharing an existing storage allocation.
 *
 * The tensor receives a new storage reference and contiguous
 * strides. The storage must have enough capacity for the
 * requested shape.
 *
 * On success, dst owns its storage reference and metadata.
 * On failure, dst remains empty.
 */
extern Ten_Status T_tensor_share_storage(Tensor *dst, TensorStorage *storage,
                                         const i64 *shape, i32 rank);

/* Creates a deep copy with independent storage. */
extern Ten_Status T_clone(Tensor *dst, const Tensor *src);

/*
 * Tensor constructors.
 *
 * These functions allocate new tensors; they do not modify
 * an existing tensor's values.
 */
extern Ten_Status T_tensor_fill_ones(Tensor *t, const DataType dtype,
                                     const i64 *shape, const i32 rank,
                                     const Device dev);

extern Ten_Status T_tensor_fill_zeros(Tensor *t, const DataType dtype,
                                      const i64 *shape, const i32 rank,
                                      const Device dev);

extern Ten_Status T_tensor_fill_value(Tensor *t, const DataType dtype,
                                      const i64 *shape, const i32 rank,
                                      const Device dev, const void *scalar);

/*
 * 3. Lifetime
 *  */

/* Destroys tensor metadata and releases its storage reference. */
extern void T_destroy_tensor(Tensor *t);

/*
 * 4. Metadata queries
 *  */

/* Returns the tensor rank, or -1 for an invalid tensor. */
extern i32 T_get_rank(const Tensor *t);

/*
 * Copies the tensor shape into caller-provided storage.
 *
 * Returns false if the tensor is invalid or capacity is
 * insufficient.
 *
 * For scalar tensors (rank == 0), no dimensions are copied.
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
 * 5. Layout queries
 *  */

/*
 * Returns the number of logical elements for the given shape.
 * Returns false for an invalid shape or element-count overflow.
 */
extern bool T_get_numel(i32 rank, const i64 *shape, i64 *out);

/*
 * Checks whether the tensor has contiguous memory layout.
 *
 * On success, writes the result to *out and returns TT_OK.
 * A valid non-contiguous tensor is not an error; *out is false.
 * Returns an error if the tensor or its metadata is invalid.
 */
extern Ten_Status T_tensor_is_contiguous(const Tensor *t, bool *out);

/*
 * 6. Host data access
 *  */

/*
 * Returns a mutable pointer to the first logical element
 * for CPU tensors.
 *
 * The pointer is borrowed and must not be freed.
 * Modifications through this pointer are visible through
 * every tensor sharing the same underlying storage.
 *
 * The pointer does not imply that the tensor is contiguous.
 * Check T_tensor_is_contiguous() before flat iteration.
 */
extern void *T_data(Tensor *t);

/*
 * Returns a const pointer to the first logical element
 * for CPU tensors.
 *
 * The pointer is borrowed and must not be freed.
 * Const access does not guarantee that another alias
 * cannot modify the underlying storage.
 */
extern const void *T_const_data(const Tensor *t);

/*
 * 7. Data movement
 *  */

/*
 * Creates a new contiguous tensor preserving the logical
 * contents of the source.
 *
 * The destination owns independent storage.
 */
extern Ten_Status T_tensor_contiguous_materialize(Tensor *dst,
                                                  const Tensor *src);

/*
 * 8. View / alias operations
 *  */

/*
 * View operations create new tensor metadata while sharing
 * the source tensor's underlying storage.
 *
 * - Each view owns its own shape and stride metadata.
 * - Each view retains a reference to shared storage.
 * - Data modifications through one view are visible through
 *   all aliases.
 * - Destroying one view releases only its own storage reference.
 * - Storage is released when its final reference is destroyed.
 * - Views do not implement copy-on-write.
 */

/*
 * Creates a TensorView from an existing tensor.
 *
 * NOTE: TensorView is opaque. This API needs a complete
 * public lifecycle design to be usable by external consumers.
 */
extern Ten_Status T_create_tensor_view(TensorView *tv, const Tensor *t);

/* Transpose two dimensions; the result shares source storage. */
extern Ten_Status T_tensor_transpose(Tensor *dst, const Tensor *src, i32 dim0,
                                     i32 dim1);

/* Permute dimensions; the result shares source storage. */
extern Ten_Status T_tensor_permute(Tensor *dst, const Tensor *src,
                                   const i64 *dims, i32 rank);

/* Remove dimensions of size 1; the result shares source storage. */
extern Ten_Status T_tensor_squeeze(Tensor *dst, const Tensor *src);

/* Remove a specific dimension of size 1; the result shares source storage. */
extern Ten_Status T_tensor_squeeze_dim(Tensor *dst, const Tensor *src, i32 dim);

/* Insert a dimension of size 1; the result shares source storage. */
extern Ten_Status T_tensor_unsqueeze(Tensor *dst, const Tensor *src, i32 dim);

/* Create a sliced tensor view sharing source storage. */
extern Ten_Status T_tensor_slice(Tensor *dst, const Tensor *src, i32 dim,
                                 i64 start, i64 end, i64 step);

/*
 * 9. Reshape operations
 *  */

/*
 * Reshape without copying, if the new shape is representable
 * by strides.
 *
 * The resulting tensor shares source storage.
 */
extern Ten_Status T_tensor_reshape_view(Tensor *dst, const Tensor *src,
                                        const i64 *shape, i32 rank);

/*
 * Creates a deep copy with contiguous strides and
 * independent storage.
 */
extern Ten_Status T_tensor_reshape(Tensor *dst, const Tensor *src,
                                   const i64 *shape, i32 rank);

#ifdef __cplusplus
}
#endif

#endif /* TEN_TENSOR_H */
