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
typedef struct TensorView TensorView;

typedef struct TensorStorage TensorStorage;

typedef struct Tensor {
  TensorStorage *storage;

  i32 rank;
  i64 *shape;
  i64 *strides;

  i64 offset;
  i64 num_el;
} Tensor;

typedef struct TensorView {
  void *data;
  i32 rank;
  DataType dtype;
  i64 shape[TT_MAX_RANK];
  i64 strides[TT_MAX_RANK];
  Device dev;
} TensorView;

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
extern Ten_Status T_create_tensor(Tensor *t, const void *input_data,
                                  DataType dtype, const i64 *shape, i32 rank,
                                  Device device);

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
extern Ten_Status T_tensor_ones(Tensor *t, const DataType dtype,
                                const i64 *shape, const i32 rank,
                                const Device dev);

extern Ten_Status T_tensor_zeros(Tensor *t, const DataType dtype,
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
extern bool T_is_ranks_valid(const i32 ranks);
extern bool T_is_shape_input_valid(const i64 *input_shape, i32 ranks);
extern Ten_Status T_validate_device(Device dev);
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
extern bool T_get_dim(const Tensor *t, i32 dim);

/* Returns the tensor's data type. */
extern bool T_get_dtype(const Tensor *t, DataType *out_t);

/* Returns the tensor's device. */
extern bool T_get_device(const Tensor *t, Device *out);

/* Returns the size of the logical tensor data in bytes, or 0 on error. */
extern bool T_get_nbytes(const Tensor *t, size_t *nbytes);

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
 * Returns a borrowed pointer to the first logical element.
 *
 * The returned pointer:
 * - Does not retain the underlying storage.
 * - Must not be freed by the caller.
 * - Remains valid only while the underlying storage remains alive
 *   and is not reallocated.
 * - Is accessible only for host-accessible storage.
 */
extern Ten_Status T_data(const Tensor *src, void **data);

/*
 * Returns a const pointer to the first logical element
 * for CPU tensors.
 *
 * The pointer is borrowed and must not be freed.
 * Const access does not guarantee that another alias
 * cannot modify the underlying storage.
 */
extern Ten_Status T_const_data(const Tensor *t, const void **data);
/*
 * Returns the underlying data pointer and its owning device.
 *
 * The returned pointer is borrowed from the tensor's storage.
 * No allocation or device transfer is performed.
 *
 * CPU:
 *   Returns a host-accessible pointer.
 *
 * CUDA:
 *   Returns a CUDA device pointer.
 *   The pointer must not be dereferenced by host code.
 *
 * The caller must use the appropriate backend to access
 * device-resident memory.
 *
 * Lifetime:
 *   The pointer remains valid only while the tensor's
 *   underlying storage remains alive and is not reallocated.
 *
 * The caller must not free or independently release the pointer.
 */
extern Ten_Status T_device_data(const Tensor *t, void **data, Device *device);
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
