#ifndef TEN_VIEW_H
#define TEN_VIEW_H

#include "ten_status.h"
#include "ten_types.h"

#ifdef __cplusplus
extern "C" {
#endif


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
#ifdef __cplusplus
}
#endif

#endif /* TEN_TENSOR_H */