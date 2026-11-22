#ifndef TEN_VIEW_H
#define TEN_VIEW_H

#include "internal/ten_tensor.h"
#include "ten_status.h"
#include "ten_types.h"
#ifdef __cplusplus
extern "C" {
#endif

/*
 * 8. View / alias operations
 */

/*
 * Tensor views create alternative interpretations of existing tensor
 * storage through shape, stride, and offset metadata.
 *
 * Tensor-level view operations:
 * - Create new tensor metadata while sharing the source tensor's storage.
 * - Each tensor view maintains its own shape, strides, and offset.
 * - Each tensor view retains a reference to the shared storage.
 * - Data modifications through one view are visible through all aliases.
 * - Destroying one view releases only its own storage reference.
 * - Storage is released when its final reference is destroyed.
 * - Views do not implement copy-on-write.
 *
 * TensorView descriptors:
 * - Provide a device-aware description of an existing memory region.
 * - Do not allocate, own, or retain the underlying storage.
 * - Borrow a pointer to the first logical element of the source tensor.
 * - Must not outlive the source tensor or its underlying storage.
 */

/*
 * Creates a TensorView descriptor from an existing tensor.
 *
 * TensorView is a non-owning, device-aware descriptor.
 *
 * Pointer semantics:
 * - data points to the first logical element, accounting for the
 *   source tensor's storage offset.
 * - dev identifies the device on which the pointer resides.
 * - CPU pointers are host-accessible.
 * - CUDA pointers are device pointers and must not be dereferenced
 *   by host code.
 * - Shape, strides, dtype, and device describe how to interpret data.
 *
 * Lifetime and ownership:
 * - The descriptor does not retain or release storage.
 * - The source tensor and its storage must remain alive while the
 *   descriptor is in use.
 * - Destroying or modifying the source tensor in a way that releases
 *   or reallocates its storage invalidates the descriptor.
 *
 * API status:
 * - TensorView is a public, non-opaque structure.
 * - Its initialization and lifecycle are caller-managed.
 * - Callers must initialize descriptors to zero before creation.
 * - No dedicated destruction operation is required because the
 *   descriptor does not own resources.
 *
 * Device constraints:
 * - Creating a descriptor does not perform a device transfer.
 * - Pointer accessibility is determined by dev.
 * - Consumers must use the appropriate backend when accessing
 *   device-resident memory.
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
