#include "ten/ten_view.h"
#include "internal/ten_backend.h"
#include "internal/ten_tensor.h"
#include "ten/ten_tensor.h"
#include <stdint.h>
#include <string.h>

bool T_is_permutation(const i64 *dims, i32 rank) {
  if (rank < 0 || rank > TT_MAX_RANK)
    return false;

  if (rank > 0 && dims == NULL)
    return false;

  bool seen[TT_MAX_RANK] = {false};

  for (i32 i = 0; i < rank; ++i) {
    if (dims[i] < 0 || dims[i] >= rank)
      return false;

    if (seen[dims[i]])
      return false;

    seen[dims[i]] = true;
  }

  return true;
}
/* Allocates shape/stride arrays for tmp->rank; copies shape when non-NULL.
 * Destroys tmp on failure. */
static Ten_Status view_alloc(Tensor *tmp, const i64 *shape) {
  if (!T_allocate_shape(tmp, shape, tmp->rank)) {
    T_destroy_tensor(tmp);
    return status_error(TT_ERR_NO_MEMORY, "shape allocation failed");
  }

  if (!T_allocate_stride(tmp)) {
    T_destroy_tensor(tmp);
    return status_error(TT_ERR_NO_MEMORY, "stride allocation failed");
  }

  return status_ok();
}

/* Retains src's storage and commits tmp to dst. Destroys tmp on failure. */
static Ten_Status view_commit(Tensor *dst, Tensor *tmp, const Tensor *src) {
  if (!T_storage_retain(src->storage)) {
    T_destroy_tensor(tmp);
    return status_error(TT_ERR_INVALID_STATE, "storage retain failed");
  }

  tmp->storage = src->storage;
  *dst = *tmp;

  return status_ok();
}
bool T_validate_tensor_bounds(const Tensor *t) {
  if (t == NULL || t->storage == NULL)
    return false;

  if (!T_is_ranks_valid(t->rank) ||
      !T_is_shape_input_valid(t->shape, t->rank) ||
      (t->rank > 0 && t->strides == NULL))
    return false;

  if (t->offset < 0 || t->num_el < 0)
    return false;

  size_t dtype_size = T_get_dtype_size(t->storage->dtype);

  if (dtype_size == 0)
    return false;

  size_t capacity = t->storage->n_bytes / dtype_size;

  if (t->num_el == 0)
    return (size_t)t->offset <= capacity;

  if (t->storage->data == NULL)
    return false;

  i64 min_offset = t->offset;
  i64 max_offset = t->offset;

  for (i32 i = 0; i < t->rank; ++i) {
    if (t->shape[i] <= 0)
      return false;

    i64 extent = t->shape[i] - 1;
    i64 delta;

    if (T_check_i64_mul(extent, t->strides[i], &delta) != RNG_OK)
      return false;

    if (delta < 0) {
      if (T_check_i64_add(min_offset, delta, &min_offset) != RNG_OK)
        return false;
    } else {
      if (T_check_i64_add(max_offset, delta, &max_offset) != RNG_OK)
        return false;
    }
  }

  return min_offset >= 0 && (uint64_t)max_offset < capacity;
}
Ten_Status T_create_tensor_view(TensorView *tv, const Tensor *t) {

  if (tv == NULL || t == NULL)
    return (Ten_Status){TT_ERR_INVALID_ARG, "null argument"};

  if (tv->data != NULL || tv->rank != 0)
    return (Ten_Status){TT_ERR_INVALID_STATE,
                        "destination tensorview is not empty"};

  if (t->storage == NULL)
    return (Ten_Status){TT_ERR_INVALID_STATE, "source tensor has no storage"};

  if (!T_is_ranks_valid(t->rank) ||
      !T_is_shape_input_valid(t->shape, t->rank) ||
      (t->rank > 0 && t->strides == NULL)) {
    return (Ten_Status){TT_ERR_INVALID_ARG, "invalid source tensor metadata"};
  }

  size_t dtype_size = T_get_dtype_size(t->storage->dtype);

  if (dtype_size == 0)
    return (Ten_Status){TT_ERR_DTYPE_UNSUPPORTED, "unsupported source dtype"};

  if (t->storage->device.devtype != CPU)
    return (Ten_Status){TT_ERR_UNSUPPORTED, "device backend unavailable"};

  if (!T_validate_tensor_bounds(t))
    return (Ten_Status){TT_ERR_INVALID_ARG, "tensor exceeds storage bounds"};

  if (t->num_el > 0 && t->storage->data == NULL)
    return (Ten_Status){TT_ERR_INVALID_STATE,
                        "nonempty tensor has NULL storage data"};
  if (t->offset < 0)
    return (Ten_Status){TT_ERR_INVALID_ARG, "negative tensor offset"};
  TensorView tmp_view = {0};

  size_t byte_offset;
  if (T_checked_mul_size((size_t)t->offset, dtype_size, &byte_offset) !=
      RNG_OK) {
    return (Ten_Status){TT_ERR_OVERFLOW, "offset overflow"};
  }

  if (t->num_el > 0) {
    tmp_view.data = (unsigned char *)t->storage->data + byte_offset;
  }

  tmp_view.rank = t->rank;
  tmp_view.dtype = t->storage->dtype;

  if (t->rank > 0) {
    memcpy(tmp_view.shape, t->shape, (size_t)t->rank * sizeof(*t->shape));
    memcpy(tmp_view.strides, t->strides, (size_t)t->rank * sizeof(*t->strides));
  }
  *tv = tmp_view;

  return (Ten_Status){TT_OK, "success"};
}

Ten_Status T_tensor_transpose(Tensor *dst, const Tensor *src, i32 dim0,
                              i32 dim1) {
  Ten_Status status = T_validate_new_tensor(
      dst, src->storage->dtype, src->shape, src->rank, src->storage->device);
  if (status.stype != TT_OK)
    return status;

  Tensor tmp = {0};
  tmp.rank = src->rank;
  tmp.offset = src->offset;

  if (!T_get_numel(src->rank, src->shape, &tmp.num_el))
    return status_error(TT_ERR_OVERFLOW,
                        "invalid shape or element count overflow");

  if (!T_is_dtype_valid(src->storage->dtype))
    return status_error(TT_ERR_INVALID_ARG, "invalid datatype");

  if (src->storage->device.devtype != CPU)
    return status_error(TT_ERR_INVALID_ARG, "unsupported device args");

  if (src->storage->data == NULL)
    return status_error(TT_ERR_INVALID_ARG, "input data is NULL");

  status = view_alloc(&tmp, src->shape);
  if (status.stype != TT_OK)
    return status;

  if (dim0 < 0 || dim1 < 0 || dim0 >= src->rank || dim1 >= src->rank) {
    return status_error(TT_ERR_INVALID_ARG, "invalid transpose dimensions");
  }
  for (i64 i = 0; i < src->rank; ++i) {
    tmp.shape[i] = src->shape[i];
    tmp.strides[i] = src->strides[i];
  }

  tmp.shape[dim0] = src->shape[dim1];
  tmp.shape[dim1] = src->shape[dim0];

  tmp.strides[dim0] = src->strides[dim1];
  tmp.strides[dim1] = src->strides[dim0];

  return view_commit(dst, &tmp, src);
}

/* Permute dimensions according to the supplied ordering. */
Ten_Status T_tensor_permute(Tensor *dst, const Tensor *src, const i64 *dims,
                            i32 rank) {
  Ten_Status status = T_validate_new_tensor(
      dst, src->storage->dtype, src->shape, src->rank, src->storage->device);
  if (status.stype != TT_OK)
    return status;

  Tensor tmp = {0};
  tmp.rank = src->rank;
  tmp.offset = src->offset;
  tmp.num_el = src->num_el;

  if (!T_is_dtype_valid(src->storage->dtype))
    return status_error(TT_ERR_INVALID_ARG, "invalid datatype");

  if (src->storage->device.devtype != CPU)
    return status_error(TT_ERR_INVALID_ARG, "unsupported device args");

  if (src->storage->n_bytes > 0 && src->storage->data == NULL) {
    return status_error(TT_ERR_INVALID_ARG, "input data is NULL");
  }

  status = view_alloc(&tmp, src->shape);
  if (status.stype != TT_OK)
    return status;

  if (rank != src->rank) {
    T_destroy_tensor(&tmp);
    return status_error(TT_ERR_INVALID_ARG,
                        "input rank and src rank must match");
  }

  if (!T_is_shape_input_valid(dims, rank)) {
    return status_error(TT_ERR_INVALID_ARG, "invalid Permute dimensions");
  }
  if (!T_is_permutation(dims, rank)) {
    return status_error(TT_ERR_INVALID_ARG,
                        "invalid Permute dimensions dims not seen in shape");
  }
  for (i64 i = 0; i < src->rank; ++i) {
    tmp.shape[i] = src->shape[dims[i]];
    tmp.strides[i] = src->strides[dims[i]];
  }

  return view_commit(dst, &tmp, src);
}

/* Remove dimensions of size 1. */
Ten_Status T_tensor_squeeze(Tensor *dst, const Tensor *src) {

  Ten_Status status = T_validate_new_tensor(
      dst, src->storage->dtype, src->shape, src->rank, src->storage->device);
  if (status.stype != TT_OK)
    return status;

  if (!T_is_dtype_valid(src->storage->dtype))
    return status_error(TT_ERR_INVALID_ARG, "invalid datatype");

  i32 count_ones = 0;
  for (i32 i = 0; i < src->rank; ++i) {
    if (src->shape[i] == 1)
      count_ones++;
  }

  Tensor tmp = {0};
  tmp.rank = src->rank - count_ones;
  tmp.offset = src->offset;
  tmp.num_el = src->num_el;

  if (!T_is_ranks_valid(tmp.rank))
    return status_error(TT_ERR_INVALID_ARG, "invalid resulting rank");

  if (src->storage->n_bytes > 0 && src->storage->data == NULL) {
    T_destroy_tensor(&tmp);
    return status_error(TT_ERR_INVALID_ARG, "input data is NULL");
  }

  status = view_alloc(&tmp, NULL);
  if (status.stype != TT_OK)
    return status;

  for (i32 i = 0, j = 0; i < src->rank; ++i) {
    if (src->shape[i] == 1)
      continue;

    tmp.shape[j] = src->shape[i];
    tmp.strides[j] = src->strides[i];
    ++j;
  }

  return view_commit(dst, &tmp, src);
}

/* Remove a specific dimension of size 1. */
Ten_Status T_tensor_squeeze_dim(Tensor *dst, const Tensor *src, i32 dim) {
  Ten_Status status = T_validate_new_tensor(
      dst, src->storage->dtype, src->shape, src->rank, src->storage->device);
  if (status.stype != TT_OK)
    return status;

  if (!T_is_dtype_valid(src->storage->dtype))
    return status_error(TT_ERR_INVALID_ARG, "invalid datatype");

  if (dim < 0 || dim >= src->rank)
    return status_error(TT_ERR_INVALID_ARG, "invalid squeeze dimension");

  if (src->shape[dim] != 1)
    return status_error(TT_ERR_INVALID_ARG,
                        "cannot squeeze dimension with size != 1");
  Tensor tmp = {0};
  tmp.rank = src->rank - 1;
  tmp.offset = src->offset;
  tmp.num_el = src->num_el;

  if (!T_is_ranks_valid(tmp.rank))
    return status_error(TT_ERR_INVALID_ARG, "invalid resulting rank");

  if (src->storage->n_bytes > 0 && src->storage->data == NULL) {
    T_destroy_tensor(&tmp);
    return status_error(TT_ERR_INVALID_ARG, "input data is NULL");
  }

  status = view_alloc(&tmp, NULL);
  if (status.stype != TT_OK)
    return status;

  for (i32 i = 0, j = 0; i < src->rank; ++i) {
    if (i == dim)
      continue;

    tmp.shape[j] = src->shape[i];
    tmp.strides[j] = src->strides[i];
    ++j;
  }

  return view_commit(dst, &tmp, src);
}

/* Insert a dimension of size 1. */
Ten_Status T_tensor_unsqueeze(Tensor *dst, const Tensor *src, i32 dim) {

  Ten_Status status = T_validate_new_tensor(
      dst, src->storage->dtype, src->shape, src->rank, src->storage->device);
  if (status.stype != TT_OK)
    return status;

  if (!T_is_dtype_valid(src->storage->dtype))
    return status_error(TT_ERR_INVALID_ARG, "invalid datatype");

  if (dim < 0 || dim > src->rank) {
    return status_error(TT_ERR_INVALID_ARG, "invalid unsqueeze dimension");
  }

  Tensor tmp = {0};
  tmp.rank = src->rank + 1;
  tmp.offset = src->offset;
  tmp.num_el = src->num_el;

  if (!T_is_ranks_valid(tmp.rank)) {
    return status_error(TT_MAX_RANK, "max rank reached");
  }

  if (src->storage->n_bytes > 0 && src->storage->data == NULL) {
    T_destroy_tensor(&tmp);
    return status_error(TT_ERR_INVALID_ARG, "input data is NULL");
  }

  status = view_alloc(&tmp, NULL);
  if (status.stype != TT_OK)
    return status;

  i64 new_stride = (dim < src->rank) ? src->strides[dim] * src->shape[dim] : 1;
  for (i32 i = 0, j = 0; i < tmp.rank; ++i) {
    if (i == dim) {
      tmp.shape[i] = 1;
      tmp.strides[i] = new_stride;
    } else {
      tmp.shape[i] = src->shape[j];
      tmp.strides[i] = src->strides[j];
      ++j;
    }
  }

  return view_commit(dst, &tmp, src);
}

Ten_Status T_is_contiguous_layout(const i64 *shape, const i64 *strides,
                                  i32 rank) {
  if (!T_is_ranks_valid(rank) || !T_is_shape_input_valid(shape, rank))
    return status_error(TT_ERR_INVALID_ARG, "invalid source shape");

  if (rank > 0 && strides == NULL)
    return status_error(TT_ERR_INVALID_ARG, "source strides are NULL");

  i64 expected_stride = 1;

  for (i32 i = rank - 1; i >= 0; --i) {
    // Singleton dimensions do not affect contiguity.
    if (shape[i] == 1)
      continue;

    if (strides[i] != expected_stride)
      return status_error(TT_ERR_NOT_CONTIGUOUS, " strides mismatch");

    if (shape[i] > INT64_MAX / expected_stride)
      return status_error(TT_ERR_OVERFLOW, "shape element may overflow");

    expected_stride *= shape[i];
  }

  return status_ok();
}

Ten_Status T_compute_reshape_strides(i64 *dst_strides, const i64 *src_shape,
                                     const i64 *src_strides, i32 src_rank,
                                     const i64 *dst_shape, i32 dst_rank) {

  if (!T_is_ranks_valid(src_rank) ||
      !T_is_shape_input_valid(src_shape, src_rank))
    return status_error(TT_ERR_INVALID_ARG, "invalid source shape");

  if (src_rank > 0 && src_strides == NULL)
    return status_error(TT_ERR_INVALID_ARG, "source strides are NULL");

  if (!T_is_ranks_valid(dst_rank) ||
      !T_is_shape_input_valid(dst_shape, dst_rank))
    return status_error(TT_ERR_INVALID_ARG, "invalid destination shape");

  if (dst_rank > 0 && dst_strides == NULL)
    return status_error(TT_ERR_INVALID_ARG, "destination strides are NULL");

  i64 src_numel;
  i64 dst_numel;

  if (!T_get_numel(src_rank, src_shape, &src_numel) ||
      !T_get_numel(dst_rank, dst_shape, &dst_numel))
    return status_error(TT_ERR_OVERFLOW, "element count calculation failed");

  if (src_numel != dst_numel)
    return status_error(TT_ERR_INVALID_ARG, "reshape element count mismatch");

  // Empty tensors have no elements to reinterpret.
  if (src_numel == 0) {
    if (!T_calculate_strides(dst_strides, dst_shape, dst_rank))
      return status_error(TT_ERR_OVERFLOW, "stride calculation failed");

    return status_ok();
  }

  // A scalar or a tensor containing only singleton dimensions
  // is also compatible with a contiguous reshape.
  if (T_is_contiguous_layout(src_shape, src_strides, src_rank).stype != TT_OK)
    return status_error(
        TT_ERR_INVALID_ARG,
        "source layout cannot be reshaped by this implementation");

  // Generate the destination's contiguous strides.
  if (!T_calculate_strides(dst_strides, dst_shape, dst_rank))
    return status_error(TT_ERR_OVERFLOW, "stride calculation failed");

  return status_ok();
}

/* Reshape without copying, if the new shape is representable by strides. */
Ten_Status T_tensor_reshape_view(Tensor *dst, const Tensor *src,
                                 const i64 *shape, i32 rank) {
  if (dst == NULL || src == NULL)
    return status_error(TT_ERR_INVALID_ARG, "source or destination is NULL");

  if (dst->storage != NULL || dst->shape != NULL || dst->strides != NULL)
    return status_error(TT_ERR_INVALID_STATE,
                        "destination tensor is not empty");

  if (src->storage == NULL)
    return status_error(TT_ERR_INVALID_STATE, "source tensor has no storage");

  if (!T_is_ranks_valid(src->rank) ||
      !T_is_shape_input_valid(src->shape, src->rank))
    return status_error(TT_ERR_INVALID_ARG, "invalid source tensor shape");

  if (src->rank > 0 && src->strides == NULL)
    return status_error(TT_ERR_INVALID_STATE, "source strides are NULL");

  if (!T_is_ranks_valid(rank) || !T_is_shape_input_valid(shape, rank))
    return status_error(TT_ERR_INVALID_ARG, "invalid destination shape");

  if (!T_is_dtype_valid(src->storage->dtype))
    return status_error(TT_ERR_DTYPE_UNSUPPORTED, "unsupported dtype");

  if (src->num_el < 0 || src->offset < 0)
    return status_error(TT_ERR_INVALID_ARG, "invalid source metadata");

  /* Validate that the source tensor fits inside its storage. */
  if (!T_validate_tensor_bounds(src))
    return status_error(TT_ERR_INVALID_ARG,
                        "source tensor exceeds storage bounds");

  i64 new_numel;

  if (!T_get_numel(rank, shape, &new_numel))
    return status_error(TT_ERR_OVERFLOW, "destination element count overflow");

  if (new_numel != src->num_el)
    return status_error(TT_ERR_INVALID_ARG, "reshape element count mismatch");

  Tensor tmp = {0};

  tmp.rank = rank;
  tmp.num_el = new_numel;
  tmp.offset = src->offset;

  /* Allocate destination metadata. */
  Ten_Status status = view_alloc(&tmp, shape);

  if (status.stype != TT_OK)
    return status;

  /* Calculate strides for the requested shape. */
  status = T_compute_reshape_strides(tmp.strides, src->shape, src->strides,
                                     src->rank, tmp.shape, tmp.rank);

  if (status.stype != TT_OK) {
    T_destroy_tensor(&tmp);
    return status;
  }

  /* Retain source storage and commit the view. */
  return view_commit(dst, &tmp, src);
}
// Ten_Status T_tensor_contiguous_logical_copy(Tensor *dst, const Tensor *src)
// {
//   if (dst == NULL || src == NULL)
//     return status_error(TT_ERR_INVALID_ARG, "source or destination is
//     NULL");
//
//   if (dst->storage == NULL || src->storage == NULL)
//     return status_error(TT_ERR_INVALID_STATE,
//                         "source or destination has no storage");
//
//   if (!T_is_dtype_valid(src->storage->dtype))
//     return status_error(TT_ERR_DTYPE_UNSUPPORTED, "unsupported source
//     dtype");
//
//   if (dst->storage->dtype != src->storage->dtype)
//     return status_error(TT_ERR_DTYPE_MISMATCH, "storage dtype mismatch");
//
//   if (dst->storage->device.devtype != CPU ||
//       src->storage->device.devtype != CPU)
//     return status_error(TT_ERR_UNSUPPORTED, "CPU copy backend only");
//
//   if (!T_is_ranks_valid(src->rank) ||
//       !T_is_shape_input_valid(src->shape, src->rank))
//     return status_error(TT_ERR_INVALID_ARG, "invalid source shape");
//
//   if (src->rank != dst->rank || src->num_el != dst->num_el)
//     return status_error(TT_ERR_INVALID_ARG, "tensor shape mismatch");
//
//   if (src->rank > 0 && src->strides == NULL)
//     return status_error(TT_ERR_INVALID_STATE, "source strides are NULL");
//
//   if (src->num_el < 0 || src->offset < 0)
//     return status_error(TT_ERR_INVALID_ARG, "invalid source metadata");
//
//   if (!T_validate_tensor_bounds(src) || !T_validate_tensor_bounds(dst))
//     return status_error(TT_ERR_INVALID_ARG, "tensor exceeds storage
//     bounds");
//
//   size_t dtype_size = T_get_dtype_size(src->storage->dtype);
//
//   if (dtype_size == 0)
//     return status_error(TT_ERR_DTYPE_UNSUPPORTED, "unsupported dtype");
//
//   if (src->num_el == 0)
//     return status_ok();
//
//   i64 indices[TT_MAX_RANK] = {0};
//
//   for (i64 n = 0; n < src->num_el; ++n) {
//     i64 src_offset = src->offset;
//
//     for (i32 d = 0; d < src->rank; ++d)
//       src_offset += indices[d] * src->strides[d];
//
//     memcpy((char *)dst->storage->data + (size_t)n * dtype_size,
//            (char *)src->storage->data + (size_t)src_offset * dtype_size,
//            dtype_size);
//
//     /* Increment multidimensional indices. */
//     for (i32 d = src->rank - 1; d >= 0; --d) {
//       indices[d]++;
//
//       if (indices[d] < src->shape[d])
//         break;
//
//       indices[d] = 0;
//     }
//   }
//
//   return status_ok();
// }
Ten_Status T_tensor_contiguous_materialize(Tensor *dst, const Tensor *src) {
  if (dst == NULL || src == NULL)
    return status_error(TT_ERR_INVALID_ARG, "source or destination is NULL");
  if (dst->storage != NULL || dst->shape != NULL || dst->strides != NULL)
    return status_error(TT_ERR_INVALID_STATE,
                        "destination tensor is not empty");
  if (!T_is_dtype_valid(src->storage->dtype))
    return status_error(TT_ERR_DTYPE_UNSUPPORTED, "unsupported source dtype");

  if (src->storage == NULL)
    return status_error(TT_ERR_INVALID_STATE, "source has no storage");
  if (src->storage->device.devtype != CPU)
    return status_error(TT_ERR_UNSUPPORTED, "CPU copy backend only");

  if (!T_is_ranks_valid(src->rank) ||
      !T_is_shape_input_valid(src->shape, src->rank))
    return status_error(TT_ERR_INVALID_ARG, "invalid source shape");

  if (src->rank > 0 && src->strides == NULL)
    return status_error(TT_ERR_INVALID_STATE, "source strides are NULL");

  if (src->num_el < 0 || src->offset < 0)
    return status_error(TT_ERR_INVALID_ARG, "invalid source metadata");

  if (!T_validate_tensor_bounds(src))
    return status_error(TT_ERR_INVALID_ARG, "tensor exceeds storage bounds");

  size_t dtype_size = T_get_dtype_size(src->storage->dtype);

  if (dtype_size == 0)
    return status_error(TT_ERR_DTYPE_UNSUPPORTED, "unsupported dtype");

  size_t n_bytes;

  if ((u64)src->num_el > SIZE_MAX / dtype_size)
    return status_error(TT_ERR_OVERFLOW, "tensor byte size overflow");

  n_bytes = (size_t)src->num_el * dtype_size;
  Tensor tmp = {0};

  tmp.rank = src->rank;
  tmp.num_el = src->num_el;
  tmp.offset = 0;
  /* Allocate metadata. */
  if (!T_allocate_shape(&tmp, src->shape, src->rank)) {
    T_destroy_tensor(&tmp);
    return status_error(TT_ERR_NO_MEMORY, "shape allocation failed");
  }

  if (!T_allocate_stride(&tmp)) {
    T_destroy_tensor(&tmp);
    return status_error(TT_ERR_NO_MEMORY, "stride allocation failed");
  }

  /* Generate contiguous strides. */
  if (!T_calculate_strides(tmp.strides, tmp.shape, tmp.rank)) {
    T_destroy_tensor(&tmp);
    return status_error(TT_ERR_OVERFLOW,
                        "contiguous stride calculation failed");
  }
  if (T_create_empty_storage(&tmp.storage, src->storage->dtype,
                             src->storage->device, n_bytes)
          .stype != TT_OK) {
    T_destroy_tensor(&tmp);
    return status_error(TT_ERR_INTERNAL, "storage allocation failed");
  }

  i64 indices[TT_MAX_RANK] = {0};

  for (i64 n = 0; n < src->num_el; ++n) {
    i64 src_offset = src->offset;

    for (i32 d = 0; d < src->rank; ++d)
      src_offset += indices[d] * src->strides[d];

    memcpy((char *)tmp.storage->data + (size_t)n * dtype_size,
           (char *)src->storage->data + (size_t)src_offset * dtype_size,
           dtype_size);

    /* Increment multidimensional indices. */
    for (i32 d = src->rank - 1; d >= 0; --d) {
      indices[d]++;

      if (indices[d] < src->shape[d])
        break;

      indices[d] = 0;
    }
  }

  *dst = tmp;
  return status_ok();
}
Ten_Status T_tensor_slice(Tensor *dst, const Tensor *src, i32 dim, i64 start,
                          i64 end, i64 step) {
  // 1. Validate source.
  if (src == NULL || src->storage == NULL || src->shape == NULL ||
      src->strides == NULL)
    return status_error(TT_ERR_INVALID_ARG, "invalid source tensor");

  if (!T_is_ranks_valid(src->rank))
    return status_error(TT_ERR_INVALID_ARG, "invalid source rank");

  if (src->storage->n_bytes > 0 && src->storage->data == NULL)
    return status_error(TT_ERR_INVALID_ARG, "source data is NULL");

  Ten_Status status = T_validate_new_tensor(
      dst, src->storage->dtype, src->shape, src->rank, src->storage->device);
  if (status.stype != TT_OK)
    return status;

  if (!T_is_dtype_valid(src->storage->dtype)) {
    return status_error(TT_ERR_DTYPE_UNSUPPORTED, "Invalid src Data type");
  }

  if (dim < 0 || dim >= src->rank)
    return status_error(TT_ERR_INVALID_ARG, "invalid slice dimension");

  if (step == 0)
    return status_error(TT_ERR_INVALID_ARG, "slice step cannot be zero");

  // Avoid overflow when calculating abs(step).
  if (step == (-I64_MAX - 1))
    return status_error(TT_ERR_INVALID_ARG, "slice step overflow");

  i64 dim_size = src->shape[dim];
  i64 length = 0;
  if (step > 0) {

    if (start < 0 || start > dim_size || end < 0 || end > dim_size ||
        start > end)
      return status_error(TT_ERR_INVALID_ARG, "invalid slice bounds");

    if (start < end)
      length = 1 + (end - 1 - start) / step;

  } else {

    if (start < -1 || start >= dim_size || end < -1 || end >= dim_size ||
        start <= end)
      return status_error(TT_ERR_INVALID_ARG, "invalid reverse slice bounds");

    if (start > end)
      length = 1 + (start - 1 - end) / (-step);
  }
  Tensor tmp = {0};
  tmp.rank = src->rank;
  tmp.offset = src->offset;

  status = view_alloc(&tmp, src->shape);
  if (status.stype != TT_OK)
    return status;
  memcpy(tmp.strides, src->strides, sizeof(i64) * src->rank);

  tmp.shape[dim] = length;

  if (T_check_i64_mul(src->strides[dim], step, &tmp.strides[dim]) != RNG_OK) {
    T_destroy_tensor(&tmp);
    return status_error(TT_ERR_INVALID_ARG, "slice stride overflow");
  }

  if (length > 0) {
    i64 offset_delta;

    if (T_check_i64_mul(start, src->strides[dim], &offset_delta) != RNG_OK ||
        T_check_i64_add(src->offset, offset_delta, &tmp.offset) != RNG_OK) {
      T_destroy_tensor(&tmp);
      return status_error(TT_ERR_INVALID_ARG, "slice offset overflow");
    }
  }

  tmp.num_el = 1;

  for (i32 i = 0; i < tmp.rank; ++i) {
    if (T_check_i64_mul(tmp.num_el, tmp.shape[i], &tmp.num_el) != RNG_OK) {
      T_destroy_tensor(&tmp);
      return status_error(TT_ERR_INVALID_ARG, "element count overflow");
    }
  }

  return view_commit(dst, &tmp, src);
}
