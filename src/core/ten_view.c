#include "internal/ten_tensor.h"
#include "ten/ten_tensor.h"
#include <string.h>

/*Alright. Let's narrow the discussion to the relationship between `tt_tensor`,
`tt_storage`, and `tt_view`, particularly how a tensor view should represent a
different interpretation of existing memory.

Your proposed design has a good separation of responsibilities:

- `tt_tensor` is the owning/reference-counted tensor handle.
- `tt_storage` represents the underlying allocation and its lifetime.
- `tt_view` is a lightweight, non-owning descriptor suitable for passing to
CPU/CUDA kernels.

The important distinction is that a tensor view and a kernel view are not
necessarily the same thing.

## 1. Your `tt_tensor` design

I like the fixed-size metadata arrays:

```
typedef struct tt_tensor {
    struct tt_storage *storage;

    tt_dtype dtype;
    int32_t rank;

    int64_t shape[TT_MAX_RANK];
    int64_t strides[TT_MAX_RANK];

    int64_t offset;   elements
int64_t numel;

uint32_t magic;
}
tt_tensor;
```

Compared with your previous dynamically allocated `shape` and `strides`, this
has several advantages:

- No separate allocations for tensor metadata.
- Creating views becomes simpler.
- Destroying a tensor only needs to release storage.
- Copying metadata becomes straightforward.
- No shape/stride allocation failures during view creation.

The trade-off is that every tensor occupies more memory, even scalars and
low-rank tensors. With `TT_MAX_RANK = 8`, I think that's reasonable.

One important rule: copying this struct with ordinary assignment must not be
treated as creating a valid independent tensor. The storage reference count must
be retained.

## 2. The `tt_view` design

Your proposed descriptor:

```
typedef struct tt_view {
  void *data; /* already includes the offset

int32_t rank;
int32_t dtype;

int64_t shape[TT_MAX_RANK];
int64_t strides[TT_MAX_RANK];
}
tt_view;
```

            This is a good kernel -
            facing representation.

            Notice the distinction :

        | Property | `tt_tensor` | `tt_view` | |
        -- -- -- -- -- -- -- -- -- -- -- -- --| -- -- -- -- -- -|
        -- -- -- -- -- -- -- -| | Storage ownership | Yes | No | |
        Reference counting | Indirectly | No | | Offset | Explicit |
        Already applied | | Shape | Yes | Yes | | Strides | Yes | Yes | |
        Dtype | Yes | Yes | | Device allocation lifetime | Managed |
        Caller - managed |

        I would keep this distinction.

        ## #Why apply the offset
    ?

    Consider a contiguous tensor
    :

``` Shape : [ 3, 4 ] Strides : [ 4, 1 ] Offset : 0
```

                                After slicing :

``` Shape : [ 2, 4 ] Strides : [ 4, 1 ] Offset : 4
```

                                The kernel -
                                facing pointer becomes :

``` view.data = (char *)storage->data + offset * dtype_size;
```

    The kernel doesn't need to know about the original storage offset.

    For an element at logical coordinates `[i, j]`:

``` element_address = view.data + i * view.strides[0] * dtype_size +
                       j * view.strides[1] * dtype_size;
```

This is clean for both CPU and CUDA.

## 3. One important issue: pointer arithmetic

I would not calculate the adjusted pointer using `void *` arithmetic.

Use:

```
static void *tt_storage_offset_ptr(
    void *data,
    int64_t offset,
    size_t dtype_size
) {
  return (unsigned char *)data + (size_t)offset * dtype_size;
}
```

    Of course,
    validate multiplication overflow and storage bounds before doing this.

    Also,
    empty tensors need special treatment
    : their data pointer may legitimately be `NULL`,
      so don't perform pointer arithmetic on it.

      ##4. The most important design decision : mutability

                                                    Suppose :

``` tt_tensor a;
tt_tensor b;

tt_tensor_slice(&b, &a, ...);
```

    Both tensors reference the same storage.

    Now :

``` tt_tensor_fill(&b, ...);
```

    Should this modify `a`
    ?

    My recommendation
    : yes.

      Views should share memory,
    and mutations should be visible through all aliases.

    However,
    this introduces a distinction between :

    -Logical tensor ownership.-
            Physical storage ownership.-
            Data mutation
                .

            The reference count only handles lifetime
                .It does not provide synchronization or prevent conflicting
                    writes
                .

            I would document that concurrent mutation through aliases requires
                al synchronization
                .

            ##5. Questions I'd settle before implementation

            There are three decisions worth making before we design the
                individual view functions
                .

            A.Negative strides

                Should slicing support reversed dimensions immediately
        ?

        Example
        :

``` Original : [ 1, 2, 3, 4, 5 ] Reversed
    : [ 5, 4, 3, 2, 1 ]

      stride = -1 offset =
          4
```

              I'd defer negative strides for the first implementation.

              B.Zero strides

                  Should broadcasting be represented through zero strides
              ?

              Example
              :

``` Original : [ 1, 2, 3 ]

                Broadcast : [ 1, 2, 3 ][1, 2, 3][1, 2, 3]

                            Shape : [ 3, 3 ] Strides
    : [ 0, 1 ]
```

      I'd support zero strides, but treat broadcast views as read-only
initially.

      C.Reshape

          Should `reshape`:

      1. Only succeed when the new shape can be represented without
          copying.2. Automatically materialize a contiguous copy when necessary.

      My recommendation is to separate these operations :

``` tt_status
      tt_tensor_reshape_view(...);
tt_status tt_tensor_reshape(...);
```

The first guarantees no copy. The second can eventually implement copy fallback.

My proposed direction: keep `tt_tensor` as the lifetime-owning descriptor,
`tt_view` as the lightweight kernel descriptor, and make all view
transformations metadata-only operations.

I'd like to settle negative strides, zero strides, and reshape semantics before
writing the public Views API. Those decisions influence almost every function in
this module.
 * */
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

  size_t max_offset = (size_t)t->offset;

  for (i32 i = 0; i < t->rank; ++i) {
    if (t->shape[i] <= 0 || t->strides[i] < 0)
      return false;

    size_t dim_offset;
    size_t next_offset;

    if (T_checked_mul_size((size_t)(t->shape[i] - 1), (size_t)t->strides[i],
                           &dim_offset) != RNG_OK)
      return false;

    if (T_checked_add_size(max_offset, dim_offset, &next_offset) != RNG_OK)
      return false;

    max_offset = next_offset;
  }

  return max_offset < capacity;
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

  if (!T_allocate_shape(&tmp, src->shape, src->rank)) {
    T_destroy_tensor(&tmp);
    return status_error(TT_ERR_NO_MEMORY, "shape allocation failed");
  }

  if (!T_allocate_stride(&tmp)) {
    T_destroy_tensor(&tmp);
    return status_error(TT_ERR_NO_MEMORY, "stride allocation failed");
  }
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

  if (!T_storage_retain(src->storage)) {
    T_destroy_tensor(&tmp);
    return status_error(TT_ERR_INVALID_STATE, "storage retain failed");
  }

  tmp.storage = src->storage;

  *dst = tmp;
  return status_ok();
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

  if (!T_allocate_shape(&tmp, src->shape, src->rank)) {
    T_destroy_tensor(&tmp);
    return status_error(TT_ERR_NO_MEMORY, "shape allocation failed");
  }

  if (!T_allocate_stride(&tmp)) {
    T_destroy_tensor(&tmp);
    return status_error(TT_ERR_NO_MEMORY, "stride allocation failed");
  }

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

  if (!T_storage_retain(src->storage)) {
    T_destroy_tensor(&tmp);
    return status_error(TT_ERR_INVALID_STATE, "storage retain failed");
  }

  tmp.storage = src->storage;

  *dst = tmp;
  return status_ok();
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

  if (!T_allocate_shape(&tmp, NULL, tmp.rank)) {
    T_destroy_tensor(&tmp);
    return status_error(TT_ERR_NO_MEMORY, "shape allocation failed");
  }

  if (!T_allocate_stride(&tmp)) {
    T_destroy_tensor(&tmp);
    return status_error(TT_ERR_NO_MEMORY, "stride allocation failed");
  }
  for (i32 i = 0, j = 0; i < src->rank; ++i) {
    if (src->shape[i] == 1)
      continue;

    tmp.shape[j] = src->shape[i];
    tmp.strides[j] = src->strides[i];
    ++j;
  }

  if (!T_storage_retain(src->storage)) {
    T_destroy_tensor(&tmp);
    return status_error(TT_ERR_INVALID_STATE, "storage retain failed");
  }

  tmp.storage = src->storage;

  *dst = tmp;
  return status_ok();
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

  if (!T_allocate_shape(&tmp, NULL, tmp.rank)) {
    T_destroy_tensor(&tmp);
    return status_error(TT_ERR_NO_MEMORY, "shape allocation failed");
  }

  if (!T_allocate_stride(&tmp)) {
    T_destroy_tensor(&tmp);
    return status_error(TT_ERR_NO_MEMORY, "stride allocation failed");
  }
  for (i32 i = 0, j = 0; i < src->rank; ++i) {
    if (i == dim)
      continue;

    tmp.shape[j] = src->shape[i];
    tmp.strides[j] = src->strides[i];
    ++j;
  }

  if (!T_storage_retain(src->storage)) {
    T_destroy_tensor(&tmp);
    return status_error(TT_ERR_INVALID_STATE, "storage retain failed");
  }

  tmp.storage = src->storage;

  *dst = tmp;
  return status_ok();
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

  if (!T_allocate_shape(&tmp, NULL, tmp.rank)) {
    T_destroy_tensor(&tmp);
    return status_error(TT_ERR_NO_MEMORY, "shape allocation failed");
  }

  if (!T_allocate_stride(&tmp)) {
    T_destroy_tensor(&tmp);
    return status_error(TT_ERR_NO_MEMORY, "stride allocation failed");
  }
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

  if (!T_storage_retain(src->storage)) {
    T_destroy_tensor(&tmp);
    return status_error(TT_ERR_INVALID_STATE, "storage retain failed");
  }

  tmp.storage = src->storage;

  *dst = tmp;
  return status_ok();
}

/* Create a sliced tensor view. */
Ten_Status T_tensor_slice(Tensor *dst, const Tensor *src, i32 rank, i64 start,
                          i64 end, i64 step);
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

  i64 new_numel;

  if (!T_get_numel(rank, shape, &new_numel))
    return status_error(TT_ERR_OVERFLOW, "invalid destination shape");

  if (new_numel != src->num_el)
    return status_error(TT_ERR_INVALID_ARG, "reshape element count mismatch");

  if (src->num_el < 0 || src->offset < 0)
    return status_error(TT_ERR_INVALID_ARG, "invalid source metadata");

  if (!T_is_dtype_valid(src->storage->dtype))
    return status_error(TT_ERR_DTYPE_UNSUPPORTED, "unsupported dtype");

  Tensor tmp = {0};

  tmp.rank = rank;
  tmp.num_el = new_numel;
  tmp.offset = src->offset;

  if (!T_allocate_shape(&tmp, shape, rank)) {
    T_destroy_tensor(&tmp);
    return status_error(TT_ERR_NO_MEMORY, "shape allocation failed");
  }

  if (!T_allocate_stride(&tmp)) {
    T_destroy_tensor(&tmp);
    return status_error(TT_ERR_NO_MEMORY, "stride allocation failed");
  }

  if (T_compute_reshape_strides(tmp.strides, src->shape, src->strides,
                                src->rank, tmp.shape, tmp.rank)
          .stype != TT_OK) {
    T_destroy_tensor(&tmp);
    return status_error(TT_ERR_INVALID_ARG,
                        "reshape cannot be represented as a view");
  }

  if (!T_storage_retain(src->storage)) {
    T_destroy_tensor(&tmp);
    return status_error(TT_ERR_INVALID_STATE, "storage retain failed");
  }

  tmp.storage = src->storage;

  *dst = tmp;

  return status_ok();
}
