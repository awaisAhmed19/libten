#include "ten/ten_tensor.h"
#include "ten/ten_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#endif
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

bool T_is_ranks_valid(const i32 ranks) {
  return 0 <= ranks && ranks <= TT_MAX_RANK;
}

bool T_is_shape_input_valid(const i64 *input_shape, i32 ranks) {
  if (!T_is_ranks_valid(ranks))
    return false;

  if (ranks == 0)
    return true; // scalar tensor;

  if (input_shape == NULL) {
    fprintf(stderr,
            "ERR: passed a NULL input pointer, fn: T_is_shape_input_valid");
    return false;
  }

  // arithmetic overflow;
  i64 num_el = 1;
  for (int i = 0; i < ranks; ++i) {
    i64 dim = input_shape[i];

    if (dim < 0)
      return false;
    if (dim != 0 && num_el > I64_MAX / dim)
      return false;

    num_el *= dim;
  }
  return true;
}

bool T_get_numel(const i32 ranks, const i64 *shape, i64 *out) {
  if (out == NULL || !T_is_ranks_valid(ranks))
    return false;

  if (ranks > 0 && shape == NULL)
    return false;

  i64 num_el = 1;

  for (i32 i = 0; i < ranks; ++i) {
    i64 dim = shape[i];

    if (dim < 0)
      return false;

    if (dim != 0 && num_el > I64_MAX / dim)
      return false;

    num_el *= dim;
  }

  *out = num_el;
  return true;
}
i32 T_get_rank(const Tensor *t) {
  if (t == NULL) {
    fprintf(stderr, "ERR: invalid input tensor pointer to get rank");
    return -1;
  }
  return t->rank;
}

/* Copies the tensor shape into caller-provided storage.
 * Returns false if the tensor is invalid or capacity is insufficient.
 */
bool T_get_shape(const Tensor *t, i64 *shape_out, i32 capacity) {
  if (t == NULL) {
    fprintf(stderr, "ERR: invalid input tensor pointer to get shape");
    return false;
  }

  if (t->shape == NULL) {
    fprintf(stderr, "ERR: input tensor shape is invalid, cannot get shape");
    return false;
  }

  if (capacity <= 0) {
    fprintf(stderr, "ERR: invalid capacity");
    return false;
  }
  shape_out = malloc((size_t)t->rank * sizeof(*t->shape));
  if (shape_out == NULL) {
    fprintf(stderr, "ERR: shapes allocation failed");
    return false;
  }

  memcpy(shape_out, t->shape, (size_t)t->rank * sizeof(*t->shape));
  return true;
}

/* Returns the size of one dimension, or -1 for an invalid dimension. */
i64 T_get_dim(const Tensor *t, i32 dim);

DataType T_get_dtype(const Tensor *t) {

  if (t == NULL || t->storage == NULL) {
    fprintf(stderr, "ERR: invalid input tensor pointer or tensor-> storage is "
                    "null, to get dtype");
    return false;
  }
  return t->storage->dtype;
}

Device T_get_device(const Tensor *t) {
  Device out = {0};
  if (t == NULL || t->storage == NULL) {
    fprintf(stderr, "ERR: invalid input tensor pointer or tensor-> storage is "
                    "null, to get dtype");
    return (Device){0};
  }
  out = t->storage->device;
  return out;
}

/* Returns the size of the logical tensor data in bytes, or 0 on error. */
size_t T_get_nbytes(const Tensor *t) {
  if (t == NULL || t->storage == NULL) {
    fprintf(stderr, "ERR: invalid input tensor pointer or tensor-> storage is "
                    "null, to get dtype");
    return 0;
  }
  return t->storage->n_bytes;
}

static bool allocate_shape(Tensor *t, const i64 *shape, i32 ranks) {
  if (t == NULL)
    return false;

  if (ranks < 0 || ranks > TT_MAX_RANK)
    return false;

  if (!T_is_shape_input_valid(shape, ranks))
    return false;
  // A scalar has rank zero and no shape array.
  t->rank = ranks;
  t->shape = NULL;
  if (ranks == 0)
    return true;

  t->shape = malloc((size_t)ranks * sizeof(*t->shape));
  if (t->shape == NULL) {
    fprintf(stderr, "ERR: shapes allocation failed");
    return false;
  }

  memcpy(t->shape, shape, (size_t)ranks * sizeof(*t->shape));

  return true;
}

static bool allocate_strides(Tensor *t) {
  if (t == NULL || !T_is_ranks_valid(t->rank)) {
    return false;
  }

  if (t->rank == 0) {
    t->strides = NULL;
    // have to have a check if the shape and stride are null then the tensor
    // is scalar or not;
    return true;
  }
  t->strides = malloc((size_t)t->rank * sizeof(*t->strides));
  return t->strides != NULL;
}

static bool calculate_strides(i64 *strides, const i64 *shape, i32 rank) {
  if (!T_is_ranks_valid(rank))
    return false;

  if (rank == 0)
    return true;

  if (strides == NULL || shape == NULL)
    return false;

  strides[rank - 1] = 1;

  for (i32 i = rank - 1; i > 0; --i) {
    i64 dim = shape[i];

    if (dim != 0 && strides[i] > I64_MAX / dim)
      return false;

    strides[i - 1] = strides[i] * dim;
  }

  return true;
}
static void free_shape(i64 *shape) { free(shape); }
static void free_strides(i64 *strides) { free(strides); }
static bool storage_allocate(void **out, Device device, size_t n_bytes) {
  if (out == NULL)
    return false;

  *out = NULL;

  if (n_bytes == 0)
    return true;

  switch (device.devtype) {
  case CPU:
    *out = malloc(n_bytes);
    return *out != NULL;

#ifdef USE_CUDA
  case CUDA:
    return cudaMalloc(out, n_bytes) == cudaSuccess;
#endif

  default:
    return false;
  }
}

static bool storage_copy_from_host(void *dst, Device device, const void *src,
                                   size_t n_bytes) {
  if (n_bytes == 0)
    return true;

  if (dst == NULL || src == NULL)
    return false;

  switch (device.devtype) {
  case CPU:
    memcpy(dst, src, n_bytes);
    return true;

#ifdef USE_CUDA
  case CUDA:
    return cudaMemcpy(dst, src, n_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
#endif

  default:
    return false;
  }
}

static void storage_deallocate(void *data, Device device) {
  if (data == NULL)
    return;

  switch (device.devtype) {
  case CPU:
    free(data);
    break;

#ifdef USE_CUDA
  case CUDA:
    (void)cudaFree(data);
    break;
#endif

  default:
    fprintf(stderr, "ERR: unsupported storage device\n");
    break;
  }
}
bool T_storage_retain(TensorStorage *storage) {
  if (storage == NULL || storage->ref_count == 0)
    return false;

  if (storage->ref_count == UINT32_MAX)
    return false;

  ++storage->ref_count;
  return true;
}

void T_storage_release(TensorStorage **storage) {
  if (storage == NULL || *storage == NULL)
    return;

  TensorStorage *st = *storage;

  /* Relinquish this caller's reference. */
  *storage = NULL;

  if (st->ref_count == 0) {
    fprintf(stderr, "ERR: invalid storage reference count\n");
    return;
  }

  --st->ref_count;

  if (st->ref_count == 0) {
    storage_deallocate(st->data, st->device);
    free(st);
  }
}

bool T_create_tensor_storage(TensorStorage **ts, const void *input_data,
                             DataType dtype, Device dev, size_t n_bytes) {
  if (ts == NULL)
    return false;

  /* Do not overwrite an existing storage pointer. */
  if (*ts != NULL)
    return false;

  size_t dt_size = T_get_dtype_size(dtype);

  if (dt_size == 0)
    return false;

  if (n_bytes % dt_size != 0)
    return false;

  if (n_bytes > 0 && input_data == NULL)
    return false;

  if (dev.devtype != CPU)
    return false;

  TensorStorage *st = calloc(1, sizeof(*st));
  if (st == NULL)
    return false;

  if (!storage_allocate((void **)&st->data, dev, n_bytes)) {
    free(st);
    return false;
  }

  if (!storage_copy_from_host(st->data, dev, input_data, n_bytes)) {
    storage_deallocate(st->data, dev);
    free(st);
    return false;
  }

  st->n_bytes = n_bytes;
  st->dtype = dtype;
  st->device = dev;
  st->ref_count = 1;

  *ts = st;
  return true;
}

static void release_tensor(Tensor *t) {
  if (t == NULL)
    return;

  T_storage_release(&t->storage);
  free_shape(t->shape);
  free_strides(t->strides);

  *t = (Tensor){0};
}

void T_destroy_tensor(Tensor *t) { release_tensor(t); }

bool T_create_tensor_by_data(Tensor *t, const void *input_data,
                             const DataType i_dtype, const i64 *i_shape,
                             const i32 i_ranks, const Device dev) {
  if (t == NULL)
    return false;

  /* The output must be a fresh, empty tensor. */
  if (t->storage != NULL || t->shape != NULL || t->strides != NULL)
    return false;

  if (!T_is_ranks_valid(i_ranks) || !T_is_shape_input_valid(i_shape, i_ranks)) {
    fprintf(stderr, "ERR: invalid tensor rank or shape\n");
    return false;
  }

  size_t dt_size = T_get_dtype_size(i_dtype);
  if (dt_size == 0)
    return false;

  Tensor tmp = {0};
  tmp.rank = i_ranks;
  tmp.offset = 0;

  if (!T_get_numel(i_ranks, i_shape, &tmp.num_el))
    return false;

  if ((size_t)tmp.num_el > SIZE_MAX / dt_size)
    return false;

  size_t n_bytes = dt_size * (size_t)tmp.num_el;

  if (n_bytes > 0 && input_data == NULL)
    return false;

  if (!allocate_shape(&tmp, i_shape, i_ranks)) {
    fprintf(stderr, "ERR: shape allocation failed\n");
    release_tensor(&tmp);
    return false;
  }

  if (!allocate_strides(&tmp)) {
    fprintf(stderr, "ERR: strides allocation failed\n");
    release_tensor(&tmp);
    return false;
  }

  if (!calculate_strides(tmp.strides, tmp.shape, tmp.rank)) {
    fprintf(stderr, "ERR: stride calculation failed\n");
    release_tensor(&tmp);
    return false;
  }
  if (!T_create_tensor_storage(&tmp.storage, input_data, i_dtype, dev,
                               n_bytes)) {
    fprintf(stderr, "ERR: storage allocation failed\n");
    release_tensor(&tmp);
    return false;
  }
  *t = tmp;
  return true;
}

bool T_clone(Tensor *dst, const Tensor *src) {
  if (dst == NULL || src == NULL)
    return false;

  /* Destination must be initialized and empty. */
  if (dst->storage != NULL || dst->shape != NULL || dst->strides != NULL)
    return false;

  if (src->storage == NULL)
    return false;

  if (!T_is_ranks_valid(src->rank) ||
      !T_is_shape_input_valid(src->shape, src->rank))
    return false;

  if (src->num_el < 0 || src->offset < 0)
    return false;

  Tensor tmp = {0};

  tmp.rank = src->rank;
  tmp.offset = src->offset;
  tmp.num_el = src->num_el;

  if (!allocate_shape(&tmp, src->shape, src->rank)) {
    release_tensor(&tmp);
    return false;
  }

  if (!allocate_strides(&tmp)) {
    release_tensor(&tmp);
    return false;
  }
  if (src->rank > 0) {
    if (src->strides == NULL) {
      release_tensor(&tmp);
      return false;
    }

    memcpy(tmp.strides, src->strides, (size_t)src->rank * sizeof(*tmp.strides));
  }

  if (!T_create_tensor_storage(&tmp.storage, src->storage->data,
                               src->storage->dtype, src->storage->device,
                               src->storage->n_bytes)) {
    release_tensor(&tmp);
    return false;
  }
  *dst = tmp;
  return true;
}

bool T_tensor_share_storage(Tensor *dst, TensorStorage *storage,
                            const i64 *input_shape, const i32 rank) {

  if (dst == NULL || storage == NULL)
    return false;

  /* Destination must be initialized and empty. */
  if (dst->storage != NULL || dst->shape != NULL || dst->strides != NULL)
    return false;

  if (storage == NULL)
    return false;

  if (!T_is_ranks_valid(rank) || !T_is_shape_input_valid(input_shape, rank))
    return false;

  size_t dt_size = T_get_dtype_size(storage->dtype);
  if (dt_size == 0)
    return false;

  Tensor tmp = {0};
  tmp.rank = rank;
  tmp.offset = 0;

  if (!T_get_numel(rank, input_shape, &tmp.num_el))
    return false;

  if ((size_t)tmp.num_el > SIZE_MAX / dt_size)
    return false;
  size_t required_bytes = (size_t)tmp.num_el * dt_size;

  if (required_bytes > storage->n_bytes)
    return false;
  if (!allocate_shape(&tmp, input_shape, rank)) {
    fprintf(stderr, "ERR: shape allocation failed\n");
    release_tensor(&tmp);
    return false;
  }

  if (!allocate_strides(&tmp)) {
    fprintf(stderr, "ERR: strides allocation failed\n");
    release_tensor(&tmp);
    return false;
  }

  if (!calculate_strides(tmp.strides, tmp.shape, tmp.rank)) {
    fprintf(stderr, "ERR: stride calculation failed\n");
    release_tensor(&tmp);
    return false;
  }

  if (!T_storage_retain(storage)) {
    fprintf(stderr, "ERR: retaining the tensor storage failed\n");
    release_tensor(&tmp);
    return false;
  }
  tmp.storage = storage;
  *dst = tmp;
  return true;
}
