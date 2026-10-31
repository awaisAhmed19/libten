#include "ten/ten_tensor.h"
#include "ten/ten_status.h"
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

typedef struct TensorView {
  struct TensorStorage *storage;
  i32 rank;
  DataType dtype;
  i64 *shape;
  i64 *strides;
  i64 offset;
} TensorView;
typedef enum { FILL_FROM_SCALAR, FILL_ONES, FILL_ZEROS } FillMode;

static bool fill_by_value(void *data, const void *scalar, const DataType dtype,
                          const i64 numel);
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

static Ten_Status validate_new_tensor(const Tensor *t, DataType dtype,
                                      const i64 *shape, i32 rank, Device dev) {
  if (t == NULL)
    return (Ten_Status){TT_ERR_INVALID_ARG, "tensor is NULL"};

  if (t->storage != NULL || t->shape != NULL || t->strides != NULL)
    return (Ten_Status){TT_ERR_INVALID_STATE,
                        "destination tensor is not empty"};

  if (!T_is_ranks_valid(rank) || !T_is_shape_input_valid(shape, rank))
    return (Ten_Status){TT_ERR_INVALID_ARG, "invalid rank or shape"};

  if (T_get_dtype_size(dtype) == 0)
    return (Ten_Status){TT_ERR_DTYPE_UNSUPPORTED, "unsupported dtype"};

  if (dev.devtype != CPU)
    return (Ten_Status){TT_ERR_UNSUPPORTED, "device backend unavailable"};

  return (Ten_Status){TT_OK, "success"};
}
static bool initialize_data(void *data, DataType dtype, i64 numel,
                            FillMode mode, const void *scalar) {
  if (numel < 0)
    return false;

  if (numel == 0)
    return true;

  if (mode == FILL_FROM_SCALAR)
    return fill_by_value(data, scalar, dtype, numel);

  if (mode == FILL_ZEROS) {
    memset(data, 0, (size_t)numel * T_get_dtype_size(dtype));
    return true;
  }

  /* FILL_ONES: use a dtype-aware scalar for each case. */
  switch (dtype) {
  case F32: {
    const float one = 1.0f;
    return fill_by_value(data, &one, dtype, numel);
  }
  case F64: {
    const double one = 1.0;
    return fill_by_value(data, &one, dtype, numel);
  }
  case I8: {
    const int8_t one = 1;
    return fill_by_value(data, &one, dtype, numel);
  }
  case I16: {
    const int16_t one = 1;
    return fill_by_value(data, &one, dtype, numel);
  }
  case I32: {
    const int32_t one = 1;
    return fill_by_value(data, &one, dtype, numel);
  }
  case I64: {
    const int64_t one = 1;
    return fill_by_value(data, &one, dtype, numel);
  }
  case U8: {
    const uint8_t one = 1;
    return fill_by_value(data, &one, dtype, numel);
  }
  case U16: {
    const uint16_t one = 1;
    return fill_by_value(data, &one, dtype, numel);
  }
  case U32: {
    const uint32_t one = 1;
    return fill_by_value(data, &one, dtype, numel);
  }
  case U64: {
    const uint64_t one = 1;
    return fill_by_value(data, &one, dtype, numel);
  }
  default:
    return false;
  }
}
static void release_tensor(Tensor *t) {
  if (t == NULL)
    return;

  T_storage_release(&t->storage);
  free_shape(t->shape);
  free_strides(t->strides);

  *t = (Tensor){0};
}
static Ten_Status tensor_create_initialized(Tensor *t, DataType dtype,
                                            const i64 *shape, i32 rank,
                                            Device dev, FillMode mode,
                                            const void *scalar) {
  /* 1. Validate arguments and destination state. */
  Ten_Status status = validate_new_tensor(t, dtype, shape, rank, dev);

  if (status.stype != TT_OK)
    return status;

  /* Validate the scalar before allocating anything. */
  if (mode == FILL_FROM_SCALAR && scalar == NULL)
    return (Ten_Status){TT_ERR_INVALID_ARG, "fill scalar is NULL"};

  /* 2. Calculate element count and allocation size. */
  size_t dt_size = T_get_dtype_size(dtype);
  if (dt_size == 0)
    return (Ten_Status){TT_ERR_DTYPE_UNSUPPORTED, "unsupported dtype"};

  Tensor tmp = {0};
  tmp.rank = rank;
  tmp.offset = 0;

  if (!T_get_numel(rank, shape, &tmp.num_el)) {
    return (Ten_Status){TT_ERR_OVERFLOW,
                        "invalid shape or element count overflow"};
  }

  if ((uint64_t)tmp.num_el > SIZE_MAX / dt_size) {
    return (Ten_Status){TT_ERR_OVERFLOW, "tensor byte size overflow"};
  }

  size_t n_bytes = (size_t)tmp.num_el * dt_size;

  /* 3. Allocate shape and strides. */
  if (!allocate_shape(&tmp, shape, rank)) {
    release_tensor(&tmp);
    return (Ten_Status){TT_ERR_NO_MEMORY, "shape allocation failed"};
  }

  if (!allocate_strides(&tmp)) {
    release_tensor(&tmp);
    return (Ten_Status){TT_ERR_NO_MEMORY, "stride allocation failed"};
  }

  if (!calculate_strides(tmp.strides, tmp.shape, tmp.rank)) {
    release_tensor(&tmp);
    return (Ten_Status){TT_ERR_OVERFLOW, "stride calculation failed"};
  }

  /* Allocate storage metadata. */
  tmp.storage = calloc(1, sizeof(*tmp.storage));
  if (tmp.storage == NULL) {
    release_tensor(&tmp);
    return (Ten_Status){TT_ERR_NO_MEMORY, "storage metadata allocation failed"};
  }

  if (!storage_allocate((void **)&tmp.storage->data, dev, n_bytes)) {
    release_tensor(&tmp);
    return (Ten_Status){TT_ERR_NO_MEMORY, "tensor data allocation failed"};
  }

  /* 4. Initialize the allocated data. */
  if (!initialize_data(tmp.storage->data, dtype, tmp.num_el, mode, scalar)) {
    release_tensor(&tmp);
    return (Ten_Status){TT_ERR_INTERNAL, "tensor data initialization failed"};
  }

  /* 5. Set storage metadata only after successful initialization. */
  tmp.storage->n_bytes = n_bytes;
  tmp.storage->dtype = dtype;
  tmp.storage->device = dev;
  tmp.storage->ref_count = 1;

  /* 6. Commit the completed tensor to the caller. */
  *t = tmp;

  return (Ten_Status){TT_OK, "success"};
}

void T_destroy_tensor(Tensor *t) { release_tensor(t); }

Ten_Status T_create_tensor_storage(TensorStorage **ts, const void *input_data,
                                   DataType dtype, Device dev, size_t n_bytes) {
  if (ts == NULL)
    return status_error(TT_ERR_INVALID_ARG, "storage pointer is NULL");

  /* Do not overwrite an existing storage pointer. */
  if (*ts != NULL)
    return status_error(TT_ERR_INVALID_STATE,
                        "storage pointer is already initialized");

  size_t dt_size = T_get_dtype_size(dtype);

  if (dt_size == 0)
    return status_error(TT_ERR_DTYPE_UNSUPPORTED, "unsupported dtype");

  if (n_bytes % dt_size != 0)
    return status_error(TT_ERR_INVALID_ARG,
                        "storage size is not aligned to dtype size");

  if (n_bytes > 0 && input_data == NULL)
    return status_error(TT_ERR_INVALID_ARG, "input data is NULL");

  if (dev.devtype != CPU)
    return status_error(TT_ERR_UNSUPPORTED, "device backend unavailable");

  TensorStorage *st = calloc(1, sizeof(*st));
  if (st == NULL)
    return status_error(TT_ERR_NO_MEMORY, "storage metadata allocation failed");

  if (!storage_allocate((void **)&st->data, dev, n_bytes)) {
    free(st);
    return status_error(TT_ERR_NO_MEMORY, "storage data allocation failed");
  }

  if (!storage_copy_from_host(st->data, dev, input_data, n_bytes)) {
    storage_deallocate(st->data, dev);
    free(st);
    return status_error(TT_ERR_INTERNAL, "storage data copy failed");
  }

  st->n_bytes = n_bytes;
  st->dtype = dtype;
  st->device = dev;
  st->ref_count = 1;

  *ts = st;

  return status_ok();
}

Ten_Status T_tensor_share_storage(Tensor *dst, TensorStorage *storage,
                                  const i64 *input_shape, const i32 rank) {
  if (dst == NULL || storage == NULL)
    return status_error(TT_ERR_INVALID_ARG, "destination or storage is NULL");

  if (dst->storage != NULL || dst->shape != NULL || dst->strides != NULL)
    return status_error(TT_ERR_INVALID_STATE,
                        "destination tensor is not empty");

  if (!T_is_ranks_valid(rank))
    return status_error(TT_ERR_INVALID_ARG, "invalid rank");

  if (rank > 0 && input_shape == NULL)
    return status_error(TT_ERR_INVALID_ARG, "shape is NULL");

  i64 numel = 1;

  for (i32 i = 0; i < rank; ++i) {
    i64 dim = input_shape[i];

    if (dim < 0)
      return status_error(TT_ERR_INVALID_ARG, "negative shape dimension");

    if (dim != 0 && numel > I64_MAX / dim)
      return status_error(TT_ERR_OVERFLOW, "shape element count overflow");

    numel *= dim;
  }

  size_t dt_size = T_get_dtype_size(storage->dtype);
  if (dt_size == 0)
    return status_error(TT_ERR_DTYPE_UNSUPPORTED, "unsupported storage dtype");

  Tensor tmp = {0};
  tmp.rank = rank;
  tmp.offset = 0;
  tmp.num_el = numel;

  if ((uint64_t)tmp.num_el > SIZE_MAX / dt_size)
    return status_error(TT_ERR_OVERFLOW, "tensor byte size overflow");

  size_t required_bytes = (size_t)tmp.num_el * dt_size;

  if (required_bytes > storage->n_bytes)
    return status_error(TT_ERR_OUT_OF_RANGE, "shape exceeds storage capacity");

  if (required_bytes > 0 && storage->data == NULL)
    return status_error(TT_ERR_INVALID_STATE, "storage data is NULL");

  if (!allocate_shape(&tmp, input_shape, rank)) {
    release_tensor(&tmp);
    return status_error(TT_ERR_NO_MEMORY, "shape allocation failed");
  }

  if (!allocate_strides(&tmp)) {
    release_tensor(&tmp);
    return status_error(TT_ERR_NO_MEMORY, "stride allocation failed");
  }

  if (!calculate_strides(tmp.strides, tmp.shape, tmp.rank)) {
    release_tensor(&tmp);
    return status_error(TT_ERR_OVERFLOW, "stride calculation failed");
  }

  if (!T_storage_retain(storage)) {
    release_tensor(&tmp);
    return status_error(TT_ERR_INVALID_STATE, "storage retain failed");
  }

  tmp.storage = storage;
  *dst = tmp;

  return status_ok();
}

Ten_Status T_create_tensor_by_data(Tensor *t, const void *input_data,
                                   const DataType i_dtype, const i64 *i_shape,
                                   const i32 i_ranks, const Device dev) {
  Ten_Status status = validate_new_tensor(t, i_dtype, i_shape, i_ranks, dev);
  if (status.stype != TT_OK)
    return status;

  size_t dt_size = T_get_dtype_size(i_dtype);
  if (dt_size == 0)
    return status_error(TT_ERR_DTYPE_UNSUPPORTED, "unsupported dtype");

  Tensor tmp = {0};
  tmp.rank = i_ranks;
  tmp.offset = 0;

  if (!T_get_numel(i_ranks, i_shape, &tmp.num_el))
    return status_error(TT_ERR_OVERFLOW,
                        "invalid shape or element count overflow");

  if ((uint64_t)tmp.num_el > SIZE_MAX / dt_size)
    return status_error(TT_ERR_OVERFLOW, "tensor byte size overflow");

  size_t n_bytes = dt_size * (size_t)tmp.num_el;

  if (n_bytes > 0 && input_data == NULL)
    return status_error(TT_ERR_INVALID_ARG, "input data is NULL");

  if (!allocate_shape(&tmp, i_shape, i_ranks)) {
    release_tensor(&tmp);
    return status_error(TT_ERR_NO_MEMORY, "shape allocation failed");
  }

  if (!allocate_strides(&tmp)) {
    release_tensor(&tmp);
    return status_error(TT_ERR_NO_MEMORY, "stride allocation failed");
  }

  if (!calculate_strides(tmp.strides, tmp.shape, tmp.rank)) {
    release_tensor(&tmp);
    return status_error(TT_ERR_OVERFLOW, "stride calculation failed");
  }

  status =
      T_create_tensor_storage(&tmp.storage, input_data, i_dtype, dev, n_bytes);

  if (status.stype != TT_OK) {
    release_tensor(&tmp);
    return status;
  }
  *t = tmp;
  return status_ok();
}

Ten_Status T_clone(Tensor *dst, const Tensor *src) {
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

  if (src->num_el < 0 || src->offset < 0)
    return status_error(TT_ERR_INVALID_ARG, "invalid source tensor metadata");

  Tensor tmp = {0};
  tmp.rank = src->rank;
  tmp.offset = src->offset;
  tmp.num_el = src->num_el;

  if (!allocate_shape(&tmp, src->shape, src->rank)) {
    release_tensor(&tmp);
    return status_error(TT_ERR_NO_MEMORY, "shape allocation failed");
  }

  if (!allocate_strides(&tmp)) {
    release_tensor(&tmp);
    return status_error(TT_ERR_NO_MEMORY, "stride allocation failed");
  }

  if (src->rank > 0) {
    if (src->strides == NULL) {
      release_tensor(&tmp);
      return status_error(TT_ERR_INVALID_STATE, "source strides are NULL");
    }

    memcpy(tmp.strides, src->strides, (size_t)src->rank * sizeof(*tmp.strides));
  }

  Ten_Status status = T_create_tensor_storage(
      &tmp.storage, src->storage->data, src->storage->dtype,
      src->storage->device, src->storage->n_bytes);

  if (status.stype != TT_OK) {
    release_tensor(&tmp);
    return status;
  }
  *dst = tmp;
  return status_ok();
}

static bool fill_by_value(void *data, const void *scalar, const DataType dtype,
                          const i64 numel) {
  if (numel < 0 || scalar == NULL || (numel > 0 && data == NULL))
    return false;

  switch (dtype) {
  case F32: {
    const float value = *(const float *)scalar;
    float *dst = (float *)data;

    for (i64 i = 0; i < numel; ++i)
      dst[i] = value;
    break;
  }

  case F64: {
    const double value = *(const double *)scalar;
    double *dst = (double *)data;

    for (i64 i = 0; i < numel; ++i)
      dst[i] = value;
    break;
  }

  case I8: {
    const int8_t value = *(const int8_t *)scalar;
    int8_t *dst = (int8_t *)data;

    for (i64 i = 0; i < numel; ++i)
      dst[i] = value;
    break;
  }

  case I16: {
    const int16_t value = *(const int16_t *)scalar;
    int16_t *dst = (int16_t *)data;

    for (i64 i = 0; i < numel; ++i)
      dst[i] = value;
    break;
  }

  case I32: {
    const int32_t value = *(const int32_t *)scalar;
    int32_t *dst = (int32_t *)data;

    for (i64 i = 0; i < numel; ++i)
      dst[i] = value;
    break;
  }

  case I64: {
    const int64_t value = *(const int64_t *)scalar;
    int64_t *dst = (int64_t *)data;

    for (i64 i = 0; i < numel; ++i)
      dst[i] = value;
    break;
  }

  case U8: {
    const uint8_t value = *(const uint8_t *)scalar;
    uint8_t *dst = (uint8_t *)data;

    for (i64 i = 0; i < numel; ++i)
      dst[i] = value;
    break;
  }

  case U16: {
    const uint16_t value = *(const uint16_t *)scalar;
    uint16_t *dst = (uint16_t *)data;

    for (i64 i = 0; i < numel; ++i)
      dst[i] = value;
    break;
  }

  case U32: {
    const uint32_t value = *(const uint32_t *)scalar;
    uint32_t *dst = (uint32_t *)data;

    for (i64 i = 0; i < numel; ++i)
      dst[i] = value;
    break;
  }

  case U64: {
    const uint64_t value = *(const uint64_t *)scalar;
    uint64_t *dst = (uint64_t *)data;

    for (i64 i = 0; i < numel; ++i)
      dst[i] = value;
    break;
  }

  default:
    return false;
  }

  return true;
}
Ten_Status T_tensor_ones(Tensor *t, const DataType dtype, const i64 *shape,
                         const i32 rank, const Device dev) {
  return tensor_create_initialized(t, dtype, shape, rank, dev, FILL_ONES, NULL);
}

Ten_Status T_tensor_zero(Tensor *t, const DataType dtype, const i64 *shape,
                         const i32 rank, const Device dev) {
  return tensor_create_initialized(t, dtype, shape, rank, dev, FILL_ZEROS,
                                   NULL);
}

Ten_Status T_tensor_fill(Tensor *t, const DataType dtype, const i64 *shape,
                         const i32 rank, const Device dev, const void *scalar) {
  return tensor_create_initialized(t, dtype, shape, rank, dev, FILL_FROM_SCALAR,
                                   scalar);
}
