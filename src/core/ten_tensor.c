#include "ten/ten_tensor.h"
#include "internal/ten_backend.h"
#include "internal/ten_tensor.h"
#include "ten/ten_status.h"
#include "ten/ten_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#endif

typedef enum { FROM_SCALAR, ONES, ZEROS } FillMode;

static bool by_value(void *data, const void *scalar, const DataType dtype,
                     const i64 numel);
bool T_is_ranks_valid(const i32 ranks) {
  return 0 <= ranks && ranks <= TT_MAX_RANK;
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

static inline bool T_storage_is_host_accessible(const TensorStorage *storage) {
  return storage != NULL && storage->device.devtype == CPU;
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

  i64 num_el;
  return T_get_numel(ranks, input_shape, &num_el);
}
i32 T_get_rank(const Tensor *t) {
  if (t == NULL) {
    fprintf(stderr, "ERR: invalid input tensor pointer to get rank");
    return -1;
  }
  return t->rank;
}

bool T_get_shape(const Tensor *t, i64 *shape_out, i32 capacity) {
  if (t == NULL || capacity < 0) {
    fprintf(stderr, "ERR: invalid argument to T_get_shape\n");
    return false;
  }

  if (t->rank < 0) {
    fprintf(stderr, "ERR: invalid tensor rank\n");
    return false;
  }

  // Scalars have no dimensions to copy.
  if (t->rank == 0)
    return true;

  if (shape_out == NULL) {
    fprintf(stderr, "ERR: shape output pointer is NULL\n");
    return false;
  }

  if (t->shape == NULL) {
    fprintf(stderr, "ERR: tensor shape is NULL\n");
    return false;
  }

  if (capacity < t->rank) {
    fprintf(stderr, "ERR: insufficient output buffer capacity\n");
    return false;
  }

  memcpy(shape_out, t->shape, (size_t)t->rank * sizeof(*shape_out));

  return true;
}

/* Returns the size of one dimension, or -1 for an invalid dimension. */
bool T_get_dim(const Tensor *t, i32 dim) {
  if (t == NULL) {
    fprintf(stderr, "ERR: invalid argument to T_get_shape\n");
    return false;
  }

  if (t->rank < 0) {
    fprintf(stderr, "ERR: invalid tensor rank\n");
    return false;
  }
  dim = t->rank;
  return true;
}

bool T_get_dtype(const Tensor *t, DataType *out_t) {
  if (out_t == NULL) {
    fprintf(stderr, "ERR: output dtype pointer is NULL\n");
    return false;
  }

  *out_t = DTYPE_COUNT;

  if (t == NULL || t->storage == NULL) {
    fprintf(stderr, "ERR: invalid tensor or missing storage to get dtype\n");
    return false;
  }

  if (!T_is_dtype_valid(t->storage->dtype)) {
    fprintf(stderr, "ERR: invalid input dtype\n");
    return false;
  }

  *out_t = t->storage->dtype;
  return true;
}

bool T_get_device(const Tensor *t, Device *out) {
  if (out == NULL) {
    fprintf(stderr, "ERR: output device pointer is NULL\n");
    return false;
  }

  *out = (Device){.devtype = CPU, .id = -1};

  if (t == NULL || t->storage == NULL) {
    fprintf(stderr, "ERR: invalid tensor or missing storage to get device\n");
    return false;
  }

  *out = t->storage->device;

  return true;
}

bool T_get_nbytes(const Tensor *t, size_t *nbytes) {
  if (nbytes == NULL) {
    fprintf(stderr, "ERR: output byte-count pointer is NULL\n");
    return false;
  }

  *nbytes = 0;

  if (t == NULL || t->storage == NULL) {
    fprintf(stderr, "ERR: invalid tensor or missing storage to get nbytes\n");
    return false;
  }

  size_t dtype_size = T_get_dtype_size(t->storage->dtype);

  if (dtype_size == 0) {
    fprintf(stderr, "ERR: unsupported tensor dtype\n");
    return false;
  }

  if (t->num_el < 0 || (uint64_t)t->num_el > SIZE_MAX / dtype_size) {
    fprintf(stderr, "ERR: tensor byte size overflow\n");
    return false;
  }

  *nbytes = (size_t)t->num_el * dtype_size;

  return true;
}

bool T_allocate_shape(Tensor *t, const i64 *shape, i32 ranks) {
  if (t == NULL)
    return false;

  if (ranks < 0 || ranks > TT_MAX_RANK)
    return false;

  if (ranks > 0 && shape != NULL) {
    if (!T_is_shape_input_valid(shape, ranks))
      return false;
  }
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
  if (shape != NULL) {
    memcpy(t->shape, shape, (size_t)ranks * sizeof(*t->shape));
  }

  return true;
}

bool T_allocate_stride(Tensor *t) {
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

bool T_calculate_strides(i64 *strides, const i64 *shape, i32 rank) {
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
static Ten_Status storage_allocate(void **out, Device device, size_t n_bytes) {
  if (out == NULL)
    return status_error(TT_ERR_INVALID_ARG, "output pointer is NULL");

  *out = NULL;

  const Ten_Backend *backend = T_get_backend(device);

  if (backend == NULL)
    return status_error(TT_ERR_UNSUPPORTED,
                        "requested device backend is unavailable");

  if (backend->memory.allocate == NULL)
    return status_error(TT_ERR_INVALID_STATE,
                        "backend allocation function is missing");

  return backend->memory.allocate(out, n_bytes);
}

static Ten_Status storage_copy_from_host(void *dst, Device device,
                                         const void *src, size_t n_bytes) {
  if (n_bytes == 0)
    return status_ok();

  if (dst == NULL || src == NULL)
    return status_error(TT_ERR_INVALID_ARG,
                        "source or destination pointer is NULL");

  const Ten_Backend *backend = T_get_backend(device);

  if (backend == NULL)
    return status_error(TT_ERR_UNSUPPORTED,
                        "requested device backend is unavailable");

  if (backend->memory.copy == NULL)
    return status_error(TT_ERR_INVALID_STATE,
                        "backend copy function is missing");

  Ten_CopyKind kind;

  switch (device.devtype) {
  case CPU:
    kind = TEN_COPY_HOST_TO_HOST;
    break;

  case CUDA:
    kind = TEN_COPY_HOST_TO_DEVICE;
    break;

  default:
    return status_error(TT_ERR_UNSUPPORTED, "unsupported device type");
  }

  return backend->memory.copy(dst, src, n_bytes, kind);
}
static Ten_Status storage_copy(void *dst, Device dst_device, const void *src,
                               Device src_device, size_t n_bytes) {
  if (n_bytes == 0)
    return status_ok();

  if (dst == NULL || src == NULL)
    return status_error(TT_ERR_INVALID_ARG,
                        "source or destination pointer is NULL");

  if (dst_device.devtype == CPU && src_device.devtype == CPU) {
    const Ten_Backend *backend = T_get_backend(dst_device);

    if (backend == NULL || backend->memory.copy == NULL)
      return status_error(TT_ERR_INVALID_STATE, "CPU backend copy unavailable");

    return backend->memory.copy(dst, src, n_bytes, TEN_COPY_HOST_TO_HOST);
  }

  Ten_CopyKind kind;
  Device backend_device;

  // if (src_device.devtype == CPU && dst_device.devtype == CPU) {
  //   kind = TEN_COPY_HOST_TO_HOST;
  //   backend_device = dst_device;
  // } else
  if (src_device.devtype == CPU && dst_device.devtype == CUDA) {
    kind = TEN_COPY_HOST_TO_DEVICE;
    backend_device = dst_device;
  } else if (src_device.devtype == CUDA && dst_device.devtype == CPU) {
    kind = TEN_COPY_DEVICE_TO_HOST;
    backend_device = src_device;
  } else if (src_device.devtype == CUDA && dst_device.devtype == CUDA) {
    if (src_device.id != dst_device.id)
      return status_error(TT_ERR_UNSUPPORTED,
                          "cross-device CUDA copies are not supported");
    kind = TEN_COPY_DEVICE_TO_DEVICE;
    backend_device = dst_device;
  } else {
    return status_error(TT_ERR_UNSUPPORTED, "unsupported memory copy devices");
  }

  const Ten_Backend *backend = T_get_backend(backend_device);
  if (backend == NULL)
    return status_error(TT_ERR_UNSUPPORTED,
                        "memory copy backend is unavailable");

  if (backend->memory.copy == NULL)
    return status_error(TT_ERR_INVALID_STATE,
                        "backend memory copy function is missing");

  return backend->memory.copy(dst, src, n_bytes, kind);
}
static Ten_Status storage_deallocate(void *data, Device device) {
  if (data == NULL)
    return status_ok();

  const Ten_Backend *backend = T_get_backend(device);

  if (backend == NULL || backend->memory.deallocate == NULL)
    return status_error(TT_ERR_UNSUPPORTED, "Storage backend is unavailable");

  /*
   * The backend deallocate function accepts void **.
   * local pointer is used because the storage structure
   * owns the original data pointer.
   */
  void *ptr = data;

  return backend->memory.deallocate(&ptr);
}
static void storage_destroy(TensorStorage **storage) {
  if (storage == NULL || *storage == NULL)
    return;

  TensorStorage *st = *storage;
  *storage = NULL;

  if (st->data != NULL) {
    Ten_Status status = storage_deallocate(st->data, st->device);

    if (status.stype != TT_OK)
      T_PRINT_STATUS(status);
  }

  free(st);
}
Ten_Status T_data(const Tensor *src, const void **data) {
  if (src == NULL)
    return status_error(TT_ERR_INVALID_ARG, "source tensor is NULL");

  if (data == NULL)
    return status_error(TT_ERR_INVALID_ARG, "data output pointer is NULL");

  *data = NULL;

  if (src->storage == NULL)
    return status_error(TT_ERR_INVALID_STATE, "tensor storage is NULL");

  if (src->storage->device.devtype != CPU)
    return status_error(TT_ERR_UNSUPPORTED, "tensor is not stored on CPU");

  if (src->num_el > 0 && src->storage->data == NULL)
    return status_error(TT_ERR_INVALID_STATE, "tensor storage data is NULL");

  *data = src->storage->data;

  return status_ok();
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

  if (st->ref_count == 0) {
    fprintf(stderr, "ERR: invalid storage reference count\n");
    return;
  }

  *storage = NULL;
  --st->ref_count;

  if (st->ref_count == 0)
    storage_destroy(&st);
}

Ten_Status T_validate_new_tensor(const Tensor *t, DataType dtype,
                                 const i64 *shape, i32 rank, Device dev) {
  if (t == NULL)
    return (Ten_Status){TT_ERR_INVALID_ARG, "tensor is NULL"};

  if (t->storage != NULL || t->shape != NULL || t->strides != NULL ||
      t->rank != 0 || t->offset != 0 || t->num_el != 0)
    return status_error(TT_ERR_INVALID_STATE,
                        "destination tensor is not empty");

  if (!T_is_ranks_valid(rank) || !T_is_shape_input_valid(shape, rank))
    return (Ten_Status){TT_ERR_INVALID_ARG, "invalid rank or shape"};

  if (T_get_dtype_size(dtype) == 0)
    return (Ten_Status){TT_ERR_DTYPE_UNSUPPORTED, "unsupported dtype"};

  return (Ten_Status){TT_OK, "success"};
}

static bool initialize_data(void *data, DataType dtype, i64 numel,
                            FillMode mode, const void *scalar) {
  if (numel < 0)
    return false;

  if (numel == 0)
    return true;

  if (mode == FROM_SCALAR)
    return by_value(data, scalar, dtype, numel);

  if (mode == ZEROS) {
    memset(data, 0, (size_t)numel * T_get_dtype_size(dtype));
    return true;
  }

  /* ONES: build a dtype-aware "one", then fill with it. */
  union {
    float as_f32;
    double as_f64;
    int8_t as_i8;
    int16_t as_i16;
    int32_t as_i32;
    int64_t as_i64;
    uint8_t as_u8;
    uint16_t as_u16;
    uint32_t as_u32;
    uint64_t as_u64;
  } one = {0};

  switch (dtype) {
  case F32:
    one.as_f32 = 1.0f;
    break;
  case F64:
    one.as_f64 = 1.0;
    break;
  case I8:
    one.as_i8 = 1;
    break;
  case I16:
    one.as_i16 = 1;
    break;
  case I32:
    one.as_i32 = 1;
    break;
  case I64:
    one.as_i64 = 1;
    break;
  case U8:
    one.as_u8 = 1;
    break;
  case U16:
    one.as_u16 = 1;
    break;
  case U32:
    one.as_u32 = 1;
    break;
  case U64:
    one.as_u64 = 1;
    break;
  default:
    return false;
  }

  return by_value(data, &one, dtype, numel);
}
static void release_tensor(Tensor *t) {
  if (t == NULL)
    return;
  T_storage_release(&t->storage);
  free_shape(t->shape);
  free_strides(t->strides);

  *t = (Tensor){0};
}
/* Allocates shape/strides for tmp and fills contiguous row-major strides.
 * Releases tmp on failure. */
static Ten_Status tensor_init_layout(Tensor *tmp, const i64 *shape, i32 rank) {
  if (!T_allocate_shape(tmp, shape, rank)) {
    release_tensor(tmp);
    return status_error(TT_ERR_NO_MEMORY, "shape allocation failed");
  }

  if (!T_allocate_stride(tmp)) {
    release_tensor(tmp);
    return status_error(TT_ERR_NO_MEMORY, "stride allocation failed");
  }

  if (!T_calculate_strides(tmp->strides, tmp->shape, tmp->rank)) {
    release_tensor(tmp);
    return status_error(TT_ERR_OVERFLOW, "stride calculation failed");
  }

  return status_ok();
}
static Ten_Status tensor_create_initialized(Tensor *t, DataType dtype,
                                            const i64 *shape, i32 rank,
                                            Device dev, FillMode mode,
                                            const void *scalar) {
  Ten_Status status = T_validate_new_tensor(t, dtype, shape, rank, dev);

  if (status.stype != TT_OK)
    return status;

  if (mode == FROM_SCALAR && scalar == NULL)
    return (Ten_Status){TT_ERR_INVALID_ARG, "fill scalar is NULL"};

  size_t dt_size = T_get_dtype_size(dtype);
  if (dt_size == 0)
    return (Ten_Status){TT_ERR_DTYPE_UNSUPPORTED, "unsupported dtype"};
  if (dev.devtype != CPU)
    return status_error(TT_ERR_UNSUPPORTED,
                        "host initialization is not supported for this device");
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

  status = tensor_init_layout(&tmp, shape, rank);
  if (status.stype != TT_OK)
    return status;

  tmp.storage = calloc(1, sizeof(*tmp.storage));

  if (tmp.storage == NULL) {
    release_tensor(&tmp);
    return status_error(TT_ERR_NO_MEMORY, "storage metadata allocation failed");
  }

  tmp.storage->n_bytes = n_bytes;
  tmp.storage->dtype = dtype;
  tmp.storage->device = dev;
  tmp.storage->ref_count = 1;

  status = storage_allocate((void **)&tmp.storage->data, dev, n_bytes);
  if (status.stype != TT_OK) {
    release_tensor(&tmp);
    return status;
  }

  if (!initialize_data(tmp.storage->data, dtype, tmp.num_el, mode, scalar)) {
    release_tensor(&tmp);
    return (Ten_Status){TT_ERR_INTERNAL, "tensor data initialization failed"};
  }

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

  if (n_bytes > 0 && input_data == NULL)
    return status_error(TT_ERR_INVALID_ARG, "input data is NULL");

  Ten_Status status = T_create_empty_storage(ts, dtype, dev, n_bytes);
  if (status.stype != TT_OK)
    return status;

  status = storage_copy_from_host((*ts)->data, dev, input_data, n_bytes);
  if (status.stype != TT_OK) {
    T_storage_release(ts);
    return status_error(status.stype, status.msg);
  }

  return status_ok();
}
Ten_Status T_create_empty_storage(TensorStorage **storage, DataType dtype,
                                  Device device, size_t n_bytes) {
  if (storage == NULL)
    return status_error(TT_ERR_INVALID_ARG, "storage pointer is NULL");

  if (*storage != NULL)
    return status_error(TT_ERR_INVALID_STATE, "storage is already initialized");

  if (!T_is_dtype_valid(dtype))
    return status_error(TT_ERR_DTYPE_UNSUPPORTED, "unsupported dtype");

  size_t dtype_size = T_get_dtype_size(dtype);

  if (dtype_size == 0)
    return status_error(TT_ERR_DTYPE_UNSUPPORTED, "unsupported dtype");

  if (n_bytes % dtype_size != 0)
    return status_error(TT_ERR_INVALID_ARG,
                        "storage size is not dtype aligned");

  TensorStorage *tmp = calloc(1, sizeof(*tmp));

  if (tmp == NULL)
    return status_error(TT_ERR_NO_MEMORY, "storage metadata allocation failed");

  Ten_Status status = storage_allocate((void **)&tmp->data, device, n_bytes);
  if (status.stype != TT_OK) {
    free(tmp);
    return status;
  }

  tmp->n_bytes = n_bytes;
  tmp->dtype = dtype;
  tmp->device = device;
  tmp->ref_count = 1;

  *storage = tmp;

  return status_ok();
}

Ten_Status T_copy_tensor_data(TensorStorage *dst, const Tensor *src) {
  if (dst == NULL || src == NULL)
    return status_error(TT_ERR_INVALID_ARG, "source or destination is NULL");

  if (src->storage == NULL)
    return status_error(TT_ERR_INVALID_STATE, "source has no storage");

  if (!T_is_dtype_valid(src->storage->dtype))
    return status_error(TT_ERR_DTYPE_UNSUPPORTED, "unsupported source dtype");

  if (dst->dtype != src->storage->dtype)
    return status_error(TT_ERR_DTYPE_MISMATCH, "storage dtype mismatch");

  if (!T_is_ranks_valid(src->rank) ||
      !T_is_shape_input_valid(src->shape, src->rank))
    return status_error(TT_ERR_INVALID_ARG, "invalid source shape");

  if (src->rank > 0 && src->strides == NULL)
    return status_error(TT_ERR_INVALID_STATE, "source strides are NULL");

  const size_t dtype_size = T_get_dtype_size(src->storage->dtype);
  if (dtype_size == 0)
    return status_error(TT_ERR_DTYPE_UNSUPPORTED, "unsupported dtype");

  i64 expected_numel;
  if (!T_get_numel(src->rank, src->shape, &expected_numel))
    return status_error(TT_ERR_OVERFLOW, "source element count overflow");

  if (src->num_el < 0 || expected_numel != src->num_el)
    return status_error(TT_ERR_INVALID_STATE, "source element count mismatch");

  if ((uint64_t)src->num_el > SIZE_MAX / dtype_size)
    return status_error(TT_ERR_OVERFLOW, "copy byte size overflow");

  const size_t required_bytes = (size_t)src->num_el * dtype_size;
  if (dst->n_bytes < required_bytes)
    return status_error(TT_ERR_OUT_OF_RANGE,
                        "destination storage is too small");

  if (required_bytes == 0)
    return status_ok();

  if (dst->data == NULL || src->storage->data == NULL)
    return status_error(TT_ERR_INVALID_STATE,
                        "source or destination data is NULL");

  if (src->storage->n_bytes % dtype_size != 0)
    return status_error(TT_ERR_INVALID_STATE, "invalid source storage size");

  if (src->storage->device.devtype == CUDA && dst->device.devtype == CUDA &&
      src->storage->device.id != dst->device.id)
    return status_error(TT_ERR_UNSUPPORTED,
                        "cross-device CUDA tensor copies are not supported");

  if (!T_validate_tensor_bounds(src))
    return status_error(TT_ERR_OUT_OF_RANGE,
                        "source tensor exceeds storage bounds");

  bool is_contiguous = false;
  Ten_Status status = T_tensor_is_contiguous(src, &is_contiguous);
  if (status.stype != TT_OK)
    return status;

  /* A contiguous logical tensor can be copied with one backend transfer. */
  if (is_contiguous) {
    size_t byte_offset;
    if (T_checked_mul_size((size_t)src->offset, dtype_size, &byte_offset) !=
        RNG_OK)
      return status_error(TT_ERR_OVERFLOW, "source byte offset overflow");

    const unsigned char *src_data =
        (const unsigned char *)src->storage->data + byte_offset;

    return storage_copy(dst->data, dst->device, src_data, src->storage->device,
                        required_bytes);
  }

  /*
   * The generic strided gather currently runs on the host. Stage the source
   * allocation to host memory when necessary, gather into a contiguous host
   * buffer, then transfer that buffer to the destination device if needed.
   */
  const Device host_device = {.devtype = CPU, .id = 0};
  const unsigned char *host_src = (const unsigned char *)src->storage->data;
  unsigned char *source_staging = NULL;

  if (src->storage->device.devtype != CPU) {
    source_staging = malloc(src->storage->n_bytes);
    if (source_staging == NULL)
      return status_error(TT_ERR_NO_MEMORY, "source staging allocation failed");

    status = storage_copy(source_staging, host_device, src->storage->data,
                          src->storage->device, src->storage->n_bytes);
    if (status.stype != TT_OK) {
      free(source_staging);
      return status;
    }

    host_src = source_staging;
  }

  unsigned char *host_dst = (unsigned char *)dst->data;
  unsigned char *destination_staging = NULL;

  if (dst->device.devtype != CPU) {
    destination_staging = malloc(required_bytes);
    if (destination_staging == NULL) {
      free(source_staging);
      return status_error(TT_ERR_NO_MEMORY,
                          "destination staging allocation failed");
    }

    host_dst = destination_staging;
  }

  for (i64 linear = 0; linear < src->num_el; ++linear) {
    i64 remaining = linear;
    i64 src_index = src->offset;

    for (i32 d = src->rank - 1; d >= 0; --d) {
      const i64 dim_size = src->shape[d];
      if (dim_size <= 0) {
        status = status_error(TT_ERR_INVALID_STATE,
                              "invalid dimension in non-empty source tensor");
        goto copy_cleanup;
      }

      const i64 coord = remaining % dim_size;
      remaining /= dim_size;

      i64 delta;
      if (T_check_i64_mul(coord, src->strides[d], &delta) != RNG_OK) {
        status = status_error(TT_ERR_OVERFLOW,
                              "source element offset multiplication overflow");
        goto copy_cleanup;
      }

      if (T_check_i64_add(src_index, delta, &src_index) != RNG_OK) {
        status = status_error(TT_ERR_OVERFLOW,
                              "source element offset addition overflow");
        goto copy_cleanup;
      }
    }

    if (src_index < 0 ||
        (uint64_t)src_index >= src->storage->n_bytes / dtype_size) {
      status = status_error(TT_ERR_OUT_OF_RANGE,
                            "source element index is outside storage bounds");
      goto copy_cleanup;
    }

    size_t src_byte_offset;
    size_t dst_byte_offset;
    if (T_checked_mul_size((size_t)src_index, dtype_size, &src_byte_offset) !=
            RNG_OK ||
        T_checked_mul_size((size_t)linear, dtype_size, &dst_byte_offset) !=
            RNG_OK) {
      status =
          status_error(TT_ERR_OVERFLOW, "tensor element byte offset overflow");
      goto copy_cleanup;
    }

    memcpy(host_dst + dst_byte_offset, host_src + src_byte_offset, dtype_size);
  }

  status = status_ok();

copy_cleanup:
  if (status.stype == TT_OK && destination_staging != NULL) {
    status = storage_copy(dst->data, dst->device, destination_staging,
                          host_device, required_bytes);
  }

  free(destination_staging);
  free(source_staging);
  return status;
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

  Ten_Status status = tensor_init_layout(&tmp, input_shape, rank);
  if (status.stype != TT_OK)
    return status;

  if (!T_storage_retain(storage)) {
    release_tensor(&tmp);
    return status_error(TT_ERR_INVALID_STATE, "storage retain failed");
  }

  tmp.storage = storage;
  *dst = tmp;

  return status_ok();
}

Ten_Status T_tensor_is_contiguous(const Tensor *t, bool *out) {
  if (t == NULL || out == NULL)
    return status_error(TT_ERR_INVALID_ARG, "null argument");

  if (!T_validate_tensor_bounds(t))
    return status_error(TT_ERR_INVALID_ARG, "invalid tensor bounds");

  if (t->num_el == 0) {
    *out = true;
    return status_ok();
  }

  i64 expected_stride = 1;

  for (i32 i = t->rank - 1; i >= 0; --i) {
    if (t->shape[i] == 1)
      continue;

    if (t->strides[i] != expected_stride) {
      *out = false;
      return status_ok();
    }

    if (expected_stride > INT64_MAX / t->shape[i])
      return status_error(TT_ERR_OVERFLOW, "stride calculation overflow");

    expected_stride *= t->shape[i];
  }

  *out = true;
  return status_ok();
}

Ten_Status T_create_tensor_by_data(Tensor *t, const void *input_data,
                                   const DataType i_dtype, const i64 *i_shape,
                                   const i32 i_ranks, const Device dev) {
  Ten_Status status = T_validate_new_tensor(t, i_dtype, i_shape, i_ranks, dev);
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

  status = tensor_init_layout(&tmp, i_shape, i_ranks);
  if (status.stype != TT_OK)
    return status;

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

  if (!T_allocate_shape(&tmp, src->shape, src->rank)) {
    release_tensor(&tmp);
    return status_error(TT_ERR_NO_MEMORY, "shape allocation failed");
  }

  if (!T_allocate_stride(&tmp)) {
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

Ten_Status T_tensor_reshape(Tensor *dst, const Tensor *src, const i64 *shape,
                            i32 rank) {
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
    return status_error(TT_ERR_OVERFLOW, "invalid shape");

  if (new_numel != src->num_el)
    return status_error(TT_ERR_INVALID_ARG, "reshape element count mismatch");

  if (src->num_el < 0 || src->offset < 0)
    return status_error(TT_ERR_INVALID_ARG, "invalid source metadata");

  if (!T_is_dtype_valid(src->storage->dtype))
    return status_error(TT_ERR_DTYPE_UNSUPPORTED, "unsupported dtype");

  size_t dtype_size = T_get_dtype_size(src->storage->dtype);

  if (dtype_size == 0)
    return status_error(TT_ERR_DTYPE_UNSUPPORTED, "unsupported dtype");

  if ((uint64_t)new_numel > SIZE_MAX / dtype_size)
    return status_error(TT_ERR_OVERFLOW, "reshape byte size overflow");

  size_t n_bytes = (size_t)new_numel * dtype_size;

  Tensor tmp = {0};

  tmp.rank = rank;
  tmp.num_el = new_numel;
  tmp.offset = 0;

  /* Allocate destination metadata. */
  Ten_Status status = tensor_init_layout(&tmp, shape, rank);
  if (status.stype != TT_OK)
    return status;

  status = T_create_empty_storage(&tmp.storage, src->storage->dtype,
                                  src->storage->device, n_bytes);

  if (status.stype != TT_OK) {
    release_tensor(&tmp);
    return status;
  }

  status = T_copy_tensor_data(tmp.storage, src);

  if (status.stype != TT_OK) {
    release_tensor(&tmp);
    return status;
  }

  *dst = tmp;

  return status_ok();
}

static bool by_value(void *data, const void *scalar, const DataType dtype,
                     const i64 numel) {
  if (numel < 0 || scalar == NULL || (numel > 0 && data == NULL))
    return false;

  size_t dt_size = T_get_dtype_size(dtype);

  /* Complex fill is unsupported (unchanged behaviour). */
  if (dt_size == 0 || T_is_dtype_complex(dtype))
    return false;

  unsigned char *dst = (unsigned char *)data;

  for (i64 i = 0; i < numel; ++i)
    memcpy(dst + (size_t)i * dt_size, scalar, dt_size);

  return true;
}

Ten_Status T_tensor_ones(Tensor *t, const DataType dtype, const i64 *shape,
                         const i32 rank, const Device dev) {
  return tensor_create_initialized(t, dtype, shape, rank, dev, ONES, NULL);
}

Ten_Status T_tensor_zeros(Tensor *t, const DataType dtype, const i64 *shape,
                          const i32 rank, const Device dev) {
  return tensor_create_initialized(t, dtype, shape, rank, dev, ZEROS, NULL);
}

Ten_Status T_tensor_fill_value(Tensor *t, const DataType dtype,
                               const i64 *shape, const i32 rank,
                               const Device dev, const void *scalar) {
  return tensor_create_initialized(t, dtype, shape, rank, dev, FROM_SCALAR,
                                   scalar);
}
