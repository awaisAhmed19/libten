#include "ten/ten_tensor.h"
#include "ten/ten_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct tt_storage {
  u0 *data; /* may be NULL only when n_bytes == 0 */
  size_t n_bytes;
  Device device;
};

typedef struct Tensor {
  struct tt_storage *storage;
  DataType dtype;
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

  for (int i = 0; i < ranks; ++i) {
    if (input_shape[i] < 0)
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

static bool allocate_shape(Tensor *t, const i64 *shape, i32 ranks) {
  if (t == NULL)
    return false;

  if (ranks < 0 || ranks > TT_MAX_RANK)
    return false;

  t->rank = ranks;
  t->shape = NULL;

  if (!T_is_shape_input_valid(shape, ranks))
    return false;

  // A scalar has rank zero and no shape array.
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

static void free_shape(i64 *shape) { free(shape); }
static void free_strides(i64 *strides) { free(strides); }

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

static i64 get_num_el(const i32 ranks, const i64 *shape) {
  if (ranks == 0)
    return 1;
  i64 num_el = 1;
  for (size_t i = 0; i < ranks; i++) {
    num_el *= shape[i];
  }
  return num_el;
}

static void release_tensor(Tensor *t) {
  if (t == NULL)
    return;

  if (t->storage != NULL) {
    free(t->storage->data);
    free(t->storage);
  }

  free(t->shape);
  free(t->strides);

  *t = (Tensor){0};
}

void T_destroy_tensor(Tensor *t) { release_tensor(t); }

bool T_create_tensor_by_data(Tensor *t, const void *input_data,
                             DataType i_dtype, const i64 *i_shape, i32 i_ranks,
                             Device dev) {
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
  tmp.dtype = i_dtype;
  tmp.rank = i_ranks;
  tmp.offset = 0;
  tmp.num_el = get_num_el(i_ranks, i_shape);

  if (tmp.num_el < 0 || (size_t)tmp.num_el > SIZE_MAX / dt_size) {
    return false;
  }

  size_t nbytes = dt_size * (size_t)tmp.num_el;

  if (nbytes > 0 && input_data == NULL)
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

  tmp.storage = calloc(1, sizeof(*tmp.storage));
  if (tmp.storage == NULL) {
    fprintf(stderr, "ERR: storage allocation failed\n");
    release_tensor(&tmp);
    return false;
  }

  tmp.storage->device = dev;
  tmp.storage->n_bytes = nbytes;

  if (nbytes > 0) {
    tmp.storage->data = malloc(nbytes);
    if (tmp.storage->data == NULL) {
      fprintf(stderr, "ERR: data allocation failed\n");
      release_tensor(&tmp);
      return false;
    }

    memcpy(tmp.storage->data, input_data, nbytes);
  }

  /* Commit only after construction succeeds. */
  *t = tmp;
  return true;
}
