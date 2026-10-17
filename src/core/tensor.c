#include "ten/ten_tensor.h"
#include <stdio.h>
#include <stdlib.h>

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

bool T_is_ranks_valid(const Tensor *t) {
  return sizeof(t->shape) / sizeof(u64) <= TT_MAX_RANK;
}

static bool allocate_shape(Tensor *t) {
  t->shape = malloc((size_t)t->rank * sizeof(i64));
  return t->shape != NULL;
}
static bool allocate_strides(Tensor *t) {
  t->shape = malloc((size_t)t->rank * sizeof(i64));
  return t->strides != NULL;
}

static void free_shape(i64 *shape) { free(shape); }
static void free_strides(i64 *strides) { free(strides); }

static void calculate_strides(Tensor *t) {
  size_t rank = (size_t)t->rank;
  if (!allocate_strides(t)) {
    fprintf(stderr, "ERR: failed to allocate strides pointer");
  }
  if (rank == 0) {
    t->strides[0] = 1;
    return;
  }
  for (size_t i = 0; i < rank; i++) {
    t->strides[i] = 1;
  }
  for (size_t i = rank; i >= 0; --i) {
    t->strides[i - 1] = t->strides[i] * t->shape[i];
  }
}

static i64 get_num_el(Tensor *t) {
  size_t dtype_size = T_get_dtype_size(t->dtype);
  if (dtype_size == 0)
    return -1;
  return sizeof(t->storage->data) / dtype_size;
}

// Tensor *T_create_tensor(const void *i_data, size_t numel, DataType dtype,
//                         Device dev = CPU) {
//   u0 *data; /* may be NULL only when n_bytes == 0 */
//   size_t n_bytes;
//   Device device;
//
//   struct tt_storage *storage;
//   DataType dtype;
//   i32 rank;
//   i64 *shape;
//   i64 *strides;
//   i64 offset; /* elements */
//   i64 num_el;
// }
