/*
 * Tensor construction, copying, storage sharing, and destruction tests.
 * White-box for now because Tensor/TensorStorage are currently opaque publicly.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/ten/ten_types.h"

size_t get_dtype_size(DataType dtype);
#include "../src/core/ten_tensor.c"

size_t get_dtype_size(DataType dtype) {
  switch (dtype) {
  case I8:
  case U8:
    return 1;
  case I16:
  case U16:
    return 2;
  case I32:
  case U32:
  case F32:
    return 4;
  case I64:
  case U64:
  case F64:
    return 8;
  default:
    return 0;
  }
}

#define CPU_DEVICE ((Device){.devtype = CPU, .id = 0})
static int tests_run;

#define CHECK(expr)                                                            \
  do {                                                                         \
    ++tests_run;                                                               \
    if (!(expr)) {                                                             \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr);          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static int test_constructs_and_copies_data(void) {
  const i64 shape[] = {2, 3};
  float input[] = {1, 2, 3, 4, 5, 6};
  Tensor t = {0};

  CHECK(T_create_tensor_by_data(&t, input, F32, shape, 2, CPU_DEVICE));
  CHECK(t.rank == 2 && t.num_el == 6 && t.offset == 0);
  CHECK(t.shape[0] == 2 && t.shape[1] == 3);
  CHECK(t.strides[0] == 3 && t.strides[1] == 1);
  CHECK(t.storage != NULL && t.storage->dtype == F32);
  CHECK(t.storage->n_bytes == sizeof(input));
  CHECK(memcmp(t.storage->data, input, sizeof(input)) == 0);

  input[0] = 99.0f;
  CHECK(((float *)t.storage->data)[0] == 1.0f);

  T_destroy_tensor(&t);
  CHECK(t.storage == NULL && t.shape == NULL && t.strides == NULL);
  return 0;
}

static int test_scalar_and_empty_tensors(void) {
  const float scalar = 42.0f;
  const i64 empty_shape[] = {2, 0, 4};
  Tensor s = {0};
  Tensor e = {0};

  CHECK(T_create_tensor_by_data(&s, &scalar, F32, NULL, 0, CPU_DEVICE));
  CHECK(s.rank == 0 && s.num_el == 1);
  CHECK(s.shape == NULL && s.strides == NULL);
  CHECK(*(float *)s.storage->data == 42.0f);
  T_destroy_tensor(&s);

  CHECK(T_create_tensor_by_data(&e, NULL, F32, empty_shape, 3, CPU_DEVICE));
  CHECK(e.rank == 3 && e.num_el == 0);
  CHECK(e.storage != NULL && e.storage->n_bytes == 0);
  CHECK(e.shape[0] == 2 && e.shape[1] == 0 && e.shape[2] == 4);
  T_destroy_tensor(&e);
  return 0;
}

static int test_invalid_construction_is_transactional(void) {
  const i64 negative[] = {2, -1};
  const float values[] = {1, 2};
  Tensor t = {0};

  CHECK(!T_create_tensor_by_data(NULL, values, F32, negative, 2, CPU_DEVICE));
  CHECK(!T_create_tensor_by_data(&t, values, F32, negative, 2, CPU_DEVICE));
  CHECK(t.storage == NULL && t.shape == NULL && t.strides == NULL);
  CHECK(
      !T_create_tensor_by_data(&t, NULL, F32, (const i64[]){2}, 1, CPU_DEVICE));
  CHECK(t.storage == NULL && t.shape == NULL && t.strides == NULL);
  return 0;
}

static int test_shared_storage_lifetime(void) {
  const float values[] = {10, 20, 30, 40};
  const i64 shape[] = {2, 2};
  TensorStorage *storage = NULL;
  Tensor view = {0};

  CHECK(T_create_tensor_storage(&storage, values, F32, CPU_DEVICE,
                                sizeof(values)));
  CHECK(storage->ref_count == 1);
  CHECK(T_tensor_share_storage(&view, storage, shape, 2));
  CHECK(view.storage == storage && storage->ref_count == 2);
  CHECK(view.shape[0] == 2 && view.shape[1] == 2);
  CHECK(view.strides[0] == 2 && view.strides[1] == 1);

  T_storage_release(&storage);
  CHECK(storage == NULL);
  CHECK(view.storage != NULL && view.storage->ref_count == 1);
  CHECK(((float *)view.storage->data)[3] == 40.0f);

  T_destroy_tensor(&view);
  CHECK(view.storage == NULL && view.shape == NULL && view.strides == NULL);
  return 0;
}

int main(void) {
  if (test_constructs_and_copies_data() || test_scalar_and_empty_tensors() ||
      test_invalid_construction_is_transactional() ||
      test_shared_storage_lifetime()) {
    return EXIT_FAILURE;
  }

  printf("PASS: %d checks\n", tests_run);
  return EXIT_SUCCESS;
}
