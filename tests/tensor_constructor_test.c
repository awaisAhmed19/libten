/*
 * Constructor tests for libten.
 *
 * This white-box test includes tensor.c because Tensor is currently opaque in
 * the public header and the constructor's declaration is not yet public.
 * Compile/run after fixing the constructor's temporary-object ownership and
 * destructor semantics. Against the implementation shown in the discussion,
 * the successful construction case is expected to expose current bugs.
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/core/ten_tensor.c"

static int tests_run;

#define CHECK(expr)                                                            \
  do {                                                                         \
    ++tests_run;                                                               \
    if (!(expr)) {                                                             \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr);          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static int test_rejects_null_output(void) {
  const i64 shape[] = {2};
  const float data[] = {1.0f, 2.0f};

  CHECK(!T_create_tensor_by_data(NULL, data, F32, shape, 1,
                                 (Device){.devtype = CPU, .id = 0}));
  return 0;
}

static int test_rejects_invalid_rank(void) {
  Tensor *t = calloc(1, sizeof(*t));
  const float data[] = {1.0f};

  CHECK(t != NULL);
  CHECK(!T_create_tensor_by_data(t, data, F32, NULL, -1,
                                 (Device){.devtype = CPU, .id = 0}));
  T_destroy_tensor(t);
  return 0;
}

static int test_constructs_and_copies_f32(void) {
  const i64 shape[] = {2, 2};
  float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
  Tensor *t = calloc(1, sizeof(*t));

  CHECK(t != NULL);
  CHECK(T_create_tensor_by_data(t, input, F32, shape, 2,
                                (Device){.devtype = CPU, .id = 0}));

  CHECK(t->rank == 2);
  CHECK(t->num_el == 4);
  CHECK(t->offset == 0);
  CHECK(t->shape[0] == 2 && t->shape[1] == 2);
  CHECK(t->strides[0] == 2 && t->strides[1] == 1);
  CHECK(t->storage != NULL);
  CHECK(t->storage->dtype == F32);
  CHECK(t->storage->n_bytes == 4 * sizeof(float));
  CHECK(t->storage->data != NULL);
  CHECK(memcmp(t->storage->data, input, sizeof(input)) == 0);

  input[0] = 99.0f;
  CHECK(((float *)t->storage->data)[0] == 1.0f);

  T_destroy_tensor(t);
  return 0;
}

static int test_scalar(void) {
  const float scalar_value = 42.0f;
  Tensor *t = calloc(1, sizeof(*t));

  CHECK(t != NULL);
  CHECK(T_create_tensor_by_data(t, &scalar_value, F32, NULL, 0,
                                (Device){.devtype = CPU, .id = 0}));

  CHECK(t->rank == 0);
  CHECK(t->num_el == 1);
  CHECK(t->shape == NULL);
  CHECK(t->strides == NULL);
  CHECK(t->storage != NULL);
  CHECK(t->storage->n_bytes == sizeof(float));
  CHECK(*(float *)t->storage->data == 42.0f);

  T_destroy_tensor(t);
  return 0;
}

static int test_empty_tensor(void) {
  const i64 shape[] = {2, 0, 4};
  Tensor *t = calloc(1, sizeof(*t));

  CHECK(t != NULL);
  CHECK(T_create_tensor_by_data(t, NULL, F32, shape, 3,
                                (Device){.devtype = CPU, .id = 0}));

  CHECK(t->num_el == 0);
  CHECK(t->storage != NULL);
  CHECK(t->storage->dtype == F32);
  CHECK(t->storage->n_bytes == 0);
  CHECK(t->storage->data == NULL);
  CHECK(t->strides[0] == 0 && t->strides[1] == 4 && t->strides[2] == 1);

  T_destroy_tensor(t);
  return 0;
}

int main(void) {
  if (test_rejects_null_output() || test_rejects_invalid_rank() ||
      test_constructs_and_copies_f32() || test_scalar() ||
      test_empty_tensor()) {
    return EXIT_FAILURE;
  }

  printf("PASS: %d checks\n", tests_run);
  return EXIT_SUCCESS;
}
