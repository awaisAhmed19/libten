
/*
 * Status compliance, checked arithmetic and allocation-failure tests
 * for libten.
 *
 * White-box test: includes tensor.c to exercise helpers before the public API
 * is complete. The dtype-size stub isolates this test from ten_dtype.c.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/ten/ten_status.h"
#include "../include/ten/ten_types.h"

/* Fault-injection wrappers. */
static size_t allocation_calls;
static size_t fail_allocation_at;

static void *test_malloc(size_t size) {
  ++allocation_calls;
  if (fail_allocation_at != 0 && allocation_calls == fail_allocation_at)
    return NULL;
  return malloc(size);
}

static void *test_calloc(size_t count, size_t size) {
  ++allocation_calls;
  if (fail_allocation_at != 0 && allocation_calls == fail_allocation_at)
    return NULL;
  return calloc(count, size);
}

static void *test_realloc(void *ptr, size_t size) {
  ++allocation_calls;
  if (fail_allocation_at != 0 && allocation_calls == fail_allocation_at)
    return NULL;
  return realloc(ptr, size);
}

static void test_free(void *ptr) { free(ptr); }

size_t get_dtype_size(DataType dtype);

#define malloc test_malloc
#define calloc test_calloc
#define realloc test_realloc
#define free test_free
#include "../src/core/ten_tensor.c"
#undef free
#undef realloc
#undef calloc
#undef malloc

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

static int tests_run;

#define CHECK(expr)                                                            \
  do {                                                                         \
    ++tests_run;                                                               \
    if (!(expr)) {                                                             \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr);          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHECK_STATUS(expr, expected)                                           \
  do {                                                                         \
    Ten_Status status_check = (expr);                                          \
    ++tests_run;                                                               \
    if (status_check.stype != (expected)) {                                    \
      fprintf(stderr, "FAIL %s:%d: expected status %d, got %d (%s)\n",         \
              __FILE__, __LINE__, (expected), status_check.stype,              \
              status_check.msg ? status_check.msg : "no message");             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static Device cpu_device(void) { return (Device){.devtype = CPU, .id = 0}; }

static int tensor_is_empty(const Tensor *t) {
  return t != NULL && t->storage == NULL && t->shape == NULL &&
         t->strides == NULL && t->rank == 0 && t->offset == 0 && t->num_el == 0;
}

/* --------------------------------------------------------- */
/* Arithmetic tests                                           */
/* --------------------------------------------------------- */

static int test_element_count_overflow(void) {
  const i64 over_i64[] = {I64_MAX, 2};
  const i64 boundary[] = {I64_MAX, 1};
  i64 count = -1;

  CHECK(!T_get_numel(2, over_i64, &count));
  CHECK(T_get_numel(2, boundary, &count));
  CHECK(count == I64_MAX);
  CHECK(!T_get_numel(1, NULL, &count));
  CHECK(!T_get_numel(2, over_i64, NULL));

  return 0;
}

static int test_shape_byte_size_overflow(void) {
  /* Shape product fits in i64, but multiplying by sizeof(float) does not. */
  const i64 huge_shape[] = {I64_MAX / 2, 2};
  Tensor t = {0};
  const float dummy = 1.0f;

  CHECK_STATUS(
      T_create_tensor_by_data(&t, &dummy, F32, huge_shape, 2, cpu_device()),
      TT_ERR_OVERFLOW);

  CHECK(tensor_is_empty(&t));
  return 0;
}

/* --------------------------------------------------------- */
/* Status validation tests                                    */
/* --------------------------------------------------------- */

static int test_success_status(void) {
  const i64 shape[] = {2, 3};
  const float values[] = {1, 2, 3, 4, 5, 6};
  Tensor t = {0};

  CHECK_STATUS(T_create_tensor_by_data(&t, values, F32, shape, 2, cpu_device()),
               TT_OK);

  CHECK(t.storage != NULL);
  CHECK(t.storage->ref_count == 1);
  CHECK(t.num_el == 6);
  CHECK(t.storage->n_bytes == sizeof(values));

  T_destroy_tensor(&t);
  CHECK(tensor_is_empty(&t));

  return 0;
}

static int test_invalid_argument_status(void) {
  const i64 shape[] = {2, 3};
  const float values[] = {1, 2, 3, 4, 5, 6};
  Tensor t = {0};

  CHECK_STATUS(
      T_create_tensor_by_data(NULL, values, F32, shape, 2, cpu_device()),
      TT_ERR_INVALID_ARG);

  CHECK_STATUS(T_create_tensor_by_data(&t, NULL, F32, shape, 2, cpu_device()),
               TT_ERR_INVALID_ARG);

  const i64 invalid_shape[] = {-1, 3};

  CHECK_STATUS(
      T_create_tensor_by_data(&t, values, F32, invalid_shape, 2, cpu_device()),
      TT_ERR_INVALID_ARG);

  CHECK(tensor_is_empty(&t));
  return 0;
}

static int test_invalid_state_status(void) {
  const i64 shape[] = {2};
  const float values[] = {1, 2};
  Tensor t = {0};

  CHECK_STATUS(T_create_tensor_by_data(&t, values, F32, shape, 1, cpu_device()),
               TT_OK);

  /* An initialized tensor cannot be used as a new destination. */
  CHECK_STATUS(T_create_tensor_by_data(&t, values, F32, shape, 1, cpu_device()),
               TT_ERR_INVALID_STATE);

  T_destroy_tensor(&t);
  CHECK(tensor_is_empty(&t));

  return 0;
}

static int test_unsupported_dtype_status(void) {
  const i64 shape[] = {2};
  const float values[] = {1, 2};
  Tensor t = {0};

  CHECK_STATUS(
      T_create_tensor_by_data(&t, values, (DataType)-1, shape, 1, cpu_device()),
      TT_ERR_DTYPE_UNSUPPORTED);

  CHECK(tensor_is_empty(&t));
  return 0;
}

static int test_unsupported_device_status(void) {
  const i64 shape[] = {2};
  const float values[] = {1, 2};
  Tensor t = {0};
  Device unsupported = {.devtype = CUDA, .id = 0};

  CHECK_STATUS(T_create_tensor_by_data(&t, values, F32, shape, 1, unsupported),
               TT_ERR_UNSUPPORTED);

  CHECK(tensor_is_empty(&t));
  return 0;
}

/* --------------------------------------------------------- */
/* Allocation failure tests                                   */
/* --------------------------------------------------------- */

static int test_allocation_failures_leave_empty_tensor(void) {
  const i64 shape[] = {2, 3};
  const float values[] = {1, 2, 3, 4, 5, 6};
  size_t fail_at;
  size_t observed_allocation_count = 0;

  /*
   * Probe allocation count with no injected failure.
   * Construction must be deterministic.
   */
  {
    Tensor probe = {0};

    allocation_calls = 0;
    fail_allocation_at = 0;

    CHECK_STATUS(
        T_create_tensor_by_data(&probe, values, F32, shape, 2, cpu_device()),
        TT_OK);

    observed_allocation_count = allocation_calls;
    T_destroy_tensor(&probe);
  }

  CHECK(observed_allocation_count > 0);

  for (fail_at = 1; fail_at <= observed_allocation_count; ++fail_at) {
    Tensor t = {0};

    allocation_calls = 0;
    fail_allocation_at = fail_at;

    CHECK_STATUS(
        T_create_tensor_by_data(&t, values, F32, shape, 2, cpu_device()),
        TT_ERR_NO_MEMORY);

    CHECK(tensor_is_empty(&t));

    /* Safe even if construction partially initialized the object. */
    T_destroy_tensor(&t);
  }

  fail_allocation_at = 0;
  allocation_calls = 0;

  return 0;
}

/* --------------------------------------------------------- */
/* Other constructor status tests                             */
/* --------------------------------------------------------- */

static int test_initialization_status(void) {
  const i64 shape[] = {2, 2};
  const float scalar = 5.0f;

  Tensor ones = {0};
  Tensor zeros = {0};
  Tensor filled = {0};

  CHECK_STATUS(T_tensor_ones(&ones, F32, shape, 2, cpu_device()), TT_OK);
  CHECK_STATUS(T_tensor_zero(&zeros, F32, shape, 2, cpu_device()), TT_OK);
  CHECK_STATUS(T_tensor_fill(&filled, F32, shape, 2, cpu_device(), &scalar),
               TT_OK);

  const float *ones_data = (const float *)ones.storage->data;
  const float *zeros_data = (const float *)zeros.storage->data;
  const float *filled_data = (const float *)filled.storage->data;

  for (i32 i = 0; i < 4; ++i) {
    CHECK(ones_data[i] == 1.0f);
    CHECK(zeros_data[i] == 0.0f);
    CHECK(filled_data[i] == 5.0f);
  }

  T_destroy_tensor(&ones);
  T_destroy_tensor(&zeros);
  T_destroy_tensor(&filled);

  return 0;
}

static int test_fill_null_scalar_status(void) {
  const i64 shape[] = {2};
  Tensor t = {0};

  CHECK_STATUS(T_tensor_fill(&t, F32, shape, 1, cpu_device(), NULL),
               TT_ERR_INVALID_ARG);

  CHECK(tensor_is_empty(&t));
  return 0;
}

/* --------------------------------------------------------- */

int main(void) {
  if (test_element_count_overflow() || test_shape_byte_size_overflow() ||
      test_success_status() || test_invalid_argument_status() ||
      test_invalid_state_status() || test_unsupported_dtype_status() ||
      test_unsupported_device_status() ||
      test_allocation_failures_leave_empty_tensor() ||
      test_initialization_status() || test_fill_null_scalar_status()) {
    fail_allocation_at = 0;
    return EXIT_FAILURE;
  }

  printf("PASS: %d checks\n", tests_run);
  return EXIT_SUCCESS;
}
