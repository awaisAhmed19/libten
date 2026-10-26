/*
 * Checked arithmetic and allocation-failure tests for libten.
 *
 * White-box test: includes tensor.c to exercise helpers before the public API
 * is complete. The dtype-size stub isolates this test from ten_dtype.c.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/ten/ten_types.h"

/* Fault-injection wrappers. Definitions precede the macros so their calls to
 * libc malloc/calloc/realloc/free are not redirected recursively.
 */
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
  const bool ok = T_create_tensor_by_data(&t, &dummy, F32, huge_shape, 2,
                                          (Device){.devtype = CPU, .id = 0});
  CHECK(!ok);
  CHECK(t.storage == NULL && t.shape == NULL && t.strides == NULL);
  return 0;
}

static int tensor_is_empty(const Tensor *t) {
  return t->storage == NULL && t->shape == NULL && t->strides == NULL &&
         t->rank == 0 && t->offset == 0 && t->num_el == 0;
}

static int test_allocation_failures_leave_empty_tensor(void) {
  const i64 shape[] = {2, 3};
  const float values[] = {1, 2, 3, 4, 5, 6};
  size_t fail_at;
  size_t observed_allocation_count = 0;

  /*
   * Probe the allocation count with no injected failure, then fail each
   * allocation in turn. This assumes construction is deterministic.
   */
  {
    Tensor probe = {0};
    allocation_calls = 0;
    fail_allocation_at = 0;
    CHECK(T_create_tensor_by_data(&probe, values, F32, shape, 2,
                                  (Device){.devtype = CPU, .id = 0}));
    observed_allocation_count = allocation_calls;
    T_destroy_tensor(&probe);
  }

  CHECK(observed_allocation_count > 0);
  for (fail_at = 1; fail_at <= observed_allocation_count; ++fail_at) {
    Tensor t = {0};
    allocation_calls = 0;
    fail_allocation_at = fail_at;

    CHECK(!T_create_tensor_by_data(&t, values, F32, shape, 2,
                                   (Device){.devtype = CPU, .id = 0}));
    CHECK(tensor_is_empty(&t));

    /* Safe even if construction partially initialized the object. */
    T_destroy_tensor(&t);
  }

  fail_allocation_at = 0;
  allocation_calls = 0;
  return 0;
}

int main(void) {
  if (test_element_count_overflow() || test_shape_byte_size_overflow() ||
      test_allocation_failures_leave_empty_tensor()) {
    fail_allocation_at = 0;
    return EXIT_FAILURE;
  }

  printf("PASS: %d checks\n", tests_run);
  return EXIT_SUCCESS;
}
