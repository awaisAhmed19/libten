/*
 * Shape, element-count, and contiguous-stride tests for libten.
 * Includes tensor.c temporarily to test internal helpers.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../include/ten/ten_types.h"

size_t T_get_dtype_size(DataType dtype);
#include "../src/core/ten_tensor.c"

size_t T_get_dtype_size(DataType dtype) {
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

static uint32_t rng_state = UINT32_C(0x5eed1234);
static uint32_t next_random(void) {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return rng_state;
}

static int test_shape_validation_boundaries(void) {
  const i64 scalar_shape_dummy = 0;
  const i64 negative[] = {2, -1};
  const i64 overflow[] = {I64_MAX, 2};
  const i64 valid_boundary[] = {I64_MAX, 1};
  const i64 empty[] = {2, 0, 4};

  CHECK(T_is_shape_input_valid(NULL, 0));
  CHECK(T_is_shape_input_valid(&scalar_shape_dummy, 1));
  CHECK(T_is_shape_input_valid(valid_boundary, 2));
  CHECK(T_is_shape_input_valid(empty, 3));
  CHECK(!T_is_shape_input_valid(NULL, 1));
  CHECK(!T_is_shape_input_valid(negative, 2));
  CHECK(!T_is_shape_input_valid(overflow, 2));
  CHECK(!T_is_shape_input_valid(negative, -1));
  CHECK(!T_is_shape_input_valid(negative, TT_MAX_RANK + 1));
  return 0;
}

static int test_known_strides(void) {
  const i64 shape2[] = {2, 3};
  const i64 shape3[] = {2, 3, 4};
  const i64 shape_empty[] = {2, 0, 4};
  i64 strides2[2] = {-1, -1};
  i64 strides3[3] = {-1, -1, -1};
  i64 strides_empty[3] = {-1, -1, -1};

  CHECK(calculate_strides(strides2, shape2, 2));
  CHECK(strides2[0] == 3 && strides2[1] == 1);

  CHECK(calculate_strides(strides3, shape3, 3));
  CHECK(strides3[0] == 12 && strides3[1] == 4 && strides3[2] == 1);

  CHECK(calculate_strides(strides_empty, shape_empty, 3));
  CHECK(strides_empty[0] == 0 && strides_empty[1] == 4 &&
        strides_empty[2] == 1);

  CHECK(calculate_strides(NULL, NULL, 0));
  CHECK(!calculate_strides(NULL, shape2, 2));
  CHECK(!calculate_strides(strides2, NULL, 2));
  return 0;
}

static int test_randomized_shapes_and_strides(void) {
  int trial;

  for (trial = 0; trial < 2000; ++trial) {
    i32 rank = (i32)(next_random() % (TT_MAX_RANK + 1));
    i64 shape[TT_MAX_RANK];
    i64 strides[TT_MAX_RANK];
    i64 expected_numel = 1;
    i64 expected_stride = 1;
    i32 i;

    for (i = 0; i < rank; ++i) {
      shape[i] = (i64)(next_random() % 6); /* includes zero dimensions */
      strides[i] = -1;
      if (shape[i] != 0 && expected_numel > I64_MAX / shape[i]) {
        return 1; /* unreachable with these deliberately small dimensions */
      }
      expected_numel *= shape[i];
    }

    CHECK(T_is_shape_input_valid(rank == 0 ? NULL : shape, rank));

    if (rank == 0) {
      i64 count = 0;
      T_get_numel(0, NULL, &count);
      CHECK(count == 1);
      CHECK(calculate_strides(NULL, NULL, 0));
      continue;
    }

    {
      i64 actual_numel = -1;
      T_get_numel(rank, shape, &actual_numel);
      CHECK(actual_numel == expected_numel);
    }

    CHECK(calculate_strides(strides, shape, rank));
    for (i = rank - 1; i >= 0; --i) {
      CHECK(strides[i] == expected_stride);
      expected_stride *= shape[i];
    }
  }
  return 0;
}

int main(void) {
  if (test_shape_validation_boundaries() || test_known_strides() ||
      test_randomized_shapes_and_strides()) {
    return EXIT_FAILURE;
  }

  printf("PASS: %d checks\n", tests_run);
  return EXIT_SUCCESS;
}
