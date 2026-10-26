/*
 * Temporary white-box tests for libten's current tensor implementation.
 *
 * This test includes tensor.c so it can exercise static helpers while the
 * public API is still being built. Do not use this pattern permanently:
 * expose a stable public API and test that API from a separate translation
 * unit once the tensor implementation is settled.
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

static int test_rank_validation(void) {
  CHECK(T_is_ranks_valid(0));
  CHECK(T_is_ranks_valid(1));
  CHECK(T_is_ranks_valid(TT_MAX_RANK));
  CHECK(!T_is_ranks_valid(-1));
  CHECK(!T_is_ranks_valid(TT_MAX_RANK + 1));
  return 0;
}

static int test_shape_validation(void) {
  const i64 one[] = {5};
  const i64 two_d[] = {2, 3};
  const i64 three_d[] = {2, 3, 4};
  const i64 empty_middle[] = {2, 0, 4};
  const i64 all_zero[] = {0, 0, 0};
  const i64 negative[] = {-2, 3};
  const i64 product_overflow[] = {I64_MAX, 2};
  const i64 product_boundary[] = {I64_MAX, 1};

  CHECK(T_is_shape_input_valid(NULL, 0)); /* scalar */
  CHECK(T_is_shape_input_valid(one, 1));
  CHECK(T_is_shape_input_valid(two_d, 2));
  CHECK(T_is_shape_input_valid(three_d, 3));
  CHECK(T_is_shape_input_valid(empty_middle, 3));
  CHECK(T_is_shape_input_valid(all_zero, 3));
  CHECK(T_is_shape_input_valid(product_boundary, 2));

  CHECK(!T_is_shape_input_valid(NULL, 1));
  CHECK(!T_is_shape_input_valid(negative, 2));
  CHECK(!T_is_shape_input_valid(product_overflow, 2));
  CHECK(!T_is_shape_input_valid(one, -1));
  CHECK(!T_is_shape_input_valid(one, TT_MAX_RANK + 1));
  return 0;
}

static int test_element_count(void) {
  const i64 one[] = {5};
  const i64 two_d[] = {2, 3};
  const i64 three_d[] = {2, 3, 4};
  const i64 empty[] = {2, 0, 4};
  const i64 singleton[] = {1, 1, 1, 1};
  const i64 large_but_valid[] = {100, 100};
  Tensor t = {0};

  t.rank = 0;
  i64 ret = 0;
  T_get_numel(t.rank, NULL, &ret);
  CHECK(ret == 1);
  T_get_numel(1, one, &ret);
  CHECK(ret == 5);
  T_get_numel(2, two_d, &ret);
  CHECK(ret == 6);
  T_get_numel(3, three_d, &ret);
  CHECK(ret == 24);
  T_get_numel(3, empty, &ret);
  CHECK(ret == 0);
  T_get_numel(4, singleton, &ret);
  CHECK(ret == 1);
  T_get_numel(2, large_but_valid, &ret);
  CHECK(ret == 10000);
  /* Do not call T_get_numel() with an overflowing shape. Validate first. */
  return 0;
}

static int test_contiguous_strides(void) {
  const i64 one[] = {5};
  const i64 two_d[] = {2, 3};
  const i64 three_d[] = {2, 3, 4};
  const i64 empty_middle[] = {2, 0, 4};
  const i64 singleton[] = {1, 1, 1};
  i64 s1[1] = {0};
  i64 s2[2] = {0, 0};
  i64 s3[3] = {0, 0, 0};
  i64 se[3] = {0, 0, 0};
  i64 ss[3] = {0, 0, 0};

  calculate_strides(s1, one, 1);
  CHECK(s1[0] == 1);

  calculate_strides(s2, two_d, 2);
  CHECK(s2[0] == 3 && s2[1] == 1);

  calculate_strides(s3, three_d, 3);
  CHECK(s3[0] == 12 && s3[1] == 4 && s3[2] == 1);

  calculate_strides(se, empty_middle, 3);
  CHECK(se[0] == 0 && se[1] == 4 && se[2] == 1);

  calculate_strides(ss, singleton, 3);
  CHECK(ss[0] == 1 && ss[1] == 1 && ss[2] == 1);

  /* Rank zero must not access shape or strides. */
  calculate_strides(NULL, NULL, 0);
  return 0;
}

int main(void) {
  if (test_rank_validation() || test_shape_validation() ||
      test_element_count() || test_contiguous_strides()) {
    return EXIT_FAILURE;
  }

  printf("PASS: %d checks\n", tests_run);
  return EXIT_SUCCESS;
}
