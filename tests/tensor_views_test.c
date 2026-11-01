
#include "internal/ten_tensor.h"
#include "ten/ten_tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_run = 0;

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
    Ten_Status status = (expr);                                                \
    ++tests_run;                                                               \
    if (status.stype != (expected)) {                                          \
      fprintf(stderr, "FAIL %s:%d: expected %d, got %d (%s)\n", __FILE__,      \
              __LINE__, (expected), status.stype,                              \
              status.msg ? status.msg : "no message");                         \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static Device cpu_device(void) { return (Device){.devtype = CPU, .id = 0}; }

/*
 * 1. Normal contiguous tensor
 *  */

static int test_contiguous_tensor_view(void) {
  const i64 shape[] = {2, 3};
  const float data[] = {1, 2, 3, 4, 5, 6};

  Tensor t = {0};
  TensorView tv = {0};

  CHECK_STATUS(T_create_tensor_by_data(&t, data, F32, shape, 2, cpu_device()),
               TT_OK);

  CHECK_STATUS(T_create_tensor_view(&tv, &t), TT_OK);

  CHECK(tv.data == t.storage->data);
  CHECK(tv.rank == 2);
  CHECK(tv.dtype == F32);

  CHECK(tv.shape[0] == 2);
  CHECK(tv.shape[1] == 3);

  CHECK(tv.strides[0] == 3);
  CHECK(tv.strides[1] == 1);

  CHECK(memcmp(tv.data, data, sizeof(data)) == 0);

  T_destroy_tensor(&t);
  return 0;
}

/*
 * 2. Nonzero offset
 *  */

static int test_nonzero_offset(void) {
  const i64 shape[] = {4};
  const float data[] = {10, 20, 30, 40};

  Tensor t = {0};
  TensorView tv = {0};

  CHECK_STATUS(T_create_tensor_by_data(&t, data, F32, shape, 1, cpu_device()),
               TT_OK);

  /* Simulate a slice beginning at element 1. */
  t.offset = 1;
  t.shape[0] = 3;
  t.num_el = 3;

  CHECK_STATUS(T_create_tensor_view(&tv, &t), TT_OK);

  CHECK(tv.data == (unsigned char *)t.storage->data + sizeof(float));
  CHECK(((float *)tv.data)[0] == 20.0f);
  CHECK(((float *)tv.data)[1] == 30.0f);
  CHECK(((float *)tv.data)[2] == 40.0f);

  T_destroy_tensor(&t);
  return 0;
}

/*
 * 3. Non-contiguous tensor
 *  */

static int test_non_contiguous_tensor_view(void) {
  const i64 shape[] = {2, 3};
  const float data[] = {1, 2, 3, 4, 5, 6};

  Tensor t = {0};
  TensorView tv = {0};

  CHECK_STATUS(T_create_tensor_by_data(&t, data, F32, shape, 2, cpu_device()),
               TT_OK);

  /*
   * Simulate a transpose.
   *
   * Original:
   * shape   = [2, 3]
   * strides = [3, 1]
   *
   * Transposed:
   * shape   = [3, 2]
   * strides = [1, 3]
   */
  t.shape[0] = 3;
  t.shape[1] = 2;

  t.strides[0] = 1;
  t.strides[1] = 3;

  CHECK_STATUS(T_create_tensor_view(&tv, &t), TT_OK);

  CHECK(tv.shape[0] == 3);
  CHECK(tv.shape[1] == 2);

  CHECK(tv.strides[0] == 1);
  CHECK(tv.strides[1] == 3);

  /* The view must not materialize or rearrange storage. */
  CHECK(tv.data == t.storage->data);

  /* Verify logical indexing using the strides. */
  float *values = tv.data;

  CHECK(values[0 * tv.strides[0] + 0 * tv.strides[1]] == 1);
  CHECK(values[1 * tv.strides[0] + 0 * tv.strides[1]] == 2);
  CHECK(values[0 * tv.strides[0] + 1 * tv.strides[1]] == 4);

  T_destroy_tensor(&t);
  return 0;
}

/*
 * 4. Scalar tensor
 *  */

static int test_scalar_view(void) {
  const float value = 42.0f;

  Tensor t = {0};
  TensorView tv = {0};

  CHECK_STATUS(T_create_tensor_by_data(&t, &value, F32, NULL, 0, cpu_device()),
               TT_OK);

  CHECK(t.rank == 0);
  CHECK(t.num_el == 1);

  CHECK_STATUS(T_create_tensor_view(&tv, &t), TT_OK);

  CHECK(tv.rank == 0);
  CHECK(tv.dtype == F32);
  CHECK(tv.data == t.storage->data);
  CHECK(*(float *)tv.data == 42.0f);

  T_destroy_tensor(&t);
  return 0;
}

/*
 * 5. Empty tensor
 *  */

static int test_empty_tensor_view(void) {
  const i64 shape[] = {2, 0, 4};

  Tensor t = {0};
  TensorView tv = {0};

  CHECK_STATUS(T_create_tensor_by_data(&t, NULL, F32, shape, 3, cpu_device()),
               TT_OK);

  CHECK(t.num_el == 0);
  CHECK(t.storage->data == NULL);

  CHECK_STATUS(T_create_tensor_view(&tv, &t), TT_OK);

  CHECK(tv.rank == 3);
  CHECK(tv.data == NULL);

  CHECK(tv.shape[0] == 2);
  CHECK(tv.shape[1] == 0);
  CHECK(tv.shape[2] == 4);

  T_destroy_tensor(&t);
  return 0;
}

/*
 * 6. Invalid arguments
 *  */

static int test_invalid_arguments(void) {
  const i64 shape[] = {2};
  const float data[] = {1, 2};

  Tensor t = {0};
  TensorView tv = {0};

  CHECK_STATUS(T_create_tensor_view(NULL, &t), TT_ERR_INVALID_ARG);
  CHECK_STATUS(T_create_tensor_view(&tv, NULL), TT_ERR_INVALID_ARG);

  CHECK_STATUS(T_create_tensor_by_data(&t, data, F32, shape, 1, cpu_device()),
               TT_OK);

  /* Destination must be empty. */
  tv.rank = 1;

  CHECK_STATUS(T_create_tensor_view(&tv, &t), TT_ERR_INVALID_STATE);

  T_destroy_tensor(&t);
  return 0;
}

/*
 * 7. Out-of-bounds offset
 *  */

static int test_out_of_bounds_offset(void) {
  const i64 shape[] = {2};
  const float data[] = {1, 2};

  Tensor t = {0};
  TensorView tv = {0};

  CHECK_STATUS(T_create_tensor_by_data(&t, data, F32, shape, 1, cpu_device()),
               TT_OK);

  /* Storage contains two elements. Offset 2 is invalid for a nonempty view. */
  t.offset = 2;

  CHECK_STATUS(T_create_tensor_view(&tv, &t), TT_ERR_INVALID_ARG);

  /* The destination must remain untouched after failure. */
  CHECK(tv.data == NULL);
  CHECK(tv.rank == 0);

  T_destroy_tensor(&t);
  return 0;
}

/*
 * 8. NULL storage data
 *  */

static int test_null_storage_data(void) {
  const i64 shape[] = {2};
  const float data[] = {1, 2};

  Tensor t = {0};
  TensorView tv = {0};

  CHECK_STATUS(T_create_tensor_by_data(&t, data, F32, shape, 1, cpu_device()),
               TT_OK);

  void *original_data = t.storage->data;

  t.storage->data = NULL;

  CHECK_STATUS(T_create_tensor_view(&tv, &t), TT_ERR_INVALID_ARG);

  CHECK(tv.data == NULL);
  CHECK(tv.rank == 0);

  /* Restore before destruction. */
  t.storage->data = original_data;

  T_destroy_tensor(&t);
  return 0;
}

/*
 * 9. Storage ownership
 *  */

static int test_view_does_not_retain_storage(void) {
  const i64 shape[] = {2};
  const float data[] = {1, 2};

  Tensor t = {0};
  TensorView tv = {0};

  CHECK_STATUS(T_create_tensor_by_data(&t, data, F32, shape, 1, cpu_device()),
               TT_OK);

  u32 original_ref_count = t.storage->ref_count;

  CHECK_STATUS(T_create_tensor_view(&tv, &t), TT_OK);

  CHECK(t.storage->ref_count == original_ref_count);

  T_destroy_tensor(&t);

  /* tv is now invalid and must not be accessed. */
  return 0;
}

/*
 * Main
 *  */

int main(void) {
  if (test_contiguous_tensor_view() || test_nonzero_offset() ||
      test_non_contiguous_tensor_view() || test_scalar_view() ||
      test_empty_tensor_view() || test_invalid_arguments() ||
      test_out_of_bounds_offset() || test_null_storage_data() ||
      test_view_does_not_retain_storage()) {
    return EXIT_FAILURE;
  }

  printf("PASS: %d checks\n", tests_run);
  return EXIT_SUCCESS;
}
