
#include "internal/ten_tensor.h"
#include "ten/ten_tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./tensor_checker.h"

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
 * 10. Contiguous reshape view
 */
static int test_contiguous_reshape_view(void) {
  const i64 src_shape[] = {2, 3};
  const i64 dst_shape[] = {3, 2};
  const float data[] = {1, 2, 3, 4, 5, 6};

  Tensor src = {0};
  Tensor dst = {0};

  CHECK_STATUS(
      T_create_tensor_by_data(&src, data, F32, src_shape, 2, cpu_device()),
      TT_OK);

  u32 original_ref_count = src.storage->ref_count;
  CHECK_STATUS(T_tensor_reshape_view(&dst, &src, dst_shape, 2), TT_OK);

  CHECK(dst.storage == src.storage);
  CHECK(src.storage->ref_count == original_ref_count + 1);
  CHECK(dst.rank == 2);
  CHECK(dst.num_el == 6);
  CHECK(dst.offset == src.offset);
  CHECK(dst.shape[0] == 3 && dst.shape[1] == 2);
  CHECK(dst.strides[0] == 2 && dst.strides[1] == 1);

  /* The logical row-major order is unchanged. */
  const float *values = dst.storage->data;
  for (i32 i = 0; i < 3; ++i) {
    for (i32 j = 0; j < 2; ++j) {
      CHECK(values[dst.offset + i * dst.strides[0] + j * dst.strides[1]] ==
            data[i * 2 + j]);
    }
  }

  T_destroy_tensor(&dst);
  T_destroy_tensor(&src);
  return 0;
}

/*
 * 11. Reshape view preserves a nonzero storage offset
 */
static int test_reshape_view_preserves_offset(void) {
  const i64 src_shape[] = {6};
  const i64 dst_shape[] = {2, 2};
  const float data[] = {10, 20, 30, 40, 50, 60};

  Tensor src = {0};
  Tensor dst = {0};

  CHECK_STATUS(
      T_create_tensor_by_data(&src, data, F32, src_shape, 1, cpu_device()),
      TT_OK);

  /* Represent the contiguous slice [20, 30, 40, 50]. */
  src.offset = 1;
  src.shape[0] = 4;
  src.num_el = 4;

  CHECK_STATUS(T_tensor_reshape_view(&dst, &src, dst_shape, 2), TT_OK);

  CHECK(dst.storage == src.storage);
  CHECK(dst.offset == 1);
  CHECK(dst.shape[0] == 2 && dst.shape[1] == 2);
  CHECK(dst.strides[0] == 2 && dst.strides[1] == 1);

  const float *values = dst.storage->data;
  CHECK(values[dst.offset] == 20.0f);
  CHECK(values[dst.offset + dst.strides[0]] == 40.0f);
  CHECK(values[dst.offset + dst.strides[1]] == 30.0f);
  CHECK(values[dst.offset + dst.strides[0] + dst.strides[1]] == 50.0f);

  T_destroy_tensor(&dst);
  T_destroy_tensor(&src);
  return 0;
}

/*
 * 12. A non-contiguous source cannot be flattened by the current reshape view
 * implementation.
 */
static int test_reshape_view_rejects_non_contiguous_source(void) {
  const i64 src_shape[] = {2, 3};
  const i64 dst_shape[] = {6};
  const float data[] = {1, 2, 3, 4, 5, 6};

  Tensor src = {0};
  Tensor dst = {0};

  CHECK_STATUS(
      T_create_tensor_by_data(&src, data, F32, src_shape, 2, cpu_device()),
      TT_OK);

  /* Metadata-only transpose: logical order becomes 1, 4, 2, 5, 3, 6. */
  src.shape[0] = 3;
  src.shape[1] = 2;
  src.strides[0] = 1;
  src.strides[1] = 3;

  u32 original_ref_count = src.storage->ref_count;
  CHECK_STATUS(T_tensor_reshape_view(&dst, &src, dst_shape, 1),
               TT_ERR_INVALID_ARG);

  CHECK(dst.storage == NULL);
  CHECK(dst.shape == NULL);
  CHECK(dst.strides == NULL);
  CHECK(src.storage->ref_count == original_ref_count);

  T_destroy_tensor(&src);
  return 0;
}

/*
 * 13. Scalar and empty reshape views
 */
static int test_scalar_and_empty_reshape_views(void) {
  const float scalar_value = 42.0f;
  const i64 scalar_dst_shape[] = {1, 1};
  const i64 empty_src_shape[] = {2, 0, 4};
  const i64 empty_dst_shape[] = {0, 8};

  Tensor scalar = {0};
  Tensor scalar_view = {0};
  Tensor empty = {0};
  Tensor empty_view = {0};

  CHECK_STATUS(T_create_tensor_by_data(&scalar, &scalar_value, F32, NULL, 0,
                                       cpu_device()),
               TT_OK);
  CHECK_STATUS(
      T_tensor_reshape_view(&scalar_view, &scalar, scalar_dst_shape, 2), TT_OK);
  CHECK(scalar_view.rank == 2);
  CHECK(scalar_view.num_el == 1);
  CHECK(scalar_view.strides[0] == 1 && scalar_view.strides[1] == 1);
  CHECK(*(const float *)scalar_view.storage->data == scalar_value);

  CHECK_STATUS(T_create_tensor_by_data(&empty, NULL, F32, empty_src_shape, 3,
                                       cpu_device()),
               TT_OK);
  CHECK_STATUS(T_tensor_reshape_view(&empty_view, &empty, empty_dst_shape, 2),
               TT_OK);
  CHECK(empty_view.num_el == 0);
  CHECK(empty_view.shape[0] == 0 && empty_view.shape[1] == 8);
  CHECK(empty_view.strides[0] == 8 && empty_view.strides[1] == 1);
  CHECK(empty_view.storage->data == NULL);

  T_destroy_tensor(&empty_view);
  T_destroy_tensor(&empty);
  T_destroy_tensor(&scalar_view);
  T_destroy_tensor(&scalar);
  return 0;
}

/*
 * 14. Invalid reshape request leaves destination untouched
 */
static int test_reshape_view_rejects_element_count_mismatch(void) {
  const i64 src_shape[] = {2, 3};
  const i64 dst_shape[] = {5};
  const float data[] = {1, 2, 3, 4, 5, 6};

  Tensor src = {0};
  Tensor dst = {0};

  CHECK_STATUS(
      T_create_tensor_by_data(&src, data, F32, src_shape, 2, cpu_device()),
      TT_OK);
  u32 original_ref_count = src.storage->ref_count;

  CHECK_STATUS(T_tensor_reshape_view(&dst, &src, dst_shape, 1),
               TT_ERR_INVALID_ARG);
  CHECK(dst.storage == NULL);
  CHECK(dst.shape == NULL);
  CHECK(dst.strides == NULL);
  CHECK(src.storage->ref_count == original_ref_count);

  T_destroy_tensor(&src);
  return 0;
}

/*
 * 15. Invalid source bounds are rejected
 */
static int test_reshape_view_rejects_invalid_source_bounds(void) {
  const i64 src_shape[] = {2};
  const i64 dst_shape[] = {2};
  const float data[] = {1, 2};

  Tensor src = {0};
  Tensor dst = {0};

  CHECK_STATUS(
      T_create_tensor_by_data(&src, data, F32, src_shape, 1, cpu_device()),
      TT_OK);
  src.offset = 1; /* Only one element remains, but the shape requires two. */

  CHECK_STATUS(T_tensor_reshape_view(&dst, &src, dst_shape, 1),
               TT_ERR_INVALID_ARG);
  CHECK(dst.storage == NULL);
  CHECK(src.storage->ref_count == 1);

  src.offset = 0;
  T_destroy_tensor(&src);
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
      test_view_does_not_retain_storage() || test_contiguous_reshape_view() ||
      test_reshape_view_preserves_offset() ||
      test_reshape_view_rejects_non_contiguous_source() ||
      test_scalar_and_empty_reshape_views() ||
      test_reshape_view_rejects_element_count_mismatch() ||
      test_reshape_view_rejects_invalid_source_bounds()) {
    return EXIT_FAILURE;
  }

  printf("PASS: %d checks\n", tests_run);
  return EXIT_SUCCESS;
}
