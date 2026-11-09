
#include "internal/ten_tensor.h"
#include "ten/ten_tensor.h"
#include "ten/ten_view.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./tensor_checker.h"

static Device cpu_device(void) { return (Device){.devtype = CPU, .id = 0}; }

#define CPU_DEVICE ((Device){.devtype = CPU, .id = 0})
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
static int test_materialize_contiguous_tensor(void) {
  const i64 shape[] = {2, 3};
  const float data[] = {1, 2, 3, 4, 5, 6};

  Tensor src = {0};
  Tensor dst = {0};

  CHECK_STATUS(T_create_tensor_by_data(&src, data, F32, shape, 2, CPU_DEVICE),
               TT_OK);

  CHECK_STATUS(T_tensor_contiguous_materialize(&dst, &src), TT_OK);

  CHECK(dst.storage != src.storage);
  CHECK(dst.rank == src.rank);
  CHECK(dst.num_el == src.num_el);
  CHECK(dst.offset == 0);

  bool contiguous = false;
  CHECK_STATUS(T_tensor_is_contiguous(&dst, &contiguous), TT_OK);
  CHECK(contiguous);

  const float *result = (const float *)dst.storage->data;

  for (i64 i = 0; i < 6; ++i)
    CHECK(result[i] == data[i]);

  T_destroy_tensor(&src);
  T_destroy_tensor(&dst);
  return 0;
}

static int test_materialize_transposed_tensor(void) {
  const i64 shape[] = {2, 3};
  const float data[] = {1, 2, 3, 4, 5, 6};
  const float expected[] = {1, 4, 2, 5, 3, 6};

  Tensor src = {0};
  Tensor view = {0};
  Tensor dst = {0};

  CHECK_STATUS(T_create_tensor_by_data(&src, data, F32, shape, 2, CPU_DEVICE),
               TT_OK);

  CHECK_STATUS(T_tensor_transpose(&view, &src, 0, 1), TT_OK);

  bool contiguous = true;
  CHECK_STATUS(T_tensor_is_contiguous(&view, &contiguous), TT_OK);
  CHECK(!contiguous);

  CHECK_STATUS(T_tensor_contiguous_materialize(&dst, &view), TT_OK);

  CHECK(dst.storage != view.storage);
  CHECK(dst.offset == 0);

  CHECK_STATUS(T_tensor_is_contiguous(&dst, &contiguous), TT_OK);
  CHECK(contiguous);

  const float *result = (const float *)dst.storage->data;

  for (i64 i = 0; i < 6; ++i)
    CHECK(result[i] == expected[i]);

  T_destroy_tensor(&src);
  T_destroy_tensor(&view);
  T_destroy_tensor(&dst);
  return 0;
}

static int test_materialize_nonzero_offset(void) {
  const i64 shape[] = {5};
  const i64 view_shape[] = {3};
  const float data[] = {10, 20, 30, 40, 50};
  const float expected[] = {30, 40, 50};

  Tensor src = {0};
  Tensor view = {0};
  Tensor dst = {0};

  CHECK_STATUS(T_create_tensor_by_data(&src, data, F32, shape, 1, CPU_DEVICE),
               TT_OK);

  CHECK(T_storage_retain(src.storage));

  view.storage = src.storage;
  view.rank = 1;
  view.num_el = 3;
  view.offset = 2;

  CHECK(T_allocate_shape(&view, view_shape, 1));
  CHECK(T_allocate_stride(&view));
  view.strides[0] = 1;

  CHECK_STATUS(T_tensor_contiguous_materialize(&dst, &view), TT_OK);

  CHECK(dst.storage != view.storage);
  CHECK(dst.offset == 0);

  const float *result = (const float *)dst.storage->data;

  for (i64 i = 0; i < 3; ++i)
    CHECK(result[i] == expected[i]);

  T_destroy_tensor(&src);
  T_destroy_tensor(&view);
  T_destroy_tensor(&dst);

  return 0;
}

static int test_materialize_independent_storage(void) {
  const i64 shape[] = {3};
  const float data[] = {1, 2, 3};

  Tensor src = {0};
  Tensor dst = {0};

  CHECK_STATUS(T_create_tensor_by_data(&src, data, F32, shape, 1, CPU_DEVICE),
               TT_OK);

  CHECK_STATUS(T_tensor_contiguous_materialize(&dst, &src), TT_OK);

  CHECK(dst.storage != src.storage);

  float *dst_data = (float *)dst.storage->data;
  CHECK(dst_data != NULL);

  dst_data[0] = 999.0f;

  const float *src_data = (const float *)src.storage->data;
  CHECK(src_data[0] == 1.0f);

  T_destroy_tensor(&src);
  T_destroy_tensor(&dst);
  return 0;
}

static int test_materialize_scalar_and_empty(void) {
  const float scalar = 42.0f;
  const i64 empty_shape[] = {2, 0, 4};

  Tensor src_scalar = {0};
  Tensor dst_scalar = {0};
  Tensor src_empty = {0};
  Tensor dst_empty = {0};

  CHECK_STATUS(
      T_create_tensor_by_data(&src_scalar, &scalar, F32, NULL, 0, CPU_DEVICE),
      TT_OK);

  CHECK_STATUS(T_tensor_contiguous_materialize(&dst_scalar, &src_scalar),
               TT_OK);

  CHECK(dst_scalar.rank == 0);
  CHECK(dst_scalar.num_el == 1);
  CHECK(dst_scalar.offset == 0);
  CHECK(dst_scalar.storage != src_scalar.storage);

  CHECK(*(const float *)dst_scalar.storage->data == 42.0f);

  CHECK_STATUS(T_create_tensor_by_data(&src_empty, NULL, F32, empty_shape, 3,
                                       CPU_DEVICE),
               TT_OK);

  CHECK_STATUS(T_tensor_contiguous_materialize(&dst_empty, &src_empty), TT_OK);

  CHECK(dst_empty.num_el == 0);
  CHECK(dst_empty.rank == 3);
  CHECK(dst_empty.offset == 0);
  CHECK(dst_empty.storage != src_empty.storage);

  T_destroy_tensor(&src_scalar);
  T_destroy_tensor(&dst_scalar);
  T_destroy_tensor(&src_empty);
  T_destroy_tensor(&dst_empty);

  return 0;
}
static int create_slice_test_tensor(Tensor *src) {
  const float data[] = {
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
  };

  const i64 shape[] = {5, 5};

  CHECK_STATUS(T_create_tensor_by_data(src, data, F32, shape, 2, CPU_DEVICE),
               TT_OK);

  return 0;
}
static int check_slice_values(const Tensor *t, const float *expected,
                              i64 count) {
  CHECK(t->num_el == count);

  const float *data = (const float *)t->storage->data;

  i64 linear = 0;

  for (i64 i = 0; i < t->shape[0]; ++i) {
    for (i64 j = 0; j < (t->rank > 1 ? t->shape[1] : 1); ++j) {
      i64 offset =
          t->offset + i * t->strides[0] + (t->rank > 1 ? j * t->strides[1] : 0);

      CHECK(data[offset] == expected[linear++]);
    }
  }

  return 0;
}
static int test_slice_full(void) {
  Tensor src = {0};
  Tensor dst = {0};

  CHECK(create_slice_test_tensor(&src) == 0);

  CHECK_STATUS(T_tensor_slice(&dst, &src, 1, 0, 5, 1), TT_OK);

  CHECK(dst.rank == 2);
  CHECK(dst.num_el == 25);

  CHECK(dst.shape[0] == 5);
  CHECK(dst.shape[1] == 5);

  CHECK(dst.strides[0] == 5);
  CHECK(dst.strides[1] == 1);

  CHECK(dst.offset == 0);
  CHECK(dst.storage == src.storage);

  T_destroy_tensor(&src);
  T_destroy_tensor(&dst);

  return 0;
}
static int test_slice_partial(void) {
  Tensor src = {0};
  Tensor dst = {0};

  const float expected[] = {
      2, 3, 4, 7, 8, 9, 12, 13, 14, 17, 18, 19, 22, 23, 24,
  };

  CHECK(create_slice_test_tensor(&src) == 0);

  CHECK_STATUS(T_tensor_slice(&dst, &src, 1, 1, 4, 1), TT_OK);

  CHECK(dst.shape[0] == 5);
  CHECK(dst.shape[1] == 3);

  CHECK(dst.strides[0] == 5);
  CHECK(dst.strides[1] == 1);

  CHECK(dst.offset == 1);
  CHECK(check_slice_values(&dst, expected, 15) == 0);

  T_destroy_tensor(&src);
  T_destroy_tensor(&dst);

  return 0;
}
static int test_slice_step_two(void) {
  Tensor src = {0};
  Tensor dst = {0};

  const float expected[] = {
      1, 3, 5, 6, 8, 10, 11, 13, 15, 16, 18, 20, 21, 23, 25,
  };

  CHECK(create_slice_test_tensor(&src) == 0);

  CHECK_STATUS(T_tensor_slice(&dst, &src, 1, 0, 5, 2), TT_OK);

  CHECK(dst.shape[0] == 5);
  CHECK(dst.shape[1] == 3);

  CHECK(dst.strides[0] == 5);
  CHECK(dst.strides[1] == 2);

  CHECK(dst.offset == 0);
  CHECK(check_slice_values(&dst, expected, 15) == 0);

  T_destroy_tensor(&src);
  T_destroy_tensor(&dst);

  return 0;
}
static int test_slice_reverse(void) {
  Tensor src = {0};
  Tensor dst = {0};

  const float expected[] = {
      5,  4,  3,  2,  1,  10, 9,  8,  7,  6,  15, 14, 13,
      12, 11, 20, 19, 18, 17, 16, 25, 24, 23, 22, 21,
  };

  CHECK(create_slice_test_tensor(&src) == 0);

  CHECK_STATUS(T_tensor_slice(&dst, &src, 1, 4, -1, -1), TT_OK);

  CHECK(dst.shape[0] == 5);
  CHECK(dst.shape[1] == 5);

  CHECK(dst.strides[0] == 5);
  CHECK(dst.strides[1] == -1);

  CHECK(dst.offset == 4);
  CHECK(check_slice_values(&dst, expected, 25) == 0);

  T_destroy_tensor(&src);
  T_destroy_tensor(&dst);

  return 0;
}
static int test_slice_reverse_step_two(void) {
  Tensor src = {0};
  Tensor dst = {0};

  const float expected[] = {
      5, 3, 1, 10, 8, 6, 15, 13, 11, 20, 18, 16, 25, 23, 21,
  };

  CHECK(create_slice_test_tensor(&src) == 0);

  CHECK_STATUS(T_tensor_slice(&dst, &src, 1, 4, -1, -2), TT_OK);

  CHECK(dst.shape[0] == 5);
  CHECK(dst.shape[1] == 3);

  CHECK(dst.strides[0] == 5);
  CHECK(dst.strides[1] == -2);

  CHECK(dst.offset == 4);
  CHECK(check_slice_values(&dst, expected, 15) == 0);

  T_destroy_tensor(&src);
  T_destroy_tensor(&dst);

  return 0;
}
static int test_slice_empty(void) {
  Tensor src = {0};
  Tensor dst = {0};

  CHECK(create_slice_test_tensor(&src) == 0);

  CHECK_STATUS(T_tensor_slice(&dst, &src, 1, 2, 2, 1), TT_OK);

  CHECK(dst.rank == 2);
  CHECK(dst.shape[0] == 5);
  CHECK(dst.shape[1] == 0);

  CHECK(dst.num_el == 0);
  CHECK(dst.strides[0] == 5);
  CHECK(dst.strides[1] == 1);

  CHECK(dst.storage == src.storage);

  T_destroy_tensor(&src);
  T_destroy_tensor(&dst);

  return 0;
}
static int test_slice_invalid_arguments(void) {
  Tensor src = {0};
  Tensor dst = {0};

  CHECK(create_slice_test_tensor(&src) == 0);
  CHECK_STATUS(T_tensor_slice(&dst, &src, 1, 0, 5, 0), TT_ERR_INVALID_ARG);
  CHECK_STATUS(T_tensor_slice(&dst, &src, 2, 0, 5, 1), TT_ERR_INVALID_ARG);
  CHECK_STATUS(T_tensor_slice(&dst, &src, -1, 0, 5, 1), TT_ERR_INVALID_ARG);
  CHECK_STATUS(T_tensor_slice(&dst, &src, 1, -1, 5, 1), TT_ERR_INVALID_ARG);
  CHECK_STATUS(T_tensor_slice(&dst, &src, 1, 0, 6, 1), TT_ERR_INVALID_ARG);
  CHECK_STATUS(T_tensor_slice(&dst, &src, 1, 0, 5, -1), TT_ERR_INVALID_ARG);
  CHECK_STATUS(T_tensor_slice(&dst, &src, 1, 4, -1, 0), TT_ERR_INVALID_ARG);
  CHECK_STATUS(T_tensor_slice(&dst, &src, 1, 4, 5, -1), TT_ERR_INVALID_ARG);

  T_destroy_tensor(&src);

  return 0;
}
static int test_slice_retains_storage(void) {
  Tensor src = {0};
  Tensor dst = {0};

  const float expected[] = {
      5,  4,  3,  2,  1,  10, 9,  8,  7,  6,  15, 14, 13,
      12, 11, 20, 19, 18, 17, 16, 25, 24, 23, 22, 21,
  };

  CHECK(create_slice_test_tensor(&src) == 0);

  CHECK_STATUS(T_tensor_slice(&dst, &src, 1, 4, -1, -1), TT_OK);

  CHECK(dst.storage == src.storage);
  CHECK(src.storage->ref_count == 2);

  T_destroy_tensor(&src);

  CHECK(check_slice_values(&dst, expected, 25) == 0);

  T_destroy_tensor(&dst);

  return 0;
}
static int test_materialize_reversed_tensor(void) {
  Tensor src = {0};
  Tensor view = {0};
  Tensor dst = {0};

  const float expected[] = {
      5,  4,  3,  2,  1,  10, 9,  8,  7,  6,  15, 14, 13,
      12, 11, 20, 19, 18, 17, 16, 25, 24, 23, 22, 21,
  };

  CHECK(create_slice_test_tensor(&src) == 0);

  CHECK_STATUS(T_tensor_slice(&view, &src, 1, 4, -1, -1), TT_OK);

  CHECK_STATUS(T_tensor_contiguous_materialize(&dst, &view), TT_OK);

  CHECK(dst.storage != view.storage);
  CHECK(dst.offset == 0);
  CHECK(dst.strides[0] == 5);
  CHECK(dst.strides[1] == 1);
  CHECK(dst.num_el == 25);

  CHECK(check_slice_values(&dst, expected, 25) == 0);

  T_destroy_tensor(&src);
  T_destroy_tensor(&view);
  T_destroy_tensor(&dst);

  return 0;
}
static int test_slice_non_contiguous_source(void) {
  Tensor src = {0};
  Tensor first = {0};
  Tensor second = {0};

  const float expected[] = {
      3, 5, 8, 10, 13, 15, 18, 20, 23, 25,
  };

  CHECK(create_slice_test_tensor(&src) == 0);

  // First slice: every second column.
  CHECK_STATUS(T_tensor_slice(&first, &src, 1, 0, 5, 2), TT_OK);

  CHECK(first.shape[0] == 5);
  CHECK(first.shape[1] == 3);
  CHECK(first.strides[0] == 5);
  CHECK(first.strides[1] == 2);

  // Second slice: slice the non-contiguous view.
  CHECK_STATUS(T_tensor_slice(&second, &first, 1, 1, 3, 1), TT_OK);

  CHECK(second.shape[0] == 5);
  CHECK(second.shape[1] == 2);

  CHECK(second.strides[0] == 5);
  CHECK(second.strides[1] == 2);

  CHECK(second.offset == 2);
  CHECK(second.num_el == 10);

  CHECK(second.storage == src.storage);

  // Destroy both parent tensors.
  T_destroy_tensor(&src);
  T_destroy_tensor(&first);

  // The final view must remain valid.
  CHECK(check_slice_values(&second, expected, 10) == 0);

  T_destroy_tensor(&second);

  return 0;
}
static int test_slice_reverse_dim0(void) {
  Tensor src = {0};
  Tensor dst = {0};

  const float expected[] = {
      21, 22, 23, 24, 25, 16, 17, 18, 19, 20, 11, 12, 13,
      14, 15, 6,  7,  8,  9,  10, 1,  2,  3,  4,  5,
  };

  CHECK(create_slice_test_tensor(&src) == 0);

  CHECK_STATUS(T_tensor_slice(&dst, &src, 0, 4, -1, -1), TT_OK);

  CHECK(dst.rank == 2);
  CHECK(dst.shape[0] == 5);
  CHECK(dst.shape[1] == 5);

  CHECK(dst.strides[0] == -5);
  CHECK(dst.strides[1] == 1);

  CHECK(dst.offset == 20);
  CHECK(dst.num_el == 25);

  CHECK(check_slice_values(&dst, expected, 25) == 0);

  T_destroy_tensor(&src);
  T_destroy_tensor(&dst);

  return 0;
}
static int test_slice_empty_tensor(void) {
  Tensor src = {0};
  Tensor dst = {0};

  const i64 shape[] = {2, 0, 4};

  CHECK_STATUS(T_create_tensor_by_data(&src, NULL, F32, shape, 3, CPU_DEVICE),
               TT_OK);

  CHECK(src.num_el == 0);

  CHECK_STATUS(T_tensor_slice(&dst, &src, 1, 0, 0, 1), TT_OK);

  CHECK(dst.rank == 3);

  CHECK(dst.shape[0] == 2);
  CHECK(dst.shape[1] == 0);
  CHECK(dst.shape[2] == 4);

  CHECK(dst.num_el == 0);
  CHECK(dst.offset == src.offset);

  CHECK(dst.storage == src.storage);

  CHECK(dst.storage->n_bytes == 0);

  T_destroy_tensor(&src);

  // The empty view must still retain its storage.
  CHECK(dst.storage != NULL);
  CHECK(dst.num_el == 0);

  T_destroy_tensor(&dst);

  return 0;
}
/*
 * 16. Transpose and permute metadata views
 */
static int test_transpose_view(void) {
  const i64 shape[] = {2, 3};
  const float data[] = {1, 2, 3, 4, 5, 6};
  Tensor src = {0};
  Tensor dst = {0};

  CHECK_STATUS(T_create_tensor_by_data(&src, data, F32, shape, 2, CPU_DEVICE),
               TT_OK);
  CHECK_STATUS(T_tensor_transpose(&dst, &src, 0, 1), TT_OK);

  CHECK(dst.storage == src.storage);
  CHECK(src.storage->ref_count == 2);
  CHECK(dst.rank == 2 && dst.num_el == 6);
  CHECK(dst.shape[0] == 3 && dst.shape[1] == 2);
  CHECK(dst.strides[0] == 1 && dst.strides[1] == 3);
  CHECK(dst.offset == src.offset);

  const float *values = (const float *)dst.storage->data;
  CHECK(values[dst.offset + 0 * dst.strides[0] + 0 * dst.strides[1]] == 1);
  CHECK(values[dst.offset + 1 * dst.strides[0] + 0 * dst.strides[1]] == 2);
  CHECK(values[dst.offset + 2 * dst.strides[0] + 0 * dst.strides[1]] == 3);
  CHECK(values[dst.offset + 0 * dst.strides[0] + 1 * dst.strides[1]] == 4);

  T_destroy_tensor(&dst);
  T_destroy_tensor(&src);
  return 0;
}

static int test_permute_view(void) {
  const i64 shape[] = {2, 3, 4};
  i64 dims[] = {2, 0, 1};
  float data[24];
  for (i32 i = 0; i < 24; ++i)
    data[i] = (float)i;

  Tensor src = {0};
  Tensor dst = {0};
  CHECK_STATUS(T_create_tensor_by_data(&src, data, F32, shape, 3, CPU_DEVICE),
               TT_OK);
  CHECK_STATUS(T_tensor_permute(&dst, &src, dims, 3), TT_OK);

  CHECK(dst.storage == src.storage);
  CHECK(src.storage->ref_count == 2);
  CHECK(dst.rank == 3 && dst.num_el == 24);
  CHECK(dst.shape[0] == 4 && dst.shape[1] == 2 && dst.shape[2] == 3);
  CHECK(dst.strides[0] == 1 && dst.strides[1] == 12 && dst.strides[2] == 4);

  const float *values = (const float *)dst.storage->data;
  CHECK(values[0 * dst.strides[0] + 0 * dst.strides[1] + 0 * dst.strides[2]] ==
        0);
  CHECK(values[1 * dst.strides[0] + 0 * dst.strides[1] + 0 * dst.strides[2]] ==
        1);
  CHECK(values[0 * dst.strides[0] + 1 * dst.strides[1] + 0 * dst.strides[2]] ==
        12);

  T_destroy_tensor(&dst);
  T_destroy_tensor(&src);
  return 0;
}

/*
 * 17. Squeeze and squeeze-dimension views
 */
static int test_squeeze_view(void) {
  const i64 shape[] = {1, 2, 1, 3};
  const float data[] = {1, 2, 3, 4, 5, 6};
  Tensor src = {0};
  Tensor dst = {0};

  CHECK_STATUS(T_create_tensor_by_data(&src, data, F32, shape, 4, CPU_DEVICE),
               TT_OK);
  CHECK_STATUS(T_tensor_squeeze(&dst, &src), TT_OK);

  CHECK(dst.storage == src.storage);
  CHECK(src.storage->ref_count == 2);
  CHECK(dst.rank == 2 && dst.num_el == 6);
  CHECK(dst.shape[0] == 2 && dst.shape[1] == 3);
  CHECK(dst.strides[0] == 3 && dst.strides[1] == 1);

  T_destroy_tensor(&dst);
  T_destroy_tensor(&src);
  return 0;
}

static int test_squeeze_dim_view(void) {
  const i64 shape[] = {1, 2, 1, 3};
  const float data[] = {1, 2, 3, 4, 5, 6};
  Tensor src = {0};
  Tensor dst = {0};

  CHECK_STATUS(T_create_tensor_by_data(&src, data, F32, shape, 4, CPU_DEVICE),
               TT_OK);
  CHECK_STATUS(T_tensor_squeeze_dim(&dst, &src, 2), TT_OK);

  CHECK(dst.storage == src.storage);
  CHECK(dst.rank == 3 && dst.num_el == 6);
  CHECK(dst.shape[0] == 1 && dst.shape[1] == 2 && dst.shape[2] == 3);
  CHECK(dst.strides[0] == 6 && dst.strides[1] == 3 && dst.strides[2] == 1);

  T_destroy_tensor(&dst);
  T_destroy_tensor(&src);
  return 0;
}

static int test_squeeze_scalar_and_empty_views(void) {
  const float scalar_value = 42.0f;
  const i64 empty_shape[] = {1, 0, 1};
  Tensor scalar = {0}, scalar_view = {0};
  Tensor empty = {0}, empty_view = {0};

  CHECK_STATUS(
      T_create_tensor_by_data(&scalar, &scalar_value, F32, NULL, 0, CPU_DEVICE),
      TT_OK);
  CHECK_STATUS(T_tensor_squeeze(&scalar_view, &scalar), TT_OK);
  CHECK(scalar_view.rank == 0 && scalar_view.num_el == 1);
  CHECK(scalar_view.storage == scalar.storage);

  CHECK_STATUS(
      T_create_tensor_by_data(&empty, NULL, F32, empty_shape, 3, CPU_DEVICE),
      TT_OK);
  CHECK_STATUS(T_tensor_squeeze(&empty_view, &empty), TT_OK);
  CHECK(empty_view.rank == 1 && empty_view.num_el == 0);
  CHECK(empty_view.shape[0] == 0);
  CHECK(empty_view.storage == empty.storage);
  CHECK(empty_view.storage->data == NULL);

  T_destroy_tensor(&empty_view);
  T_destroy_tensor(&empty);
  T_destroy_tensor(&scalar_view);
  T_destroy_tensor(&scalar);
  return 0;
}

/*
 * 18. Unsqueeze views, including insertion at both ends and scalar input
 */
static int test_unsqueeze_view_positions(void) {
  const i64 shape[] = {2, 3};
  const float data[] = {1, 2, 3, 4, 5, 6};
  Tensor src = {0};
  Tensor before = {0}, middle = {0}, after = {0};

  CHECK_STATUS(T_create_tensor_by_data(&src, data, F32, shape, 2, CPU_DEVICE),
               TT_OK);

  CHECK_STATUS(T_tensor_unsqueeze(&before, &src, 0), TT_OK);
  CHECK(before.rank == 3 && before.shape[0] == 1 && before.shape[1] == 2 &&
        before.shape[2] == 3);
  CHECK(before.strides[0] == 6 && before.strides[1] == 3 &&
        before.strides[2] == 1);

  CHECK_STATUS(T_tensor_unsqueeze(&middle, &src, 1), TT_OK);
  CHECK(middle.shape[0] == 2 && middle.shape[1] == 1 && middle.shape[2] == 3);
  CHECK(middle.strides[0] == 3 && middle.strides[1] == 3 &&
        middle.strides[2] == 1);

  CHECK_STATUS(T_tensor_unsqueeze(&after, &src, 2), TT_OK);
  CHECK(after.shape[0] == 2 && after.shape[1] == 3 && after.shape[2] == 1);
  CHECK(after.strides[0] == 3 && after.strides[1] == 1 &&
        after.strides[2] == 1);

  CHECK(before.storage == src.storage && middle.storage == src.storage &&
        after.storage == src.storage);
  CHECK(src.storage->ref_count == 4);

  T_destroy_tensor(&before);
  T_destroy_tensor(&middle);
  T_destroy_tensor(&after);
  T_destroy_tensor(&src);
  return 0;
}

static int test_unsqueeze_scalar_and_noncontiguous(void) {
  const float scalar_value = 7.0f;
  const i64 shape[] = {2, 3};
  const float data[] = {1, 2, 3, 4, 5, 6};
  Tensor scalar = {0}, scalar_view = {0};
  Tensor src = {0}, transposed = {0}, expanded = {0};

  CHECK_STATUS(
      T_create_tensor_by_data(&scalar, &scalar_value, F32, NULL, 0, CPU_DEVICE),
      TT_OK);
  CHECK_STATUS(T_tensor_unsqueeze(&scalar_view, &scalar, 0), TT_OK);
  CHECK(scalar_view.rank == 1 && scalar_view.shape[0] == 1);
  CHECK(scalar_view.strides[0] == 1);

  CHECK_STATUS(T_create_tensor_by_data(&src, data, F32, shape, 2, CPU_DEVICE),
               TT_OK);
  CHECK_STATUS(T_tensor_transpose(&transposed, &src, 0, 1), TT_OK);
  CHECK_STATUS(T_tensor_unsqueeze(&expanded, &transposed, 1), TT_OK);
  CHECK(expanded.shape[0] == 3 && expanded.shape[1] == 1 &&
        expanded.shape[2] == 2);
  CHECK(expanded.strides[0] == 1 && expanded.strides[1] == 6 &&
        expanded.strides[2] == 3);
  CHECK(expanded.storage == src.storage);

  T_destroy_tensor(&expanded);
  T_destroy_tensor(&transposed);
  T_destroy_tensor(&src);
  T_destroy_tensor(&scalar_view);
  T_destroy_tensor(&scalar);
  return 0;
}

/*
 * 19. Invalid view arguments must leave destinations untouched
 */
static int test_transform_view_invalid_arguments(void) {
  const i64 shape[] = {2, 3};
  const i64 bad_permutation[] = {0, 0};
  const float data[] = {1, 2, 3, 4, 5, 6};
  Tensor src = {0};
  Tensor dst = {0};

  CHECK_STATUS(T_tensor_transpose(&dst, NULL, 0, 1), TT_ERR_INVALID_ARG);
  CHECK_STATUS(T_tensor_permute(&dst, NULL, NULL, 0), TT_ERR_INVALID_ARG);
  CHECK_STATUS(T_tensor_squeeze(&dst, NULL), TT_ERR_INVALID_ARG);
  CHECK_STATUS(T_tensor_squeeze_dim(&dst, NULL, 0), TT_ERR_INVALID_ARG);
  CHECK_STATUS(T_tensor_unsqueeze(&dst, NULL, 0), TT_ERR_INVALID_ARG);

  CHECK_STATUS(T_create_tensor_by_data(&src, data, F32, shape, 2, CPU_DEVICE),
               TT_OK);

  /* A non-NULL tensor with missing storage must fail without dereferencing it.
   */
  Tensor missing_storage = src;
  missing_storage.storage = NULL;
  CHECK_STATUS(T_tensor_transpose(&dst, &missing_storage, 0, 1),
               TT_ERR_INVALID_STATE);
  CHECK_STATUS(T_tensor_permute(&dst, &missing_storage, shape, 2),
               TT_ERR_INVALID_STATE);
  CHECK_STATUS(T_tensor_squeeze(&dst, &missing_storage), TT_ERR_INVALID_STATE);
  CHECK_STATUS(T_tensor_squeeze_dim(&dst, &missing_storage, 0),
               TT_ERR_INVALID_STATE);
  CHECK_STATUS(T_tensor_unsqueeze(&dst, &missing_storage, 0),
               TT_ERR_INVALID_STATE);
  CHECK_STATUS(T_tensor_transpose(&dst, &src, -1, 1), TT_ERR_INVALID_ARG);
  CHECK_STATUS(T_tensor_transpose(&dst, &src, 0, 2), TT_ERR_INVALID_ARG);
  CHECK_STATUS(T_tensor_permute(&dst, &src, bad_permutation, 2),
               TT_ERR_INVALID_ARG);
  CHECK_STATUS(T_tensor_permute(&dst, &src, shape, 1), TT_ERR_INVALID_ARG);
  CHECK_STATUS(T_tensor_permute(&dst, &src, NULL, 2), TT_ERR_INVALID_ARG);
  CHECK_STATUS(T_tensor_squeeze_dim(&dst, &src, -1), TT_ERR_INVALID_ARG);
  CHECK_STATUS(T_tensor_squeeze_dim(&dst, &src, 0), TT_ERR_INVALID_ARG);
  CHECK_STATUS(T_tensor_unsqueeze(&dst, &src, -1), TT_ERR_INVALID_ARG);
  CHECK_STATUS(T_tensor_unsqueeze(&dst, &src, 3), TT_ERR_INVALID_ARG);

  CHECK(dst.storage == NULL && dst.shape == NULL && dst.strides == NULL);
  CHECK(src.storage->ref_count == 1);

  T_destroy_tensor(&src);
  return 0;
}

static int test_transform_views_accept_empty_tensor(void) {
  const i64 shape[] = {2, 0, 4};
  const i64 permutation[] = {2, 0, 1};
  Tensor src = {0}, transposed = {0}, permuted = {0}, squeezed = {0};
  Tensor expanded = {0};

  CHECK_STATUS(T_create_tensor_by_data(&src, NULL, F32, shape, 3, CPU_DEVICE),
               TT_OK);
  CHECK_STATUS(T_tensor_transpose(&transposed, &src, 0, 2), TT_OK);
  CHECK(transposed.num_el == 0 && transposed.storage->data == NULL);
  CHECK(transposed.shape[0] == 4 && transposed.shape[1] == 0 &&
        transposed.shape[2] == 2);

  CHECK_STATUS(T_tensor_permute(&permuted, &src, permutation, 3), TT_OK);
  CHECK(permuted.num_el == 0 && permuted.storage->data == NULL);
  CHECK(permuted.shape[0] == 4 && permuted.shape[1] == 2 &&
        permuted.shape[2] == 0);

  CHECK_STATUS(T_tensor_squeeze(&squeezed, &src), TT_OK);
  CHECK(squeezed.rank == 3 && squeezed.num_el == 0);

  CHECK_STATUS(T_tensor_unsqueeze(&expanded, &src, 1), TT_OK);
  CHECK(expanded.rank == 4 && expanded.num_el == 0);
  CHECK(expanded.shape[0] == 2 && expanded.shape[1] == 1 &&
        expanded.shape[2] == 0 && expanded.shape[3] == 4);

  T_destroy_tensor(&expanded);
  T_destroy_tensor(&squeezed);
  T_destroy_tensor(&permuted);
  T_destroy_tensor(&transposed);
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
      test_reshape_view_rejects_invalid_source_bounds() ||
      test_materialize_contiguous_tensor() ||
      test_materialize_transposed_tensor() ||
      test_materialize_nonzero_offset() ||
      test_materialize_independent_storage() ||
      test_materialize_scalar_and_empty() || test_slice_full() ||
      test_slice_partial() || test_slice_step_two() || test_slice_reverse() ||
      test_slice_reverse_step_two() || test_slice_empty() ||
      test_slice_invalid_arguments() || test_slice_retains_storage() ||
      test_slice_non_contiguous_source() || test_slice_reverse_dim0() ||
      test_slice_empty_tensor() || test_materialize_reversed_tensor() ||
      test_transpose_view() || test_permute_view() || test_squeeze_view() ||
      test_squeeze_dim_view() || test_squeeze_scalar_and_empty_views() ||
      test_unsqueeze_view_positions() ||
      test_unsqueeze_scalar_and_noncontiguous() ||
      test_transform_view_invalid_arguments() ||
      test_transform_views_accept_empty_tensor()) {
    return EXIT_FAILURE;
  }

  printf("PASS: %d checks\n", tests_run);
  return EXIT_SUCCESS;
}
