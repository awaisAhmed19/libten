/*
 * Tests for T_tensor_share_storage().
 *
 * Follows the temporary white-box style used by tensor_helpers_test.c and
 * tensor_constructor_test.c: include tensor.c so the tests can inspect
 * TensorStorage while the public API is still being developed.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/ten/ten_status.h"
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

#define CPU_DEVICE ((Device){.devtype = CPU, .id = 0})

static int test_rejects_null_arguments(void) {
  const i64 shape[] = {2};
  const float values[] = {1.0f, 2.0f};
  TensorStorage *storage = NULL;
  Tensor dst = {0};

  CHECK_STATUS(T_tensor_share_storage(NULL, NULL, shape, 1),
               TT_ERR_INVALID_ARG);
  CHECK_STATUS(T_tensor_share_storage(&dst, NULL, shape, 1),
               TT_ERR_INVALID_ARG);

  CHECK_STATUS(T_create_tensor_storage(&storage, values, F32, CPU_DEVICE,
                                       sizeof(values)),
               TT_OK);
  CHECK_STATUS(T_tensor_share_storage(NULL, storage, shape, 1),
               TT_ERR_INVALID_ARG);

  T_storage_release(&storage);
  CHECK(storage == NULL);
  return 0;
}

static int test_shares_storage_and_copies_metadata(void) {
  const float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const i64 shape[] = {2, 3};
  TensorStorage *storage = NULL;
  Tensor dst = {0};

  CHECK_STATUS(T_create_tensor_storage(&storage, values, F32, CPU_DEVICE,
                                       sizeof(values)),
               TT_OK);
  CHECK(storage->ref_count == 1);

  CHECK_STATUS(T_tensor_share_storage(&dst, storage, shape, 2), TT_OK);

  CHECK(dst.storage == storage);
  CHECK(storage->ref_count == 2);
  CHECK(dst.rank == 2);
  CHECK(dst.offset == 0);
  CHECK(dst.num_el == 6);
  CHECK(dst.shape != NULL);
  CHECK(dst.shape[0] == 2 && dst.shape[1] == 3);
  CHECK(dst.strides != NULL);
  CHECK(dst.strides[0] == 3 && dst.strides[1] == 1);
  CHECK(dst.storage->dtype == F32);
  CHECK(dst.storage->n_bytes == sizeof(values));
  CHECK(dst.storage->data == storage->data);

  /* Releasing the creator's reference must not invalidate the tensor. */
  T_storage_release(&storage);
  CHECK(storage == NULL);
  CHECK(dst.storage != NULL);
  CHECK(dst.storage->ref_count == 1);
  CHECK(((float *)dst.storage->data)[5] == 6.0f);

  T_destroy_tensor(&dst);
  CHECK(tensor_is_empty(&dst));
  CHECK(dst.storage == NULL);
  CHECK(dst.shape == NULL);
  CHECK(dst.strides == NULL);
  return 0;
}

static int test_rejects_insufficient_storage_capacity(void) {
  const float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
  const i64 too_large[] = {2, 3};
  TensorStorage *storage = NULL;
  Tensor dst = {0};

  CHECK_STATUS(T_create_tensor_storage(&storage, values, F32, CPU_DEVICE,
                                       sizeof(values)),
               TT_OK);
  CHECK_STATUS(T_tensor_share_storage(&dst, storage, too_large, 2),
               TT_ERR_OUT_OF_RANGE);
  CHECK(dst.storage == NULL);
  CHECK(dst.shape == NULL);
  CHECK(dst.strides == NULL);
  CHECK(storage->ref_count == 1);

  T_storage_release(&storage);
  return 0;
}

static int test_rejects_invalid_shapes_without_retaining(void) {
  const float values[] = {1.0f, 2.0f};
  const i64 negative[] = {2, -1};
  const i64 overflow[] = {I64_MAX, 2};
  const i64 valid_shape[] = {2};
  TensorStorage *storage = NULL;
  Tensor dst = {0};

  CHECK_STATUS(T_create_tensor_storage(&storage, values, F32, CPU_DEVICE,
                                       sizeof(values)),
               TT_OK);
  CHECK_STATUS(T_tensor_share_storage(&dst, storage, negative, 2),
               TT_ERR_INVALID_ARG);
  CHECK_STATUS(T_tensor_share_storage(&dst, storage, overflow, 2),
               TT_ERR_OVERFLOW);
  CHECK_STATUS(T_tensor_share_storage(&dst, storage, NULL, 1),
               TT_ERR_INVALID_ARG);
  CHECK_STATUS(T_tensor_share_storage(&dst, storage, valid_shape, -1),
               TT_ERR_INVALID_ARG);
  CHECK(storage->ref_count == 1);
  CHECK(dst.storage == NULL);

  T_storage_release(&storage);
  return 0;
}

static int test_scalar_tensor(void) {
  const float value = 42.0f;
  TensorStorage *storage = NULL;
  Tensor dst = {0};

  CHECK_STATUS(
      T_create_tensor_storage(&storage, &value, F32, CPU_DEVICE, sizeof(value)),
      TT_OK);
  CHECK_STATUS(T_tensor_share_storage(&dst, storage, NULL, 0), TT_OK);
  CHECK(dst.rank == 0);
  CHECK(dst.num_el == 1);
  CHECK(dst.offset == 0);
  CHECK(dst.shape == NULL);
  CHECK(dst.strides == NULL);
  CHECK(dst.storage == storage);
  CHECK(storage->ref_count == 2);

  T_destroy_tensor(&dst);
  CHECK(tensor_is_empty(&dst));
  CHECK(storage->ref_count == 1);
  T_storage_release(&storage);
  return 0;
}

static int test_empty_tensor(void) {
  const i64 shape[] = {2, 0, 4};
  TensorStorage *storage = NULL;
  Tensor dst = {0};

  CHECK_STATUS(T_create_tensor_storage(&storage, NULL, F32, CPU_DEVICE, 0),
               TT_OK);
  CHECK_STATUS(T_tensor_share_storage(&dst, storage, shape, 3), TT_OK);
  CHECK(dst.rank == 3);
  CHECK(dst.num_el == 0);
  CHECK(dst.shape[0] == 2 && dst.shape[1] == 0 && dst.shape[2] == 4);
  CHECK(dst.strides[0] == 0 && dst.strides[1] == 4 && dst.strides[2] == 1);
  CHECK(dst.storage == storage);
  CHECK(storage->ref_count == 2);

  T_destroy_tensor(&dst);
  CHECK(tensor_is_empty(&dst));
  T_storage_release(&storage);
  return 0;
}

static int test_rejects_nonempty_destination(void) {
  const float values[] = {1.0f, 2.0f};
  const i64 shape[] = {2};
  TensorStorage *storage = NULL;
  Tensor dst = {0};

  CHECK_STATUS(T_create_tensor_storage(&storage, values, F32, CPU_DEVICE,
                                       sizeof(values)),
               TT_OK);
  CHECK_STATUS(T_tensor_share_storage(&dst, storage, shape, 1), TT_OK);
  CHECK_STATUS(T_tensor_share_storage(&dst, storage, shape, 1),
               TT_ERR_INVALID_STATE);
  CHECK(storage->ref_count == 2);

  T_destroy_tensor(&dst);
  T_storage_release(&storage);
  return 0;
}
static int test_rejects_reference_count_overflow(void) {
  const float values[] = {1.0f, 2.0f};
  const i64 shape[] = {2};
  TensorStorage *storage = NULL;
  Tensor dst = {0};

  CHECK_STATUS(T_create_tensor_storage(&storage, values, F32, CPU_DEVICE,
                                       sizeof(values)),
               TT_OK);

  storage->ref_count = UINT32_MAX;

  CHECK_STATUS(T_tensor_share_storage(&dst, storage, shape, 1),
               TT_ERR_INVALID_STATE);

  CHECK(dst.storage == NULL);
  CHECK(dst.shape == NULL);
  CHECK(dst.strides == NULL);
  CHECK(storage->ref_count == UINT32_MAX);

  /* Restore a valid count before releasing the storage. */
  storage->ref_count = 1;
  T_storage_release(&storage);

  return 0;
}

int main(void) {
  if (test_rejects_null_arguments() ||
      test_shares_storage_and_copies_metadata() ||
      test_rejects_insufficient_storage_capacity() ||
      test_rejects_invalid_shapes_without_retaining() || test_scalar_tensor() ||
      test_empty_tensor() || test_rejects_nonempty_destination() ||
      test_rejects_reference_count_overflow()) {
    return EXIT_FAILURE;
  }

  printf("PASS: %d checks\n", tests_run);
  return EXIT_SUCCESS;
}
