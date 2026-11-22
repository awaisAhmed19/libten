/*
 * Contiguous materialization tests for libten.
 *
 * White-box: includes ten_tensor.c and ten_view.c. T_get_backend is renamed
 * for those two files so CPU backend allocation/copy can be forced to fail
 * and live allocations can be counted. CUDA still uses the real backend.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "backend/cpu/ten_memory.h"
#include "backend/cpu/ten_ops.h"
#ifdef TEN_TEST_HAS_CUDA
#include "backend/cuda/ten_memory.h"
#endif
#include "internal/ten_backend.h"
#include "ten/ten_status.h"
#include "ten/ten_tensor.h"

#ifdef TEN_TEST_HAS_CUDA
#include <cuda_runtime.h>
#endif

#include "cuda_test_common.h"

const Ten_Backend *test_get_backend(Device device);

#define T_get_backend test_get_backend
#include "../src/core/ten_tensor.c"
#include "../src/core/ten_view.c"
#undef T_get_backend

#include "./tensor_checker.h"

/* Fault-injecting CPU backend. */
static int fail_alloc;
static int fail_copy;
static long live_allocs;
static long copy_calls;

static Ten_Status test_allocate(void **ptr, size_t n_bytes) {
  if (fail_alloc && n_bytes > 0) {
    if (ptr != NULL)
      *ptr = NULL;
    return status_error(TT_ERR_NO_MEMORY, "injected allocation failure");
  }

  Ten_Status status = cpu_allocate(ptr, n_bytes);

  if (status.stype == TT_OK && ptr != NULL && *ptr != NULL)
    ++live_allocs;

  return status;
}

static Ten_Status test_deallocate(void **ptr) {
  if (ptr != NULL && *ptr != NULL)
    --live_allocs;

  return cpu_deallocate(ptr);
}

static Ten_Status test_copy(void *dst, const void *src, size_t n_bytes,
                            Ten_CopyKind kind) {
  ++copy_calls;

  if (fail_copy)
    return status_error(TT_ERR_INTERNAL, "injected copy failure");

  return cpu_copy(dst, src, n_bytes, kind);
}

static const Ten_Backend test_cpu_backend = {
    .memory =
        {
            .allocate = test_allocate,
            .deallocate = test_deallocate,
            .copy = test_copy,
        },
    .ops =
        {
            .fill = cpu_storage_fill,
        },
};

const Ten_Backend *test_get_backend(Device device) {
  if (device.devtype == CPU)
    return &test_cpu_backend;

  return T_get_backend(device);
}

static void reset_injection(void) {
  fail_alloc = 0;
  fail_copy = 0;
  live_allocs = 0;
  copy_calls = 0;
}

static Device cpu_device(void) { return (Device){.devtype = CPU, .id = 0}; }

#ifdef TEN_TEST_CUDA_ONLY
static Device cuda_device(void) { return (Device){.devtype = CUDA, .id = 0}; }
#endif

static int tensor_is_empty(const Tensor *tensor) {
  return tensor != NULL && tensor->storage == NULL && tensor->shape == NULL &&
         tensor->strides == NULL && tensor->rank == 0 && tensor->offset == 0 &&
         tensor->num_el == 0;
}

#ifndef TEN_TEST_CUDA_ONLY
static int check_no_live_allocations(void) {
  if (live_allocs == 0)
    return 0;

  fprintf(stderr, "FAIL: expected zero live allocations, got %ld\n",
          live_allocs);
  return 1;
}
#endif

/* Shared success-case runner compiled into the CPU and CUDA test binaries. */
typedef enum { CASE_CONTIGUOUS, CASE_TRANSPOSED, CASE_SLICED } CaseKind;

static int run_case(Device dev, CaseKind kind) {
  static const float expected_contiguous[] = {1, 2, 3, 4, 5, 6};
  static const float expected_transposed[] = {1, 4, 2, 5, 3, 6};
  static const float expected_sliced[] = {0, 2, 4, 6, 8, 10};

  float data[12] = {0};
  i64 shape[2] = {0};
  i64 out_shape[2] = {0};
  i64 out_strides[2] = {0};
  const float *expected = NULL;
  i64 count = 0;

  Tensor src = {0};
  Tensor view = {0};
  Tensor dst = {0};
  const Tensor *source = &src;
  int result = 0;
  void *device_input = NULL;
  const void *tensor_input = data;
  const size_t input_bytes = (kind == CASE_SLICED ? 12u : 6u) * sizeof(float);

  if (kind == CASE_SLICED) {
    for (i32 i = 0; i < 12; ++i)
      data[i] = (float)i;

    shape[0] = 3;
    shape[1] = 4;
    out_shape[0] = 3;
    out_shape[1] = 2;
    out_strides[0] = 2;
    out_strides[1] = 1;
    expected = expected_sliced;
    count = 6;
  } else {
    for (i32 i = 0; i < 6; ++i)
      data[i] = (float)(i + 1);

    shape[0] = 2;
    shape[1] = 3;
    count = 6;

    if (kind == CASE_CONTIGUOUS) {
      out_shape[0] = 2;
      out_shape[1] = 3;
      out_strides[0] = 3;
      out_strides[1] = 1;
      expected = expected_contiguous;
    } else {
      out_shape[0] = 3;
      out_shape[1] = 2;
      out_strides[0] = 2;
      out_strides[1] = 1;
      expected = expected_transposed;
    }
  }

  /*
   * T_create_tensor() expects input_data to reside on the device supplied in
   * `dev`. Stage the host fixture explicitly for CUDA instead of passing a
   * host pointer as though it were device memory.
   */
  if (dev.devtype == CUDA) {
#ifdef TEN_TEST_HAS_CUDA
    CUDA_TEST_STATUS(cuda_allocate(&device_input, input_bytes), TT_OK);
    CUDA_TEST_CHECK(device_input != NULL);
    CUDA_TEST_STATUS(
        cuda_copy(device_input, data, input_bytes, TEN_COPY_HOST_TO_DEVICE),
        TT_OK);
    tensor_input = device_input;
#else
    fprintf(stderr, "FAIL: CUDA case compiled without CUDA support\n");
    result = 1;
    goto cleanup;
#endif
  }

  CUDA_TEST_STATUS(T_create_tensor(&src, tensor_input, F32, shape, 2, dev),
                   TT_OK);

  /* The tensor constructor owns its copy; the staging buffer is temporary. */
#ifdef TEN_TEST_HAS_CUDA
  if (device_input != NULL) {
    Ten_Status free_status = cuda_deallocate(&device_input);
    if (free_status.stype != TT_OK) {
      fprintf(stderr, "FAIL: could not release CUDA input staging buffer: %s\n",
              free_status.msg ? free_status.msg : "no message");
      result = 1;
      goto cleanup;
    }
  }
#endif

  if (kind == CASE_TRANSPOSED) {
    CUDA_TEST_STATUS(T_tensor_transpose(&view, &src, 0, 1), TT_OK);
    source = &view;
  } else if (kind == CASE_SLICED) {
    CUDA_TEST_STATUS(T_tensor_slice(&view, &src, 1, 0, 4, 2), TT_OK);
    source = &view;
  }

  if (kind != CASE_CONTIGUOUS) {
    bool contiguous = true;
    CUDA_TEST_STATUS(T_tensor_is_contiguous(source, &contiguous), TT_OK);
    CUDA_TEST_CHECK(!contiguous);
  }

  CUDA_TEST_STATUS(T_tensor_contiguous_materialize(&dst, source), TT_OK);

  CUDA_TEST_CHECK(dst.storage != NULL && dst.storage != source->storage);
  CUDA_TEST_CHECK(dst.storage->data != src.storage->data);
  CUDA_TEST_CHECK(dst.rank == 2);
  CUDA_TEST_CHECK(dst.num_el == count);
  CUDA_TEST_CHECK(dst.offset == 0);
  CUDA_TEST_CHECK(dst.shape[0] == out_shape[0] && dst.shape[1] == out_shape[1]);
  CUDA_TEST_CHECK(dst.strides[0] == out_strides[0] &&
                  dst.strides[1] == out_strides[1]);

  bool contiguous = false;
  CUDA_TEST_STATUS(T_tensor_is_contiguous(&dst, &contiguous), TT_OK);
  CUDA_TEST_CHECK(contiguous);

  /* The output must remain on the source device. */
  Device out_device;
  CUDA_TEST_CHECK(T_get_device(&dst, &out_device));
  CUDA_TEST_CHECK(out_device.devtype == dev.devtype && out_device.id == dev.id);

  /* Read back through the backend; never dereference device memory. */
  void *ptr = NULL;
  Device ptr_device;
  float host[12] = {0};

  CUDA_TEST_STATUS(T_device_data(&dst, &ptr, &ptr_device), TT_OK);
  CUDA_TEST_CHECK(ptr != NULL);
  CUDA_TEST_CHECK(ptr_device.devtype == dev.devtype && ptr_device.id == dev.id);
  CUDA_TEST_STATUS(storage_copy(host, cpu_device(), ptr, ptr_device,
                                (size_t)count * sizeof(float)),
                   TT_OK);
  CUDA_TEST_CHECK(memcmp(host, expected, (size_t)count * sizeof(float)) == 0);

  CUDA_TEST_CHECK(!tensor_is_empty(&dst));

cleanup:
#ifdef TEN_TEST_HAS_CUDA
  if (device_input != NULL)
    (void)cuda_deallocate(&device_input);
#endif
  T_destroy_tensor(&dst);
  T_destroy_tensor(&view);
  T_destroy_tensor(&src);

  if (!tensor_is_empty(&dst)) {
    fprintf(stderr, "FAIL: destination was not empty after cleanup\n");
    result = 1;
  }

  return result;
}

#ifndef TEN_TEST_CUDA_ONLY
static int test_cpu_contiguous(void) {
  const i64 shape[] = {2, 3};
  const float data[] = {1, 2, 3, 4, 5, 6};

  Tensor src = {0};
  Tensor dst = {0};
  int result = 0;

  reset_injection();

  CUDA_TEST_CHECK(run_case(cpu_device(), CASE_CONTIGUOUS) == 0);

  /* Mutating the output must not change the source. */
  CUDA_TEST_STATUS(T_create_tensor(&src, data, F32, shape, 2, cpu_device()),
                   TT_OK);
  CUDA_TEST_STATUS(T_tensor_contiguous_materialize(&dst, &src), TT_OK);

  ((float *)dst.storage->data)[0] = 99.0f;
  CUDA_TEST_CHECK(((float *)src.storage->data)[0] == 1.0f);

cleanup:
  T_destroy_tensor(&dst);
  T_destroy_tensor(&src);

  if (check_no_live_allocations() != 0)
    result = 1;

  return result;
}

static int test_cpu_transposed(void) {
  int result = 0;

  reset_injection();
  CUDA_TEST_CHECK(run_case(cpu_device(), CASE_TRANSPOSED) == 0);

cleanup:
  if (check_no_live_allocations() != 0)
    result = 1;

  return result;
}

static int test_cpu_sliced(void) {
  int result = 0;

  reset_injection();
  CUDA_TEST_CHECK(run_case(cpu_device(), CASE_SLICED) == 0);

cleanup:
  if (check_no_live_allocations() != 0)
    result = 1;

  return result;
}

#endif /* !TEN_TEST_CUDA_ONLY */

#ifdef TEN_TEST_CUDA_ONLY
static int run_cuda_case(CaseKind kind) {
  reset_injection();
  return run_case(cuda_device(), kind);
}

static int test_cuda_contiguous(void) { return run_cuda_case(CASE_CONTIGUOUS); }

static int test_cuda_transposed(void) { return run_cuda_case(CASE_TRANSPOSED); }

static int test_cuda_sliced(void) { return run_cuda_case(CASE_SLICED); }
#endif

#ifndef TEN_TEST_CUDA_ONLY
static int test_empty_tensor(void) {
  const i64 shape[] = {2, 0, 3};
  Tensor src = {0};
  Tensor dst = {0};
  int result = 0;

  reset_injection();

  CUDA_TEST_STATUS(T_create_tensor(&src, NULL, F32, shape, 3, cpu_device()),
                   TT_OK);

  const long copies_before = copy_calls;

  CUDA_TEST_STATUS(T_tensor_contiguous_materialize(&dst, &src), TT_OK);

  CUDA_TEST_CHECK(dst.rank == 3);
  CUDA_TEST_CHECK(dst.num_el == 0);
  CUDA_TEST_CHECK(dst.offset == 0);
  CUDA_TEST_CHECK(dst.shape[0] == 2 && dst.shape[1] == 0 && dst.shape[2] == 3);
  CUDA_TEST_CHECK(dst.strides[0] == 0 && dst.strides[1] == 3 &&
                  dst.strides[2] == 1);
  CUDA_TEST_CHECK(dst.storage != NULL && dst.storage != src.storage);
  CUDA_TEST_CHECK(dst.storage->n_bytes == 0);
  CUDA_TEST_CHECK(dst.storage->data == NULL);

  bool contiguous = false;
  CUDA_TEST_STATUS(T_tensor_is_contiguous(&dst, &contiguous), TT_OK);
  CUDA_TEST_CHECK(contiguous);

  /* No data transfer for a zero-element tensor. */
  CUDA_TEST_CHECK(copy_calls == copies_before);

cleanup:
  T_destroy_tensor(&dst);
  T_destroy_tensor(&src);

  if (check_no_live_allocations() != 0)
    result = 1;

  return result;
}

static int test_invalid_cuda_device_id(void) {
  const i64 shape[] = {2, 3};
  const float data[] = {1, 2, 3, 4, 5, 6};

  Tensor src = {0};
  Tensor dst = {0};
  int result = 0;
  int device_was_retagged = 0;

  reset_injection();

  CUDA_TEST_STATUS(T_create_tensor(&src, data, F32, shape, 2, cpu_device()),
                   TT_OK);

  /*
   * Retag the CPU allocation with an invalid CUDA ID. Validation must fail
   * before backend access. The original device is restored before destruction.
   */
  src.storage->device = (Device){.devtype = CUDA, .id = -1};
  device_was_retagged = 1;

  CUDA_TEST_STATUS(T_tensor_contiguous_materialize(&dst, &src),
                   TT_ERR_INVALID_ARG);
  CUDA_TEST_CHECK(tensor_is_empty(&dst));

cleanup:
  if (device_was_retagged && src.storage != NULL)
    src.storage->device = cpu_device();

  T_destroy_tensor(&dst);
  T_destroy_tensor(&src);

  if (check_no_live_allocations() != 0)
    result = 1;

  return result;
}

static int test_null_source_data(void) {
  const i64 shape[] = {2, 3};
  const float data[] = {1, 2, 3, 4, 5, 6};

  Tensor src = {0};
  Tensor dst = {0};
  void *original_data = NULL;
  int result = 0;
  int source_data_was_cleared = 0;

  reset_injection();

  CUDA_TEST_STATUS(T_create_tensor(&src, data, F32, shape, 2, cpu_device()),
                   TT_OK);

  original_data = src.storage->data;
  src.storage->data = NULL;
  source_data_was_cleared = 1;

  CUDA_TEST_STATUS(T_tensor_contiguous_materialize(&dst, &src),
                   TT_ERR_OUT_OF_RANGE);
  CUDA_TEST_CHECK(tensor_is_empty(&dst));

cleanup:
  if (source_data_was_cleared && src.storage != NULL)
    src.storage->data = original_data;

  T_destroy_tensor(&dst);
  T_destroy_tensor(&src);

  if (check_no_live_allocations() != 0)
    result = 1;

  return result;
}

static int test_destination_allocation_failure(void) {
  const i64 shape[] = {2, 3};
  const float data[] = {1, 2, 3, 4, 5, 6};

  Tensor src = {0};
  Tensor dst = {0};
  int result = 0;

  reset_injection();

  CUDA_TEST_STATUS(T_create_tensor(&src, data, F32, shape, 2, cpu_device()),
                   TT_OK);

  const long baseline = live_allocs;
  CUDA_TEST_CHECK(baseline == 1);

  fail_alloc = 1;
  CUDA_TEST_STATUS(T_tensor_contiguous_materialize(&dst, &src),
                   TT_ERR_NO_MEMORY);
  fail_alloc = 0;

  CUDA_TEST_CHECK(tensor_is_empty(&dst));
  CUDA_TEST_CHECK(live_allocs == baseline);

  /* The source is untouched and still usable. */
  CUDA_TEST_CHECK(memcmp(src.storage->data, data, sizeof(data)) == 0);

cleanup:
  fail_alloc = 0;

  T_destroy_tensor(&dst);
  T_destroy_tensor(&src);

  if (check_no_live_allocations() != 0)
    result = 1;

  return result;
}

static int test_copy_failure(void) {
  const i64 shape[] = {2, 3};
  const float data[] = {1, 2, 3, 4, 5, 6};

  Tensor src = {0};
  Tensor dst = {0};
  int result = 0;

  reset_injection();

  CUDA_TEST_STATUS(T_create_tensor(&src, data, F32, shape, 2, cpu_device()),
                   TT_OK);

  const long baseline = live_allocs;
  const long copies_before = copy_calls;

  /*
   * Contiguous materialization uses storage_copy(), which dispatches through
   * the fault-injecting CPU backend.
   */
  fail_copy = 1;

  CUDA_TEST_STATUS(T_tensor_contiguous_materialize(&dst, &src),
                   TT_ERR_INTERNAL);
  fail_copy = 0;

  CUDA_TEST_CHECK(copy_calls > copies_before);
  CUDA_TEST_CHECK(tensor_is_empty(&dst));
  CUDA_TEST_CHECK(live_allocs == baseline);

  /* Source remains valid and unchanged. */
  CUDA_TEST_CHECK(memcmp(src.storage->data, data, sizeof(data)) == 0);

cleanup:
  fail_copy = 0;

  T_destroy_tensor(&dst);
  T_destroy_tensor(&src);

  if (check_no_live_allocations() != 0)
    result = 1;

  return result;
}

#endif /* !TEN_TEST_CUDA_ONLY */

int main(void) {
#ifdef TEN_TEST_CUDA_ONLY
  int device_status = ten_cuda_test_require_device();
  if (device_status != 0)
    return device_status;

  if (test_cuda_contiguous() != 0 || test_cuda_transposed() != 0 ||
      test_cuda_sliced() != 0)
    return EXIT_FAILURE;
#else
  if (test_cpu_contiguous() != 0 || test_cpu_transposed() != 0 ||
      test_cpu_sliced() != 0 || test_empty_tensor() != 0 ||
      test_invalid_cuda_device_id() != 0 || test_null_source_data() != 0 ||
      test_destination_allocation_failure() != 0 || test_copy_failure() != 0)
    return EXIT_FAILURE;
#endif

  printf("PASS: %d checks\n", tests_run);
  return EXIT_SUCCESS;
}
