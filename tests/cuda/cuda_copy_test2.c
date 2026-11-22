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
#include "internal/ten_backend.h"
#include "ten/ten_status.h"
#include "ten/ten_tensor.h"
#ifdef TEN_TEST_HAS_CUDA
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#endif

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
static Device cuda_device(void) { return (Device){.devtype = CUDA, .id = 0}; }

static int tensor_is_empty(const Tensor *t) {
  return t != NULL && t->storage == NULL && t->shape == NULL &&
         t->strides == NULL && t->rank == 0 && t->offset == 0 && t->num_el == 0;
}

/* Shared success-case runner for CPU and CUDA. */

typedef enum { CASE_CONTIGUOUS, CASE_TRANSPOSED, CASE_SLICED } CaseKind;

static int run_case(Device dev, CaseKind kind) {
  static const float expected_contiguous[] = {1, 2, 3, 4, 5, 6};
  static const float expected_transposed[] = {1, 4, 2, 5, 3, 6};
  static const float expected_sliced[] = {0, 2, 4, 6, 8, 10};

  float data[12];
  i64 shape[2];
  i64 out_shape[2];
  i64 out_strides[2];
  const float *expected;
  i64 count;

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

  Tensor src = {0};
  Tensor view = {0};
  Tensor dst = {0};
  const Tensor *source = &src;

  CHECK_STATUS(T_create_tensor(&src, data, F32, shape, 2, dev), TT_OK);

  if (kind == CASE_TRANSPOSED) {
    CHECK_STATUS(T_tensor_transpose(&view, &src, 0, 1), TT_OK);
    source = &view;
  } else if (kind == CASE_SLICED) {
    CHECK_STATUS(T_tensor_slice(&view, &src, 1, 0, 4, 2), TT_OK);
    source = &view;
  }

  if (kind != CASE_CONTIGUOUS) {
    bool contiguous = true;
    CHECK_STATUS(T_tensor_is_contiguous(source, &contiguous), TT_OK);
    CHECK(!contiguous);
  }

  CHECK_STATUS(T_tensor_contiguous_materialize(&dst, source), TT_OK);

  CHECK(dst.storage != NULL && dst.storage != source->storage);
  CHECK(dst.storage->data != src.storage->data);
  CHECK(dst.rank == 2);
  CHECK(dst.num_el == count);
  CHECK(dst.offset == 0);
  CHECK(dst.shape[0] == out_shape[0] && dst.shape[1] == out_shape[1]);
  CHECK(dst.strides[0] == out_strides[0] && dst.strides[1] == out_strides[1]);

  bool contiguous = false;
  CHECK_STATUS(T_tensor_is_contiguous(&dst, &contiguous), TT_OK);
  CHECK(contiguous);

  /* The output must remain on the source device. */
  Device out_device;
  CHECK(T_get_device(&dst, &out_device));
  CHECK(out_device.devtype == dev.devtype && out_device.id == dev.id);

  /* Read back through the backend; never dereference device memory. */
  void *ptr = NULL;
  Device ptr_device;
  float host[12] = {0};

  CHECK_STATUS(T_device_data(&dst, &ptr, &ptr_device), TT_OK);
  CHECK(ptr != NULL);
  CHECK(ptr_device.devtype == dev.devtype && ptr_device.id == dev.id);
  CHECK_STATUS(storage_copy(host, cpu_device(), ptr, ptr_device,
                            (size_t)count * sizeof(float)),
               TT_OK);
  CHECK(memcmp(host, expected, (size_t)count * sizeof(float)) == 0);

  T_destroy_tensor(&dst);
  T_destroy_tensor(&view);
  T_destroy_tensor(&src);
  CHECK(tensor_is_empty(&dst));
  return 0;
}

static int test_cpu_contiguous(void) {
  reset_injection();
  CHECK(run_case(cpu_device(), CASE_CONTIGUOUS) == 0);

  /* Mutating the output must not change the source. */
  const i64 shape[] = {2, 3};
  const float data[] = {1, 2, 3, 4, 5, 6};
  Tensor src = {0};
  Tensor dst = {0};

  CHECK_STATUS(T_create_tensor(&src, data, F32, shape, 2, cpu_device()), TT_OK);
  CHECK_STATUS(T_tensor_contiguous_materialize(&dst, &src), TT_OK);
  ((float *)dst.storage->data)[0] = 99.0f;
  CHECK(((float *)src.storage->data)[0] == 1.0f);

  T_destroy_tensor(&dst);
  T_destroy_tensor(&src);
  CHECK(live_allocs == 0);
  return 0;
}

static int test_cpu_transposed(void) {
  reset_injection();
  CHECK(run_case(cpu_device(), CASE_TRANSPOSED) == 0);
  CHECK(live_allocs == 0);
  return 0;
}

static int test_cpu_sliced(void) {
  reset_injection();
  CHECK(run_case(cpu_device(), CASE_SLICED) == 0);
  CHECK(live_allocs == 0);
  return 0;
}

#ifdef TEN_TEST_HAS_CUDA
static bool cuda_device_available(void) {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);

  if (err != cudaSuccess || count == 0) {
    (void)cudaGetLastError();
    return false;
  }

  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }

  return true;
}
#endif

static int run_cuda_case(CaseKind kind, const char *name) {
#ifdef TEN_TEST_HAS_CUDA
  if (!cuda_device_available()) {
    fprintf(stderr, "[SKIP] %s: no usable CUDA device\n", name);
    return 0;
  }

  reset_injection();
  CHECK(run_case(cuda_device(), kind) == 0);
  return 0;
#else
  (void)kind;
  fprintf(stderr, "[SKIP] %s: CUDA not compiled\n", name);
  return 0;
#endif
}

static int test_cuda_contiguous(void) {
  return run_cuda_case(CASE_CONTIGUOUS, "cuda contiguous");
}

static int test_cuda_transposed(void) {
  return run_cuda_case(CASE_TRANSPOSED, "cuda transposed");
}

static int test_cuda_sliced(void) {
  return run_cuda_case(CASE_SLICED, "cuda sliced");
}

static int test_empty_tensor(void) {
  const i64 shape[] = {2, 0, 3};
  Tensor src = {0};
  Tensor dst = {0};

  reset_injection();

  CHECK_STATUS(T_create_tensor(&src, NULL, F32, shape, 3, cpu_device()), TT_OK);

  const long copies_before = copy_calls;

  CHECK_STATUS(T_tensor_contiguous_materialize(&dst, &src), TT_OK);

  CHECK(dst.rank == 3);
  CHECK(dst.num_el == 0);
  CHECK(dst.offset == 0);
  CHECK(dst.shape[0] == 2 && dst.shape[1] == 0 && dst.shape[2] == 3);
  CHECK(dst.strides[0] == 0 && dst.strides[1] == 3 && dst.strides[2] == 1);
  CHECK(dst.storage != NULL && dst.storage != src.storage);
  CHECK(dst.storage->n_bytes == 0);
  CHECK(dst.storage->data == NULL);

  bool contiguous = false;
  CHECK_STATUS(T_tensor_is_contiguous(&dst, &contiguous), TT_OK);
  CHECK(contiguous);

  /* No data transfer for a zero-element tensor. */
  CHECK(copy_calls == copies_before);

  T_destroy_tensor(&dst);
  T_destroy_tensor(&src);
  CHECK(live_allocs == 0);
  return 0;
}

static int test_invalid_cuda_device_id(void) {
  const i64 shape[] = {2, 3};
  const float data[] = {1, 2, 3, 4, 5, 6};
  Tensor src = {0};
  Tensor dst = {0};

  reset_injection();

  CHECK_STATUS(T_create_tensor(&src, data, F32, shape, 2, cpu_device()), TT_OK);

  /*
   * Retag the CPU allocation with an invalid CUDA ID. Validation must fail
   * before any backend access. id == 1 on a single-GPU machine is only
   * rejected once AT02 adds the Device.id check, so it is not tested here.
   */
  src.storage->device = (Device){.devtype = CUDA, .id = -1};

  CHECK_STATUS(T_tensor_contiguous_materialize(&dst, &src), TT_ERR_INVALID_ARG);
  CHECK(tensor_is_empty(&dst));

  src.storage->device = cpu_device();
  T_destroy_tensor(&src);
  CHECK(live_allocs == 0);
  return 0;
}

static int test_null_source_data(void) {
  const i64 shape[] = {2, 3};
  const float data[] = {1, 2, 3, 4, 5, 6};
  Tensor src = {0};
  Tensor dst = {0};

  reset_injection();

  CHECK_STATUS(T_create_tensor(&src, data, F32, shape, 2, cpu_device()), TT_OK);

  void *original = src.storage->data;
  src.storage->data = NULL;

  CHECK_STATUS(T_tensor_contiguous_materialize(&dst, &src),
               TT_ERR_OUT_OF_RANGE);
  CHECK(tensor_is_empty(&dst));

  src.storage->data = original;
  T_destroy_tensor(&src);
  CHECK(live_allocs == 0);
  return 0;
}

static int test_destination_allocation_failure(void) {
  const i64 shape[] = {2, 3};
  const float data[] = {1, 2, 3, 4, 5, 6};
  Tensor src = {0};
  Tensor dst = {0};

  reset_injection();

  CHECK_STATUS(T_create_tensor(&src, data, F32, shape, 2, cpu_device()), TT_OK);
  const long baseline = live_allocs;
  CHECK(baseline == 1);

  fail_alloc = 1;
  CHECK_STATUS(T_tensor_contiguous_materialize(&dst, &src), TT_ERR_NO_MEMORY);
  fail_alloc = 0;

  CHECK(tensor_is_empty(&dst));
  CHECK(live_allocs == baseline);

  /* The source is untouched and still usable. */
  CHECK(memcmp(src.storage->data, data, sizeof(data)) == 0);

  T_destroy_tensor(&src);
  CHECK(live_allocs == 0);
  return 0;
}

static int test_copy_failure(void) {
  const i64 shape[] = {2, 3};
  const float data[] = {1, 2, 3, 4, 5, 6};
  Tensor src = {0};
  Tensor view = {0};
  Tensor dst = {0};

  reset_injection();

  CHECK_STATUS(T_create_tensor(&src, data, F32, shape, 2, cpu_device()), TT_OK);
  CHECK_STATUS(T_tensor_transpose(&view, &src, 0, 1), TT_OK);
  const long baseline = live_allocs;

  /*
   * Materialize does not call T_copy_tensor_data(); its only backend copy is
   * the final transfer into the destination storage.
   */
  fail_copy = 1;
  CHECK_STATUS(T_tensor_contiguous_materialize(&dst, &view), TT_ERR_INTERNAL);
  fail_copy = 0;

  CHECK(tensor_is_empty(&dst));
  CHECK(live_allocs == baseline);

  T_destroy_tensor(&view);
  T_destroy_tensor(&src);
  CHECK(live_allocs == 0);
  return 0;
}

int main(void) {
  if (test_cpu_contiguous() || test_cpu_transposed() || test_cpu_sliced() ||
      test_cuda_contiguous() || test_cuda_transposed() || test_cuda_sliced() ||
      test_empty_tensor() || test_invalid_cuda_device_id() ||
      test_null_source_data() || test_destination_allocation_failure() ||
      test_copy_failure()) {
    return EXIT_FAILURE;
  }

  printf("PASS: %d checks\n", tests_run);
  return EXIT_SUCCESS;
}
