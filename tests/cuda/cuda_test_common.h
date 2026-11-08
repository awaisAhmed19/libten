#ifndef TEN_CUDA_TEST_COMMON_H
#define TEN_CUDA_TEST_COMMON_H

#include "ten/ten_status.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

/*
 * Return 0 when a usable CUDA device is available, or 77 when the test
 * should be skipped by CTest.
 */
static int ten_cuda_test_require_device(void) {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);

  if (err != cudaSuccess || device_count == 0) {
    fprintf(stderr, "[SKIP] No usable CUDA device: %s\n",
            err == cudaSuccess ? "no CUDA devices found"
                               : cudaGetErrorString(err));
    (void)cudaGetLastError();
    return 77;
  }

  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    fprintf(stderr, "[SKIP] Could not select CUDA device 0: %s\n",
            cudaGetErrorString(err));
    (void)cudaGetLastError();
    return 77;
  }

  return 0;
}

#define CUDA_TEST_CHECK(condition)                                             \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);     \
      result = 1;                                                              \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

#define CUDA_TEST_STATUS(expression, expected)                                 \
  do {                                                                         \
    Ten_Status test_status = (expression);                                     \
    if (test_status.stype != (expected)) {                                     \
      fprintf(stderr, "FAIL %s:%d: expected status %d, got %d (%s)\n",         \
              __FILE__, __LINE__, (expected), test_status.stype,               \
              test_status.msg ? test_status.msg : "no message");               \
      result = 1;                                                              \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

#endif
