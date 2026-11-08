#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "backend/cuda/ten_memory.h"
#include "cuda_test_common.h"

int main(void) {
  int device_status = ten_cuda_test_require_device();
  if (device_status != 0)
    return device_status;

  int result = 0;
  const float source[] = {1.25f, -2.0f, 3.5f, 4.75f,
                          0.0f,  9.0f,  -1.5f, 8.25f};
  const size_t n_bytes = sizeof(source);

  float host_result[sizeof(source) / sizeof(source[0])] = {0};
  float host_result_d2d[sizeof(source) / sizeof(source[0])] = {0};
  float host_copy[sizeof(source) / sizeof(source[0])] = {0};

  void *device_a = NULL;
  void *device_b = NULL;

  CUDA_TEST_STATUS(cuda_allocate(&device_a, n_bytes), TT_OK);
  CUDA_TEST_CHECK(device_a != NULL);

  CUDA_TEST_STATUS(cuda_allocate(&device_b, n_bytes), TT_OK);
  CUDA_TEST_CHECK(device_b != NULL);

  /* Host -> Device */
  CUDA_TEST_STATUS(
      cuda_copy(device_a, source, n_bytes, TEN_COPY_HOST_TO_DEVICE), TT_OK);

  /* Device -> Host */
  CUDA_TEST_STATUS(
      cuda_copy(host_result, device_a, n_bytes, TEN_COPY_DEVICE_TO_HOST),
      TT_OK);
  CUDA_TEST_CHECK(memcmp(source, host_result, n_bytes) == 0);

  /* Device -> Device */
  CUDA_TEST_STATUS(
      cuda_copy(device_b, device_a, n_bytes, TEN_COPY_DEVICE_TO_DEVICE),
      TT_OK);

  CUDA_TEST_STATUS(
      cuda_copy(host_result_d2d, device_b, n_bytes, TEN_COPY_DEVICE_TO_HOST),
      TT_OK);
  CUDA_TEST_CHECK(memcmp(source, host_result_d2d, n_bytes) == 0);

  /* Host -> Host is also supported by the CUDA memory backend. */
  CUDA_TEST_STATUS(
      cuda_copy(host_copy, source, n_bytes, TEN_COPY_HOST_TO_HOST), TT_OK);
  CUDA_TEST_CHECK(memcmp(source, host_copy, n_bytes) == 0);

  /* Invalid direction and non-zero-byte NULL pointers must be rejected. */
  CUDA_TEST_STATUS(
      cuda_copy(host_copy, source, sizeof(float), (Ten_CopyKind)-1),
      TT_ERR_INVALID_ARG);
  CUDA_TEST_STATUS(
      cuda_copy(NULL, source, sizeof(float), TEN_COPY_HOST_TO_DEVICE),
      TT_ERR_INVALID_ARG);
  CUDA_TEST_STATUS(
      cuda_copy(host_copy, NULL, sizeof(float), TEN_COPY_DEVICE_TO_HOST),
      TT_ERR_INVALID_ARG);

  /*
   * The implementation treats zero-byte copies as no-ops before validating
   * pointers or direction. These assertions document that contract.
   */
  CUDA_TEST_STATUS(cuda_copy(NULL, NULL, 0, TEN_COPY_HOST_TO_DEVICE), TT_OK);
  CUDA_TEST_STATUS(cuda_copy(NULL, NULL, 0, (Ten_CopyKind)-1), TT_OK);

  puts("CUDA copy tests passed.");

cleanup:
  if (device_a != NULL)
    (void)cuda_deallocate(&device_a);
  if (device_b != NULL)
    (void)cuda_deallocate(&device_b);
  return result;
}
