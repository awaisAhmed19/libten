#include <stddef.h>
#include <stdio.h>

#include "backend/cuda/ten_memory.h"
#include "cuda_test_common.h"

int main(void) {
  int device_status = ten_cuda_test_require_device();
  if (device_status != 0)
    return device_status;

  int result = 0;
  void *device_ptr = NULL;
  void *zero_ptr = (void *)1;

  /* Zero-byte allocation is a successful no-op and clears the output. */
  CUDA_TEST_STATUS(cuda_allocate(&zero_ptr, 0), TT_OK);
  CUDA_TEST_CHECK(zero_ptr == NULL);

  /* A real allocation must return a non-NULL device pointer. */
  CUDA_TEST_STATUS(cuda_allocate(&device_ptr, 256), TT_OK);
  CUDA_TEST_CHECK(device_ptr != NULL);

  /* Deallocation releases the allocation and clears the caller's pointer. */
  CUDA_TEST_STATUS(cuda_deallocate(&device_ptr), TT_OK);
  CUDA_TEST_CHECK(device_ptr == NULL);

  /* Deallocating an already-NULL pointer is intentionally idempotent. */
  CUDA_TEST_STATUS(cuda_deallocate(&device_ptr), TT_OK);
  CUDA_TEST_CHECK(device_ptr == NULL);

  /* Invalid pointer-to-pointer arguments must be rejected. */
  CUDA_TEST_STATUS(cuda_allocate(NULL, 16), TT_ERR_INVALID_ARG);
  CUDA_TEST_STATUS(cuda_deallocate(NULL), TT_ERR_INVALID_ARG);

  puts("CUDA memory tests passed.");

cleanup:
  if (device_ptr != NULL)
    (void)cuda_deallocate(&device_ptr);
  return result;
}
