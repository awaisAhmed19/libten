#include "./ten_ops.h"
#include <string.h>

Ten_Status cpu_storage_fill(void *data, const void *host_scalar, DataType dtype,
                            size_t numel) {
  if (host_scalar == NULL || (numel > 0 && data == NULL))
    return status_error(TT_ERR_INVALID_ARG, "invalid fill arguments");

  size_t dtype_size = T_get_dtype_size(dtype);

  if (dtype_size == 0 || T_is_dtype_complex(dtype))
    return status_error(TT_ERR_DTYPE_UNSUPPORTED, "unsupported fill dtype");

  if (numel == 0)
    return status_ok();

  size_t total_bytes = 0;

  if (T_checked_mul_size(numel, dtype_size, &total_bytes) != RNG_OK)
    return status_error(TT_ERR_OVERFLOW, "fill size overflow");

  unsigned char *dst = (unsigned char *)data;

  for (size_t i = 0; i < numel; ++i) {
    memcpy(dst + i * dtype_size, host_scalar, dtype_size);
  }

  return status_ok();
}
