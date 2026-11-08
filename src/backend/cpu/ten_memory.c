#include "ten_memory.h"
#include "ten/ten_status.h"

#include <stdlib.h>
#include <string.h>

Ten_Status cpu_allocate(void **ptr, size_t n_bytes) {
  if (ptr == NULL)
    return status_error(TT_ERR_INVALID_ARG, "output pointer is NULL");

  *ptr = NULL;

  if (n_bytes == 0)
    return status_ok();

  *ptr = malloc(n_bytes);

  if (*ptr == NULL)
    return status_error(TT_ERR_INTERNAL, "CPU memory allocation failed");

  return status_ok();
}

Ten_Status cpu_deallocate(void **ptr) {
  if (ptr == NULL)
    return status_error(TT_ERR_INVALID_ARG, "pointer argument is NULL");

  if (*ptr == NULL)
    return status_ok();

  free(*ptr);
  *ptr = NULL;

  return status_ok();
}
Ten_Status cpu_copy(void *dst, const void *src, size_t n_bytes,
                    Ten_CopyKind kind) {
  if (n_bytes == 0)
    return status_ok();

  if (kind != TEN_COPY_HOST_TO_HOST)
    return status_error(TT_ERR_INVALID_ARG,
                        "CPU backend only supports host-to-host copying");

  if (dst == NULL || src == NULL)
    return status_error(TT_ERR_INVALID_ARG,
                        "Source or destination pointer is NULL");

  memcpy(dst, src, n_bytes);
  return status_ok();
}
