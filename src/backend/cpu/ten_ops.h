#ifndef TEN_CPU_OPS_H
#define TEN_CPU_OPS_H

#include "ten/ten_status.h"
#include "ten/ten_tensor.h"
#ifdef __cplusplus
extern "C" {
#endif
Ten_Status cpu_add(Tensor *dst, const Tensor *a, const Tensor *b);
Ten_Status cpu_matmul(Tensor *dst, const Tensor *a, const Tensor *b);
Ten_Status cpu_storage_fill(void *dst, const void *host_scalar, DataType dtype,
                            size_t numel);
#ifdef __cplusplus
}
#endif

#endif /* TEN_CPU_OPS_H*/
