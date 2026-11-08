#ifndef TEN_CUDA_OPS_H
#define TEN_CUDA_OPS_H

#include "ten/ten_status.h"
#include "ten/ten_tensor.h"
#ifdef __cplusplus
extern "C" {
#endif
Ten_Status cuda_add(Tensor *dst, const Tensor *a, const Tensor *b);
Ten_Status cuda_matmul(Tensor *dst, const Tensor *a, const Tensor *b);
#ifdef __cplusplus
}
#endif

#endif /* TEN_CUDA_OPS_H*/
