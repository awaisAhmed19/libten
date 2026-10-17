#pragma once
#include "ten_types.h"

#define TT_MAX_RANK 8
typedef struct Tensor Tensor;

extern u0 *T_allocate_tensor_data(size_t rows, size_t cols);
extern bool T_is_shape_valid(const Tensor *t1,
                             const Tensor *t2); /*
                               1. this function checks if the number of
                               shape array elements is less than TT_MAX_RANK
                               TODO: need function for
                               2. t1.shape == t2.shape for matmul
                               3. shape compatible for broadcasting rules

                                                 */

extern i64 T_get_numel(const Tensor *t);
