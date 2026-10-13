#pragma once
#include "ten_types.h"

#define TT_MAX_RANK 8
struct tt_storage {
  void *data; /* may be NULL only when nbytes == 0 */
  size_t nbytes;
  Device device;
  uint32_t refcount;                      /* C11 <stdatomic.h> */
  void (*free_fn)(void *data, void *ctx); /* NULL = borrowed, never freed */
  void *free_ctx;
  uint32_t magic; /* canary, cleared on free */
};

typedef struct {
  struct tt_storage *storage; /* strong reference */
  DataType dtype;
  int32_t rank;
  int64_t shape[TT_MAX_RANK];
  int64_t strides[TT_MAX_RANK];
  int64_t offset; /* elements */
  int64_t numel;
  uint32_t magic;
} Tensor;
