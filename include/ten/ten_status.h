#pragma once

enum Ten_StatusType {
  TT_OK = 0,
  TT_ERR_INVALID_ARG = 1, /* null pointer, bad rank, negative dim, bad axis */
  TT_ERR_SHAPE_MISMATCH = 2,    /* incompatible shapes, out has wrong shape */
  TT_ERR_DTYPE_MISMATCH = 3,    /* operands have different dtypes */
  TT_ERR_DTYPE_UNSUPPORTED = 4, /* dtype not supported by this operation */
  TT_ERR_NO_MEMORY = 5,
  TT_ERR_OVERFLOW = 6,         /* size or offset arithmetic would overflow */
  TT_ERR_OUT_OF_RANGE = 7,     /* index or scalar out of range */
  TT_ERR_NOT_CONTIGUOUS = 8,   /* operation needs a contiguous tensor */
  TT_ERR_ALIASING = 9,         /* out overlaps an input or itself */
  TT_ERR_DEVICE_MISMATCH = 10, /* tensor device differs from ctx device */
  TT_ERR_UNSUPPORTED = 11,     /* operation, device or feature not available */
  TT_ERR_INVALID_STATE = 12,   /* destroyed or corrupt handle, poisoned ctx */
  TT_ERR_CUDA = 13,
  TT_ERR_INTERNAL = 14
};

typedef struct {
  enum Ten_StatusType stype;
  char *msg;
} Ten_Status;
