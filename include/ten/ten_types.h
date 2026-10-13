#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#define str(x) #x
#define xstr(x) str(x)

enum DevType { CPU = 0, CUDA };

typedef enum { TT_SCALAR_FLOAT, TT_SCALAR_INT } TT_ScalarKind;

typedef struct {
  TT_ScalarKind kind;
  union {
    double f;
    int64_t i;
  } value;
} TT_Scalar;

typedef struct {
  float real;
  float imaginary;
} Complex64;

typedef struct {
  double real;
  double imaginary;
} Complex128;

typedef struct {
  enum DevType devtype;
  int id;
} Device;

typedef enum DataType {
  I8 = 0,
  I16,
  I32,
  I64,

  U8,
  U16,
  U32,
  U64,

  F32,
  F64,

  C64,
  C128,

  DTYPE_COUNT
} DataType;

extern const double PI;
extern size_t ten_dtype_size(enum DataType dtype);

extern bool T_is_dtype_sint(const enum DataType type);
extern bool T_is_dtype_int(const enum DataType type);
extern bool T_is_dtype_float(const enum DataType type);
extern bool T_is_dtype_uint(const enum DataType type);
extern bool T_is_dtype_complex(const enum DataType type);
