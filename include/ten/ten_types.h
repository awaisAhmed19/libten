#pragma once
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#define str(x) #x
#define xstr(x) str(x)

#define ZERO 0

#define U8_MIN ZERO
#define U16_MIN ZERO
#define U32_MIN ZERO
#define U64_MIN ZERO

#define U8_MAX UINT8_MAX
#define U16_MAX UINT16_MAX
#define U32_MAX UINT32_MAX
#define U64_MAX UINT64_MAX

#define I8_MAX INT8_MAX
#define I8_MIN INT8_MIN

#define I16_MAX INT16_MAX
#define I16_MIN INT16_MIN

#define I32_MAX INT32_MAX
#define I32_MIN INT32_MIN

#define I64_MAX INT64_MAX
#define I64_MIN INT64_MIN

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef void u0;

typedef float f32;
typedef double f64;

enum DevType { CPU = 0, CUDA = 1 };

typedef enum RangeStatus {
  RNG_OK = 0,
  RNG_UNDERFLOW,
  RNG_OVERFLOW,
  RNG_DIVISIBLE_BY_ZERO,
  RNG_INVALID_ARG
} RangeStatus;

typedef struct {
  void *result; // can be made to be any DataType for a valid result or return
                // null;
  RangeStatus status;
} Results;
// else just do a check in a fuction and define the operation after the check
// fuction passes then this result struct is unnecessary
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

typedef Complex64 c64;
typedef struct {
  double real;
  double imaginary;
} Complex128;

typedef Complex128 c128;
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
extern size_t T_get_dtype_size(enum DataType dtype);

extern bool T_is_dtype_sint(const enum DataType type);
extern bool T_is_dtype_int(const enum DataType type);
extern bool T_is_dtype_float(const enum DataType type);
extern bool T_is_dtype_uint(const enum DataType type);
extern bool T_is_dtype_complex(const enum DataType type);
extern bool T_is_dtype_valid(const enum DataType type);
extern RangeStatus T_checked_add_size(size_t a, size_t b, size_t *out);
extern RangeStatus T_checked_mul_size(size_t a, size_t b, size_t *out);

extern RangeStatus T_check_i8_add(i8 a, i8 b);
extern RangeStatus T_check_i16_add(i16 a, i16 b);
extern RangeStatus T_check_i32_add(i32 a, i32 b);
extern RangeStatus T_check_i64_add(i64 a, i64 b);

extern RangeStatus T_check_u8_add(u8 a, u8 b);
extern RangeStatus T_check_u16_add(u16 a, u16 b);
extern RangeStatus T_check_u32_add(u32 a, u32 b);
extern RangeStatus T_check_u64_add(u64 a, u64 b);

extern RangeStatus T_check_u8_sub(u8 a, u8 b);
extern RangeStatus T_check_u16_sub(u16 a, u16 b);
extern RangeStatus T_check_u32_sub(u32 a, u32 b);
extern RangeStatus T_check_u64_sub(u64 a, u64 b);

extern RangeStatus T_check_f64_add(f64 a, f64 b);
extern RangeStatus T_check_f32_add(f32 a, f32 b);
