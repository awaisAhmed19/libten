#include "ten/ten_types.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

RangeStatus T_checked_add_size(size_t a, size_t b, size_t *out) {
  if (out == NULL || a > SIZE_MAX - b)
    return RNG_OVERFLOW;

  *out = a + b;
  return RNG_OK;
}
RangeStatus T_checked_mul_size(size_t a, size_t b, size_t *out) {
  if (out == NULL || (b != 0 && a > SIZE_MAX / b))
    return RNG_OVERFLOW;

  *out = a * b;
  return RNG_OK;
}
RangeStatus T_check_i8_add(i8 a, i8 b) {
  i8 result;
  if (__builtin_add_overflow(a, b, &result))
    return (a < 0 && b < 0) ? RNG_UNDERFLOW : RNG_OVERFLOW;
  return RNG_OK;
}
RangeStatus T_check_i16_add(i16 a, i16 b) {
  i16 result;
  if (__builtin_add_overflow(a, b, &result))
    return (a < 0 && b < 0) ? RNG_UNDERFLOW : RNG_OVERFLOW;
  return RNG_OK;
}
RangeStatus T_check_i32_add(i32 a, i32 b) {
  i32 result;
  if (__builtin_add_overflow(a, b, &result))
    return (a < 0 && b < 0) ? RNG_UNDERFLOW : RNG_OVERFLOW;
  return RNG_OK;
}
RangeStatus T_check_i64_add(i64 a, i64 b) {
  i64 result;
  if (__builtin_add_overflow(a, b, &result))
    return (a < 0 && b < 0) ? RNG_UNDERFLOW : RNG_OVERFLOW;
  return RNG_OK;
}
RangeStatus T_check_u8_add(u8 a, u8 b) {
  if (a > U8_MAX - b) {
    return RNG_OVERFLOW;
  }
  return RNG_OK;
}
RangeStatus T_check_u16_add(u16 a, u16 b) {
  if (a > U16_MAX - b) {
    return RNG_OVERFLOW;
  }
  return RNG_OK;
}
RangeStatus T_check_u32_add(u32 a, u32 b) {
  if (a > U32_MAX - b) {
    return RNG_OVERFLOW;
  }
  return RNG_OK;
}
RangeStatus T_check_u64_add(u64 a, u64 b) {
  if (a > U64_MAX - b) {
    return RNG_OVERFLOW;
  }
  return RNG_OK;
}
RangeStatus T_check_u8_sub(u8 a, u8 b) {
  if (a < b) {
    return RNG_UNDERFLOW;
  }
  return RNG_OK;
}
RangeStatus T_check_u16_sub(u16 a, u16 b) {
  if (a < b) {
    return RNG_UNDERFLOW;
  }
  return RNG_OK;
}
RangeStatus T_check_u32_sub(u32 a, u32 b) {
  if (a < b) {
    return RNG_UNDERFLOW;
  }
  return RNG_OK;
}
RangeStatus T_check_u64_sub(u64 a, u64 b) {
  if (a < b) {
    return RNG_UNDERFLOW;
  }
  return RNG_OK;
}
RangeStatus T_check_f32_add(f32 a, f32 b) {
  if (!isfinite(a) || !isfinite(b))
    return RNG_INVALID_ARG;

  f32 result = a + b;

  if (isinf(result))
    return RNG_OVERFLOW;

  return RNG_OK;
}
RangeStatus T_check_f64_add(f64 a, f64 b) {
  if (!isfinite(a) || !isfinite(b))
    return RNG_INVALID_ARG;

  f64 result = a + b;

  if (isinf(result))
    return RNG_OVERFLOW;

  return RNG_OK;
}
