#include "ten/ten_status.h"
#include "ten/ten_types.h"
#include <stdbool.h>
#include <stddef.h>
const double PI =
    3.14159265358979323846264338327950288419716939937510582097494458734437321538295376117614603945866419742306;
const enum DataType promotion_table[DTYPE_COUNT][DTYPE_COUNT] = {
    /*            I8    I16   I32   I64   U8    U16   U32   U64   F32   F64 C64
       C128 */
    /* I8   */ {I8, I16, I32, I64, I16, I32, I64, F64, F32, F64, C64, C128},
    /* I16  */ {I16, I16, I32, I64, I16, I32, I64, F64, F32, F64, C64, C128},
    /* I32  */ {I32, I32, I32, I64, I32, I32, I64, F64, F32, F64, C64, C128},
    /* I64  */ {I64, I64, I64, I64, I64, I64, I64, F64, F64, F64, C64, C128},

    /* U8   */ {I16, I16, I32, I64, U8, U16, U32, U64, F32, F64, C64, C128},
    /* U16  */ {I32, I32, I32, I64, U16, U16, U32, U64, F32, F64, C64, C128},
    /* U32  */ {I64, I64, I64, I64, U32, U32, U32, U64, F32, F64, C64, C128},
    /* U64  */ {F64, F64, F64, F64, U64, U64, U64, U64, F64, F64, C64, C128},

    /* F32  */ {F32, F32, F32, F64, F32, F32, F32, F64, F32, F64, C64, C128},
    /* F64  */ {F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, C128, C128},

    /* C64  */ {C64, C64, C64, C64, C64, C64, C64, C64, C64, C128, C64, C128},
    /* C128 */
    {C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128}};

size_t T_get_dtype_size(enum DataType dtype) {
  switch (dtype) {
  case I8:
    return sizeof(int8_t);
  case I16:
    return sizeof(int16_t);
  case I32:
    return sizeof(int32_t);
  case I64:
    return sizeof(int64_t);

  case U8:
    return sizeof(uint8_t);
  case U16:
    return sizeof(uint16_t);
  case U32:
    return sizeof(uint32_t);
  case U64:
    return sizeof(uint64_t);

  case F32:
    return sizeof(float);
  case F64:
    return sizeof(double);

  case C64:
    return sizeof(Complex64);
  case C128:
    return sizeof(Complex128);

  default:
    return 0;
  }
}
bool T_is_dtype_int(const enum DataType type) {
  return (type >= I8 && type <= U64);
}
bool T_is_dtype_sint(const enum DataType type) {
  return (type >= I8 && type < U8);
}
bool T_is_dtype_float(const enum DataType type) {
  return (type == F32 || type == F64);
}
bool T_is_dtype_uint(const enum DataType type) {
  return (type >= U8 && type < F32);
}
bool T_is_dtype_complex(const enum DataType type) {
  return (type == C64 || type == C128);
}

bool T_is_dtype_valid(const enum DataType type) {
  return (type >= I8 && type <= C128);
}

Ten_Status T_get_implied_dtype(const enum DataType a, const enum DataType b,
                               enum DataType *out) {
  if (out == NULL)
    return (Ten_Status){.stype = TT_ERR_INVALID_ARG,
                        .msg = "Output datatype pointer is NULL"};

  if (a < I8 || a > F64)
    return (Ten_Status){.stype = TT_ERR_DTYPE_UNSUPPORTED,
                        .msg = "Datatype of variable 'a' is invalid"};

  if (b < I8 || b > F64)
    return (Ten_Status){.stype = TT_ERR_DTYPE_UNSUPPORTED,
                        .msg = "Datatype of variable 'b' is invalid"};

  *out = promotion_table[a][b];

  return (Ten_Status){.stype = TT_OK, .msg = ""};
}

// TODO:to make a dtype overflow checker
// wrapper for uint*_t
