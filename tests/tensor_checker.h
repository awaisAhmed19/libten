#ifndef TEN_TEST_CHECKER_H
#define TEN_TEST_CHECKER_H

#include <stdio.h>

#include "../include/ten/ten_status.h"

/* Shared by every white-box test; each test translation unit gets its own copy.
 */
static int tests_run;

#define CPU_DEVICE ((Device){.devtype = CPU, .id = 0})
#define CHECK(expr)                                                            \
  do {                                                                         \
    ++tests_run;                                                               \
    if (!(expr)) {                                                             \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr);          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHECK_STATUS(expr, expected)                                           \
  do {                                                                         \
    Ten_Status status_check = (expr);                                          \
    ++tests_run;                                                               \
    if (status_check.stype != (expected)) {                                    \
      fprintf(stderr, "FAIL %s:%d: expected status %d, got %d (%s)\n",         \
              __FILE__, __LINE__, (expected), status_check.stype,              \
              status_check.msg ? status_check.msg : "no message");             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#endif /* TEN_TEST_CHECKER_H */
