# libten tensor tests

These are temporary white-box tests for the current tensor implementation.

Why do they include `../src/core/tensor.c`? `Tensor` and its helper functions are
currently private to that implementation file, so a normal external test cannot
inspect shape, strides, storage, or `num_el`. This is useful while debugging,
but it is not the ideal long-term architecture.

## Build

Run these commands from the repository root. Each test is built separately and
links `ten_dtype.c`; do not link the `ten` library as well, because each test
includes `tensor.c` directly.

```sh
cc -std=c11 -Wall -Wextra -Wpedantic -g \
  -Iinclude tests/tensor_helpers_test.c src/core/ten_dtype.c \
  -o /tmp/tensor_helpers_test

cc -std=c11 -Wall -Wextra -Wpedantic -g \
  -Iinclude tests/tensor_constructor_test.c src/core/ten_dtype.c \
  -o /tmp/tensor_constructor_test
```

Run:

```sh
/tmp/tensor_helpers_test
/tmp/tensor_constructor_test
```

## Sanitizers

```sh
cc -std=c11 -Wall -Wextra -Wpedantic -g \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -Iinclude tests/tensor_helpers_test.c src/core/ten_dtype.c \
  -o /tmp/tensor_helpers_test_asan

/tmp/tensor_helpers_test_asan
```

Repeat with `tensor_constructor_test.c` after the constructor and cleanup bugs
have been fixed.

## Expected status against the code currently shown

The tests are intended to expose defects; they are not expected to pass against
the current implementation yet. In particular:

- `calculate_strides()` currently indexes out of bounds and computes incorrect
  strides.
- `T_create_tensor_by_data()` mixes allocations on `t` with calculations on
  `tmp`, reads the dtype from `t` instead of `i_dtype`, and has unsafe cleanup.
- The current destructor frees the `Tensor` object itself, so tests allocate
  the object on the heap.

Do not run the constructor success tests under an ordinary process until the
constructor's known null-pointer and out-of-bounds issues are fixed; use the
sanitizer build to diagnose them once the helper tests are repaired.

## Longer-term recommendation

Stop including `tensor.c` in test files once the API is stable. Add public
constructor/destructor declarations and metadata accessors to `ten_tensor.h`,
then write consumer-style tests that link against `libten::ten`.
