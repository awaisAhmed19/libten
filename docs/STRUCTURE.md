# libten — Directory and File Responsibilities

Each file has one primary responsibility. The proposed files below are a map, not a demand to create empty placeholders.

## Repository map

```text
libten/
├── CMakeLists.txt
├── build.sh
├── README.md
├── LICENSE
├── cmake/
│   └── tenConfig.cmake.in
├── include/ten/
│   ├── ten.h
│   ├── ten_types.h
│   ├── ten_tensor.h
│   └── ten_ops.h
├── src/
│   ├── core/
│   ├── cpu/
│   └── cuda/
├── tests/
├── examples/
└── docs/
```

## Root files

- **`CMakeLists.txt`** — project/language setup, CUDA detection, source lists, library target, tests, examples, installation/export. CMake is authoritative for backend selection.
- **`build.sh`** — convenience wrapper for configure/build/test/install/clean. It should not contradict CMake's CUDA detection.
- **`README.md`** — project purpose, features, prerequisites, quick start, and links to deeper docs.
- **`LICENSE`** — license terms.

## `cmake/`

- **`tenConfig.cmake.in`** — installed package configuration template. Imports exported targets and finds CUDA dependencies only if this build actually includes CUDA. Package name, exported target filename, namespace, and consumer instructions must agree.

## `include/ten/` — public API

Headers must compile as C11 and be includable from C++ with `extern "C"` guards where needed. Do not expose private storage layouts, backend tables, or CUDA runtime types.

- **`ten.h`** — umbrella header including the commonly used public headers; no duplicate declarations.
- **`ten_types.h`** — public status codes, dtype enum, public device identifiers/constants, and shared small value types. No tensor lifecycle or arithmetic implementations.
- **`ten_tensor.h`** — create/destroy, metadata queries, views, layout operations, explicit transfers, contiguous materialization, and documented host-data access if supported.
- **`ten_ops.h`** — elementwise arithmetic, unary math, matmul, reductions, complex operations, and later ML primitives. Every function documents dtypes, shape rules, ownership, aliasing, device support, and errors.

## `src/core/` — backend-independent logic

Core owns invariants and orchestration, not CPU loops or CUDA kernels. Introduce files when code exists for them:

- **`status.c`** — status-to-string and common error helpers.
- **`dtype.c`** — dtype sizes, alignment, categories, and support queries.
- **`checked.c`** — overflow-safe integer arithmetic and allocation-size calculations.
- **`shape.c`** — shape validation, element count, contiguous strides, and index/offset calculations.
- **`storage.c`** — private storage metadata and lifetime management.
- **`tensor.c`** — tensor creation/destruction and metadata queries.
- **`views.c`** — view construction and shape/stride transformations.
- **`dispatch.c`** — routes validated requests to CPU/CUDA implementations.
- **`main.c`** — a library should not normally have an application `main()`. Rename this generic filename to the responsibility its current functions actually implement.

## `src/cpu/` — CPU backend

Straightforward host-memory operations and reference kernels. Possible files:
- **`cpu_backend.c`** — backend capability/wiring if a function table is used.
- **`elementwise.c`** — real and complex elementwise kernels.
- **`matmul.c`** — CPU reference matrix multiplication.
- **`reduce.c`** — reductions.
- **`copy.c`** — CPU copy/cast/contiguous operations when useful to separate.

Prioritize correctness and readability; optimize after profiling.

## `src/cuda/` — CUDA backend

CUDA `.cu` sources and CUDA-specific resource handling:
- **`cuda_backend.cu`** — backend capability/wiring.
- **`cuda_memory.cu`** — device allocation/release/transfers.
- **`elementwise.cu`** — CUDA elementwise kernels.
- **`matmul.cu`** — custom kernels and/or cuBLAS integration.
- **`reduce.cu`** — reductions.
- **`copy.cu`** — device copy/cast/contiguous kernels where needed.

Functions called from C core code must use C linkage and C-compatible signatures. CUDA types remain private.

## `tests/`

Prefer public-API tests. Suggested files:
- `test_checked.c` — overflow-safe arithmetic.
- `test_dtype.c` — dtype metadata.
- `test_shape.c` — shapes, counts, strides, offsets.
- `test_tensor.c` — lifecycle and metadata.
- `test_views.c` — aliasing, slicing, transpose, reshape.
- `test_ops.c` — arithmetic, broadcasting, matmul, reductions.
- `test_complex.c` — complex operations.
- `test_cuda.c` — availability, transfers, kernels, CPU/CUDA parity.

Test discovery must match CMake patterns. A CUDA compiler being installed does not guarantee a GPU is available at test runtime.

## `examples/`

Small complete programs using only the public API.
- **`basic_tensor.c`** — first smoke test for tensor creation, operation, result access, and cleanup.

## `docs/`

- `README.md` — documentation index and maintenance.
- `DESIGN.md` — architecture and system boundaries.
- `MATHEMATICS.md` — formulas and numerical invariants.
- `DECISIONS.md` — recorded decisions and open questions.
- `BACKLOG.md` — milestones, tasks, and acceptance criteria.
- `STRUCTURE.md` — this repository map.

## Dependency direction

`application -> public API -> core -> backend`

Tests and examples normally use the public API. CPU and CUDA implement the same operation contract. The public API must not depend on backend internals.

## Rules for growth

1. One primary responsibility per source file.
2. Public declarations in `include/ten/`; private declarations under `src/`.
3. Do not expose internal structs unless ABI commitment is intentional.
4. Do not duplicate shape/dtype semantics separately in CPU and CUDA.
5. Every operation gets a mathematical definition and tests before optimization.
6. Do not create empty files for hypothetical features.
