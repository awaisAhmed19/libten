# libten

**A small, standalone tensor library in C11 with a CPU backend and optional CUDA acceleration.**

`libten` is being designed as a reusable foundation for tensor operations and machine-learning workloads. It is intended to remain independent of any one application, including TinyTTS.

> **Project status:** Early development. Public APIs, datatype-promotion rules, and backend coverage may change before the first stable release.

## Goals

- Provide a straightforward C11 public API.
- Represent tensors with explicit shape, dtype, device, and storage metadata.
- Support signed and unsigned fixed-width integers, real floating-point values, and complex values.
- Provide a CPU implementation as the baseline.
- Detect CUDA support at build time and enable CUDA components when the compiler, toolkit, and implementation sources are available.
- Keep backend-specific implementation details separate from public headers.
- Report invalid arguments, shape errors, unsupported operations, allocation failures, and arithmetic overflow through explicit status values.

## Planned datatype set

| Category               | Datatypes                 |
| ---------------------- | ------------------------- |
| Signed integers        | `I8`, `I16`, `I32`, `I64` |
| Unsigned integers      | `U8`, `U16`, `U32`, `U64` |
| Real floating point    | `F32`, `F64`              |
| Complex floating point | `C64`, `C128`             |

The datatype set and promotion behavior are still under development. See [`docs/DESIGN.md`](docs/DESIGN.md) and [`docs/MATHEMATICS.md`](docs/MATHEMATICS.md) if those documents are present in your checkout.

## Repository layout

```text
include/ten/   Public headers
src/core/      Backend-independent tensor and datatype logic
src/cpu/       CPU backend
src/cuda/      CUDA backend and kernels
tests/         Unit and backend tests
examples/      Small usage examples
docs/          Design notes, mathematics, decisions, and backlog
```

Some directories and files may be incomplete while the project is being developed.

## Build

The project uses CMake. A typical local build looks like:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
```

Run tests if the test target is available:

```sh
ctest --test-dir build --output-on-failure
```

CUDA is intended to be detected automatically. If CUDA is unavailable, the build should use the CPU backend. The exact options and target names may change during early development; check the current `CMakeLists.txt` for the supported configuration.

## Design principles

- **Explicit behavior:** dtype promotion, integer overflow, device transfers, and unsupported operations should have documented contracts.
- **Safe size calculations:** check tensor element counts, allocation sizes, strides, and offsets for overflow before using them.
- **Backend consistency:** CPU and CUDA implementations should follow the same public operation semantics.
- **Small public API:** expose stable types and functions, not internal kernel or dispatch details.
- **Incremental implementation:** validate core metadata and CPU behavior before expanding operation and CUDA coverage.

## Development

1. Read the design documents in `docs/`.
2. Check `docs/BACKLOG.md` for current tasks, if available.
3. Add tests for new behavior and edge cases.
4. Keep the public C API compatible with C11.
5. Document any change to dtype semantics or CPU/CUDA behavior.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).

Replace the copyright-holder placeholder in `LICENSE` before publishing a release.
