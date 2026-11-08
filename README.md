# libten

**A small, standalone tensor library written in C11, with a CPU backend and optional CUDA acceleration.**

`libten` is designed as a reusable foundation for tensor operations and machine-learning workloads. It is intended to remain independent of any particular application, including TinyTTS.

> **Project status:** Early development. Public APIs, datatype-promotion rules, and backend coverage may change before the first stable release.

## Goals

- Provide a straightforward C11 public API.
- Represent tensors with explicit shape, dtype, device, and storage metadata.
- Support signed and unsigned fixed-width integers, real floating-point values, and complex values.
- Provide a CPU implementation as the baseline.
- Support optional CUDA acceleration.
- Keep backend-specific implementation details separate from public headers.
- Report invalid arguments, shape errors, unsupported operations, allocation failures, and arithmetic overflow through explicit status values.

## Planned datatype set

| Category | Datatypes |
|---|---|
| Signed integers | `I8`, `I16`, `I32`, `I64` |
| Unsigned integers | `U8`, `U16`, `U32`, `U64` |
| Real floating point | `F32`, `F64` |
| Complex floating point | `C64`, `C128` |

The datatype set and promotion behavior are still under development.

See [`docs/DESIGN.md`](docs/DESIGN.md) and [`docs/MATHEMATICS.md`](docs/MATHEMATICS.md) for design decisions and mathematical conventions, if available in your checkout.

## Repository layout

```text
libten/
├── include/
│   └── ten/             Public headers
├── src/
│   ├── core/            Backend-independent tensor logic
│   ├── internal/        Internal interfaces and backend dispatch
│   └── backend/
│       ├── cpu/         CPU implementation
│       └── cuda/        CUDA implementation
├── tests/               Unit and backend tests
├── examples/            Usage examples
├── docs/                Design notes, mathematics, and backlog
├── CMakeLists.txt
└── build.sh
```

Some directories and files may be incomplete while the project is being developed.

## Requirements

### CPU build

- C compiler with C11 support
- CMake
- Make or Ninja (depending on the configured CMake generator)

### CUDA build

All CPU requirements, plus:

- NVIDIA CUDA Toolkit
- CUDA compiler (`nvcc`)
- A compatible NVIDIA GPU and driver to run CUDA tests and workloads

CUDA support is optional. A CPU build does not require the CUDA Toolkit.

## Building

The project includes a helper script for configuring, building, testing, and cleaning the project.

First, make the script executable:

```bash
chmod +x build.sh
```

### CPU build

The default configuration builds the CPU backend in Debug mode:

```bash
./build.sh
```

You can also specify the configuration explicitly:

```bash
./build.sh cpu debug
./build.sh cpu release
./build.sh cpu relwithdebinfo
```

### CUDA build

CUDA is enabled explicitly. The CUDA configuration includes both CPU and CUDA backends.

```bash
./build.sh cuda debug
```

For a release build:

```bash
./build.sh cuda release
```

The CUDA Toolkit and compiler must be available to CMake. The script does not silently switch between CPU and CUDA configurations.

### Build directories

CPU and CUDA configurations use separate build directories:

```text
build/
├── cpu/
│   ├── debug/
│   ├── release/
│   └── relwithdebinfo/
└── cuda/
    ├── debug/
    ├── release/
    └── relwithdebinfo/
```

This keeps the two configurations independent and avoids accidentally reusing incompatible build settings.

## Running tests

Run the test suite using the build script.

### CPU tests

```bash
./build.sh test cpu debug
```

### CUDA tests

```bash
./build.sh test cuda debug
```

The CUDA test suite includes tests for CUDA memory management and data transfers. Tests requiring a usable CUDA device are skipped when one is unavailable.

You can also invoke CTest directly:

```bash
ctest --test-dir build/cpu/debug --output-on-failure
```

For CUDA:

```bash
ctest --test-dir build/cuda/debug --output-on-failure
```

## Cleaning

Remove generated build directories:

```bash
./build.sh clean
```

To clean and run a fresh CPU test build:

```bash
./build.sh clean
./build.sh test cpu debug
```

## Build script reference

| Command | Description |
|---|---|
| `./build.sh` | Default CPU Debug build |
| `./build.sh cpu debug` | CPU Debug build |
| `./build.sh cpu release` | CPU Release build |
| `./build.sh cuda debug` | CPU + CUDA Debug build |
| `./build.sh cuda release` | CPU + CUDA Release build |
| `./build.sh test cpu debug` | Build and run CPU tests |
| `./build.sh test cuda debug` | Build and run CUDA tests |
| `./build.sh clean` | Remove build directories |
| `./build.sh help` | Display available commands |

Additional build settings can be configured through environment variables. For example:

```bash
CUDA_ARCH=75 ./build.sh cuda release
BUILD_JOBS=8 ./build.sh cpu debug
```

Check `build.sh` for the currently supported variables and defaults.

## Design principles

- **Explicit behavior:** Dtype promotion, integer overflow, device transfers, and unsupported operations should have documented contracts.
- **Safe size calculations:** Check tensor element counts, allocation sizes, strides, and offsets for overflow before using them.
- **Backend consistency:** CPU and CUDA implementations should follow the same public operation semantics.
- **Small public API:** Expose stable types and functions, not internal kernel or dispatch details.
- **Incremental implementation:** Validate core metadata and CPU behavior before expanding operation and CUDA coverage.

## Development

1. Read the design documents in `docs/`.
2. Check `docs/BACKLOG.md` for current tasks, if available.
3. Add tests for new behavior and edge cases.
4. Keep the public C API compatible with C11.
5. Document changes to datatype semantics or CPU/CUDA behavior.
6. Run the relevant test suite before submitting changes.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).

Replace the copyright-holder placeholder in `LICENSE` before publishing a release.
