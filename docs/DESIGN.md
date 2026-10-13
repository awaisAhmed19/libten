# libten — Engineering Design

**Status:** Initial design · **Version:** 0.1.0 · **Core language:** C11

## Goals

Build a reusable tensor foundation for ML and general numerical work, independent of TinyTTS. It should support arbitrary-rank tensors, real and complex dtypes, explicit memory/device behavior, elementwise operations, views, matrix multiplication, reductions, and later ML primitives.

The CPU backend is always available and acts as the reference implementation. CUDA is detected automatically and compiled when a usable CUDA toolchain/toolkit and CUDA source files are available. Otherwise CMake builds CPU-only. Build-time fallback does not automatically imply runtime fallback when a compiled CUDA backend has no usable GPU or driver.

## Non-goals for the initial core

No model zoo, audio pipeline, TinyTTS-specific code, mandatory autograd engine, distributed execution, graph compiler, kernel fusion, or hidden dtype/device conversions. Keep higher-level ML frameworks separate.

## Layers

1. **Public API — `include/ten/`:** stable C-compatible declarations.
2. **Core — `src/core/`:** validation, dtype and shape rules, tensor metadata, storage lifetime, views, and dispatch.
3. **Backends — `src/cpu/`, `src/cuda/`:** allocations, transfers, and numerical kernels.
4. **Verification — `tests/`, `examples/`:** correctness, safety, parity, and smoke tests.

Public headers must not expose private storage structs, CUDA runtime types, or backend tables. CUDA `.cu` files may use CUDA C++; functions crossing into C core code must have C linkage and C-compatible signatures.

## Tensor representation

A logical tensor is described by dtype, rank, shape, element strides, storage offset in elements, storage reference, device/backend identity, and any needed flags. Prefer an opaque public tensor handle so internal representation can evolve.

Storage owns an allocation and enough metadata to release it safely: allocation size, memory/backend identity, and lifetime state. Views share storage and retain it; destroying one view must not invalidate another. Never assume a CUDA pointer is host-dereferenceable.

Contiguous row-major layout is the initial canonical layout. Strides allow transpose, permutation, slicing, and other views without copying. Broadcast views may use zero strides, but writes to aliased broadcast locations must be prohibited or precisely specified. Negative strides are deferred.

## Dtypes

Real and complex floating-point are core requirements. Proposed types:
- `F32`, `F64`
- `C64` (two F32 components), `C128` (two F64 components)
- `I32`, `I64`, `U8`

Finalize the exact first-release dtype set and integer overflow semantics before implementing dtype-specific kernels. Do not rely on undocumented compiler-specific complex layouts for a public ABI.

## Operation contract

Every operation documents accepted shapes/dtypes, output shape/dtype, broadcasting, ownership/allocation behavior, aliasing, device support, errors, and numerical tolerances. Initial groups: lifecycle and metadata; fill/copy/cast/contiguous; views; elementwise arithmetic; complex operations; rank-2 matmul then batched matmul; sum/mean reductions. No implicit dtype promotion or hidden device transfers initially.

## Error and safety policy

Public functions return status codes; library code does not terminate the process. Distinguish invalid arguments/shapes, unsupported operations/dtypes, overflow, allocation failure, unavailable devices, and backend failures. All element counts, strides, offsets, and byte-size calculations use checked arithmetic before pointer arithmetic or allocation.

## Backend behavior

- **CPU:** always built; simple reference kernels prioritize correctness.
- **CUDA:** compiled only when detected and `.cu` sources exist; report the actual configured backend.
- **Dispatch:** validate requests before dispatch where possible. Keep backend abstraction small until multiple real backends require more.
- **Runtime availability:** separately define and test behavior when CUDA was compiled but no usable GPU/driver exists. Never silently transfer tensors.

## Testing

Test ordinary results, scalar/empty/singleton cases where supported, non-contiguous views, invalid inputs, overflow, dtype behavior, aliasing, CPU reference results, and CPU/CUDA parity with dtype-appropriate tolerances. Use Debug/Release builds and ASan/UBSan for CPU code where supported.

## TinyTTS integration

TinyTTS consumes public headers and links to the installed/exported library target. Do not copy `libten` source into TinyTTS or put application-specific audio/model logic here.
