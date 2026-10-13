# libten — Architectural Decisions

Record changes here whenever a choice affects compatibility, memory semantics, numerical results, or backend behavior.

## D001 — Standalone library
**Decision:** `libten` is independent of TinyTTS.  
**Reason:** numerical primitives should be reusable.  
**Consequence:** TinyTTS consumes the public API; application-specific audio/model code stays outside this repository.

## D002 — C11 core and public API
**Decision:** core and public API use C11. CUDA `.cu` files may use CUDA C++ for kernels.  
**Consequence:** no C++ classes/templates/types in public headers; cross-boundary functions use C linkage and C-compatible signatures.

## D003 — Automatic CUDA detection and CPU fallback
**Decision:** CPU is always built. CUDA is compiled when compiler/toolkit and CUDA sources are available; otherwise build CPU-only.  
**Consequence:** configuration reports the actual backend. Build-time fallback is not runtime fallback.

## D004 — CPU is the reference backend
CPU implementations define baseline semantics and provide an independent correctness reference for CUDA. Prefer clarity before optimization.

## D005 — Tensor metadata is separate from storage
Views share storage. Storage lifetime must outlive every referencing tensor/view; reference counting is the initial candidate.

## D006 — Element-based strides and offsets
Shapes, strides, and storage offsets use element units; allocation sizes use bytes. Every conversion and arithmetic step is checked.

## D007 — Complex arithmetic is a core requirement
Real and complex dtypes, operations, tests, and CPU/CUDA parity are part of the intended library, not a speculative add-on. Finalize the initial dtype set before kernels.

## D008 — No implicit dtype promotion initially
Unsupported dtype combinations return documented errors. Casting is explicit.

## D009 — No silent device transfers
Device movement is explicit. Operations validate device compatibility and report unsupported combinations.

## D010 — Hide backend internals
Public headers do not expose CUDA handles, internal storage structures, or backend function tables.

## D011 — Small composable primitives first
Build lifecycle, views, data movement, elementwise math, matmul, and reductions before advanced framework features. CNN/RNN/transformer-specific systems should be composed above the core.

## D012 — Consistent CMake identity
Use target `ten` and alias `libten::ten`; make installed export namespace, package config, and consumer documentation consistent.

## Open decisions before kernels
1. Exact initial dtype set and integer overflow semantics.
2. Opaque public tensor handle (preferred for ABI stability) versus exposed metadata.
3. Storage ownership and output allocation rules.
4. Empty tensor/reduction behavior.
5. Negative-stride support (initially defer).
6. In-place operations and aliasing rules.
7. Status codes and diagnostic message lifetime/threading.
8. Runtime behavior when CUDA is compiled but no GPU/driver is usable.
9. cuBLAS use for eligible matrix multiplication operations.
