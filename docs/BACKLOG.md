# libten — Implementation Backlog

Mark a task complete only when code and acceptance criteria are verified.

## M0 — Build and API foundation

### T01 Build layout
- [ ] Verify C11 configure/build.
- [ ] Verify automatic CUDA detection and CPU fallback.
- [ ] Verify CUDA sources are included only when available.
- [ ] Align target, install/export namespace, package config, and consumer instructions.
**Done when:** CPU-only builds work without CUDA; CUDA build works on a configured machine.

### T02 Status and dtype definitions
- [ ] Define status enum and status-string conversion.
- [ ] Define dtype enum and metadata queries.
- [ ] Finalize real, complex, and integer dtype set.
- [ ] Define integer overflow behavior.
**Done when:** unit tests cover every supported dtype and status.

### T03 Checked arithmetic and shape utilities
- [ ] Checked integer add/multiply and byte-size calculation.
- [ ] Shape validation and element counts.
- [ ] Contiguous stride calculation.
- [ ] Scalar and zero-sized-axis semantics.
**Done when:** overflow and boundary tests pass without undefined behavior.

## M1 — Storage and tensor lifecycle

### T04 Storage
- [ ] Private storage metadata.
- [ ] CPU allocation/release.
- [ ] Track allocation size, backend/device, and lifetime.
- [ ] Implement reference counting or equivalent.
**Done when:** allocations are released exactly once and views retain storage safely.

### T05 Tensor lifecycle
- [ ] Create/destroy and metadata queries.
- [ ] Validate dtype, shape, and allocation sizes.
**Done when:** invalid inputs fail safely; lifecycle tests pass under sanitizers.

### T06 Views
- [ ] Transpose/permute, slice, squeeze, unsqueeze.
- [ ] Reshape when representable as a view.
- [ ] Contiguous query/materialization.
- [ ] Document aliasing and mutability.
**Done when:** view indexing matches an independent reference and stays within storage.

### T07 Safety tests
- [ ] Invalid shape/index, overflow, and allocation-failure tests.
- [ ] Randomized shape/stride/view tests.
- [ ] ASan/UBSan instructions.
**Done when:** CPU tests pass with sanitizers on supported compilers.

## M2 — CPU numerical core

### T08 Data movement
- [ ] Fill/copy, explicit cast, contiguous materialization.
- [ ] Define host data access semantics.
**Done when:** dtype and non-contiguous cases are tested.

### T09 Elementwise math
- [ ] Add/subtract/multiply/divide and selected unary operations.
- [ ] Broadcasting.
- [ ] Real and complex kernels.
**Done when:** reference tests cover broadcasting and error policies.

### T10 Matrix multiplication
- [ ] Rank-2 CPU reference matmul.
- [ ] Supported real and complex dtypes.
- [ ] Define accumulator/output dtype.
- [ ] Batched matmul after rank-2 semantics stabilize.
**Done when:** rectangular and edge cases match a simple reference within tolerance.

### T11 Reductions
- [ ] Sum and mean.
- [ ] Axis, keepdims, empty reduction, accumulator dtype.
- [ ] Complex reduction where mathematically appropriate.
**Done when:** supported axes/dtypes and edge cases are tested.

## M3 — CUDA backend

### T12 CUDA storage and transfers
- [ ] Device allocation/release.
- [ ] Explicit host/device copies.
- [ ] Error and resource-lifetime handling.
**Done when:** transfer tests pass on a usable GPU.

### T13 CUDA kernels
- [ ] Elementwise real/complex operations.
- [ ] Reductions and matmul; use cuBLAS where appropriate.
- [ ] Launch error and synchronization policy.
**Done when:** each CUDA operation has CPU parity tests with documented tolerances.

### T14 Dispatch/runtime behavior
- [ ] Validate device compatibility before dispatch.
- [ ] Define no-device/no-driver behavior.
- [ ] Prohibit silent transfers.
**Done when:** unsupported paths return documented errors without invalid pointer access.

## M4 — ML primitives

### T15 Neural-network building blocks
- [ ] ReLU, sigmoid, tanh.
- [ ] Numerically stable softmax.
- [ ] Bias/broadcast and normalization primitives.
- [ ] Convolution-related operations after layout rules stabilize.
**Done when:** each primitive has mathematical reference and edge-case tests.

### T16 Gradient-readiness review
- [ ] Review operation interfaces for future backward operations.
- [ ] Document potential saved-intermediate metadata.
- [ ] Keep autograd out of the core until contracts stabilize.
**Done when:** a short extension note exists; no premature autograd implementation.

## M5 — Packaging and integration

### T17 Install and examples
- [ ] Install headers and library.
- [ ] Export consistent CMake target/package.
- [ ] Add `examples/basic_tensor.c`.
- [ ] Document CPU-only and CUDA build modes.
**Done when:** a separate sample project links to the installed package.

### T18 TinyTTS integration
- [ ] Consume public headers and library target.
- [ ] Remove duplicate implementation only after parity.
- [ ] Keep TinyTTS-specific behavior outside libten.
**Done when:** TinyTTS builds without compiling libten sources directly.

## M6 — Hardening

### T19 CI
- [ ] CPU-only CI.
- [ ] CUDA CI where a suitable runner exists.
- [ ] Debug/Release, sanitizers, install/consumer smoke test.
**Done when:** regressions are caught automatically and backend status is clear.

## Deferred until justified

Autograd engine, FP16/BF16, quantized types, FFT, kernel fusion, graph execution, memory pools, multi-GPU/distributed execution, dynamic plugins, and advanced CPU vectorization. Promote only when a real workload justifies the complexity.
