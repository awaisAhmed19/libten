# libten — Tensor/Core Refactoring Sprint Backlog

**Sprint goal:** Refactor the current tensor and view implementation into small, clearly owned core modules, establish a stable public tensor API, and create the CPU/CUDA execution seams needed to implement the upcoming data-movement and numerical backlogs.

**Scope basis:** `src(8).zip`, especially `include/ten/ten_tensor.h`, `src/core/ten_tensor.c`, `src/core/ten_view.c`, `src/internal/ten_tensor.h`, `docs/STRUCTURE.md`, `docs/BACKLOG.md`, and `docs/DECISIONS.md`.

**Sprint boundary:** This sprint is a refactor and architecture-preparation sprint. It does not attempt to complete CUDA kernels, casting, elementwise math, matmul, or reductions. It must leave the repository ready to begin those implementation backlogs without another large structural rewrite.

**Completion rule:** A task is complete only when its code, tests, public/internal API boundaries, and acceptance criteria are verified. Moving code without preserving behaviour and tests does not count as completion.

---

## Sprint outcomes

By the end of the sprint:

- [ ] `ten_tensor.h` contains only intentional public tensor APIs, grouped and documented consistently.
- [ ] Internal storage, allocation, validation, and low-level copy helpers are removed from the public header.
- [ ] Tensor lifecycle, storage, shape/layout, views, and data movement have separate implementation ownership.
- [ ] Metadata-only view operations do not depend on CPU memory access and do not reject CUDA tensors solely because they are CUDA tensors.
- [ ] CPU-only operations are explicitly identified as CPU-only; no function silently treats a CUDA pointer as host memory.
- [ ] Core code has a backend dispatch boundary for data movement, even if CUDA implementations are not yet present.
- [ ] Existing CPU behaviour is preserved unless a bug or ambiguous contract is explicitly corrected and tested.
- [ ] The CPU test suite passes after every extraction; sanitizer runs pass where supported.
- [ ] The next backlog tasks can add CPU and CUDA implementations without putting backend-specific work into `ten_tensor.c` or `ten_view.c`.

## Target source layout

The exact filenames may be adjusted to match the repository's build conventions, but responsibilities must remain separate.

```text
include/ten/
    ten_tensor.h          # public tensor API
    ten_ops.h             # public numerical operations

src/core/
    tensor.c              # tensor construction, destruction, metadata, clone
    storage.c              # storage metadata and lifetime management
    shape.c                # shape validation, numel, contiguous stride calculation
    tensor_layout.c        # bounds, offsets, contiguity, reshape-stride reasoning
    views.c                # metadata-only view transformations
    data_movement.c        # public fill/copy/cast/materialize validation + dispatch
    dispatch.c             # backend selection and dispatch boundary

src/cpu/
    cpu_backend.c          # CPU backend registration/implementation table
    copy.c                 # CPU fill/copy/cast/materialize execution primitives

src/cuda/
    cuda_backend.cu        # CUDA backend registration/implementation table
    cuda_memory.cu         # CUDA allocation and release
    copy.cu                # CUDA fill/copy/cast/materialize kernels

src/internal/
    ten_tensor.h           # private Tensor/TensorStorage definitions
    backend.h              # private backend operation contract
    tensor_internal.h      # private tensor/storage helper declarations, if needed
```

Do not create empty files merely to match this tree. Add a module when its first real implementation lands. Keep CUDA-specific headers and types out of public headers and ordinary C core files.

---

# Sprint backlog

## R01 — Public tensor header/API cleanup

**Priority:** P0  
**Depends on:** None  
**Purpose:** Make the public contract coherent before moving implementations.

### Work

- [ ] Reorganize `include/ten/ten_tensor.h` into these sections:
  1. Opaque public types and constants.
  2. Construction.
  3. Lifetime.
  4. Metadata queries.
  5. Layout queries.
  6. Host data access.
  7. Data movement.
  8. View/alias operations.
  9. Reshape operations.
- [ ] Review every public declaration against an implementation and a test.
- [ ] Remove private/internal declarations from the public header:
  - `T_storage_retain`
  - `T_storage_release`
  - `T_create_tensor_storage`
  - `T_create_empty_storage`
  - `T_allocate_shape`
  - `T_allocate_stride`
  - `T_calculate_strides`
  - `T_validate_new_tensor`
  - `T_validate_tensor_bounds`
  - `T_copy_tensor_data`
  - `T_allocate_tensor_data`, unless a real public use case is established.
- [ ] Decide whether `T_is_ranks_valid` and `T_is_shape_input_valid` are intentionally public utilities. If they are implementation validation helpers, move them to an internal header; if retained publicly, document them as supported API.
- [ ] Resolve the `TensorView` public-type problem:
  - `TensorView` is opaque in the public header but its current structure is defined internally.
  - `T_create_tensor_view(TensorView *, ...)` is not practically usable by consumers without a public allocation/lifecycle API.
  - Prefer removing this low-level API from the public surface if the public `Tensor` view operations cover the supported use cases. Otherwise design a complete opaque-handle lifecycle, not a partially exposed type.
- [ ] Keep public declarations backend-neutral. Do not expose `cudaStream_t`, CUDA runtime types, CPU-specific storage pointers, or backend function tables.
- [ ] Ensure every public function has a concise contract: argument requirements, ownership, aliasing, device behaviour, output-on-failure behaviour, and error/status semantics.
- [ ] Ensure the header remains valid C and usable from C++ through its existing `extern "C"` boundary.

### Acceptance criteria

- [ ] Every declaration in `ten_tensor.h` is classified as public API and has a definition, or is explicitly marked as a planned API and removed until implemented.
- [ ] No private storage-management or raw allocation helper remains public without an explicit design reason.
- [ ] No public API requires consumers to know the private layout of `Tensor` or `TensorStorage`.
- [ ] A small external C translation unit can include `ten/ten.h` and compile using only public declarations.
- [ ] Public header compiles as both C and C++.

---

## R02 — Repair existing API/implementation mismatches

**Priority:** P0  
**Depends on:** R01  
**Purpose:** Fix known correctness and completeness problems before extraction, so the refactor does not preserve defects.

### Work

- [ ] Fix `T_get_shape()` so it copies into caller-provided memory rather than replacing the local output pointer.
- [ ] Enforce `capacity >= rank`; define and test the scalar (`rank == 0`) behaviour.
- [ ] Implement `T_get_dim()` if it remains public; otherwise remove the declaration and any stale forward declaration.
- [ ] Resolve the declared-but-unimplemented `T_tensor_slice()`:
  - Implement and test its documented view semantics, or
  - Remove it from the public API and backlog it explicitly.
- [ ] Resolve `T_data()` and `T_const_data()`:
  - Implement the documented CPU-only borrowed-pointer semantics, or
  - Remove them until their contract is complete.
  - A CUDA device pointer must never be returned as though it were a host pointer.
  - Define what happens for non-contiguous tensors. A raw pointer to the first logical element does not make the whole tensor a contiguous flat array.
- [ ] Correct `T_get_nbytes()` to return logical tensor bytes as documented, not the full backing-storage allocation size. If allocation size is needed, add a separately named internal or public query with unambiguous semantics.
- [ ] Review `T_clone()` semantics. It should copy the logical tensor contents into independent storage; it must not unexpectedly duplicate unrelated elements from a larger backing allocation.
- [ ] Review `T_tensor_reshape()` semantics and document whether it always creates independent contiguous storage.
- [ ] Review every view constructor for null checks before dereferencing `src`, `src->storage`, shape, or strides.
- [ ] Review failure paths after temporary allocations; destroy all partially allocated metadata/storage before returning an error.
- [ ] Review destination-state requirements consistently: either require an empty destination for constructors/materializers or define safe replacement semantics. Do not mix these silently.
- [ ] Review status codes/messages for invalid argument, invalid state, unsupported device, shape mismatch, and allocation failure.

### Acceptance criteria

- [ ] Regression tests cover every corrected public API.
- [ ] Invalid inputs fail without crashes, leaks, or partially initialized destination tensors.
- [ ] Existing tests still pass, except tests intentionally updated for corrected documented behaviour.
- [ ] ASan/UBSan runs show no new memory errors on supported toolchains.

---

## R03 — Extract storage ownership into `core/storage.c`

**Priority:** P0  
**Depends on:** R01, R02  
**Purpose:** Make storage lifetime a distinct concern and prepare storage to be allocated by different devices.

### Work

- [ ] Move storage metadata lifecycle operations out of `ten_tensor.c`:
  - storage creation/initialization,
  - retain,
  - release,
  - reference-count validation,
  - storage metadata validation.
- [ ] Keep `TensorStorage` definition private in `src/internal/ten_tensor.h` or a more focused internal storage header.
- [ ] Separate storage metadata from the physical allocation mechanism:
  - core owns dtype, device identity, byte capacity, and ownership/refcount metadata;
  - backend owns allocation and release of the actual memory.
- [ ] Introduce or confirm a backend-neutral allocation/release interface. Core must not call `malloc/free` on CUDA-owned pointers.
- [ ] Define the zero-byte allocation invariant. The current internal comment says `data` may be NULL only when `n_bytes == 0`; enforce this consistently.
- [ ] Define reference-count overflow/underflow behaviour.
- [ ] Keep thread-safety limitations explicit. The current `u32 ref_count` is not thread-safe; document that limitation or create a separate future task for atomic ownership. Do not imply thread safety.
- [ ] Ensure a storage release invokes the correct backend release path based on the storage device.
- [ ] Ensure storage is released exactly once when its final reference is dropped.

### Acceptance criteria

- [ ] Tensor and view tests pass after storage extraction.
- [ ] Shared-storage tests verify retain/release counts and final release.
- [ ] Zero-byte storage tests pass.
- [ ] CPU-only build does not require CUDA headers or runtime.
- [ ] Core never frees device memory using host `free()`.

---

## R04 — Extract shape and layout responsibilities

**Priority:** P0  
**Depends on:** R01, R02  
**Purpose:** Remove shape/stride and view-layout logic from tensor lifecycle code.

### `core/shape.c` — shape mathematics

- [ ] Move or consolidate:
  - rank validation,
  - shape-input validation,
  - element-count calculation,
  - contiguous stride calculation,
  - checked shape/byte-size calculations where appropriate.
- [ ] Ensure scalar (`rank == 0`) and zero-sized-axis semantics match the existing decisions and tests.
- [ ] Ensure overflow is detected before allocation or pointer arithmetic.
- [ ] Keep these helpers independent of CPU/CUDA memory.

### `core/tensor_layout.c` — layout and address reasoning

- [ ] Move or consolidate:
  - contiguity checks,
  - storage-bound validation,
  - offset/stride range calculations,
  - reshape-view stride calculation,
  - logical-index-to-storage-offset calculation if shared by multiple core operations.
- [ ] Define the relationship between `offset`, `shape`, `strides`, `num_el`, and storage capacity.
- [ ] Handle empty tensors without invalid pointer arithmetic.
- [ ] Handle scalar tensors and rank limits.
- [ ] Keep layout calculations as metadata-only operations; do not dereference tensor data.

### Acceptance criteria

- [ ] Shape and layout helpers compile without CPU/CUDA execution dependencies.
- [ ] Existing shape, checked-arithmetic, tensor and view tests pass.
- [ ] Randomized view/stride tests remain valid.
- [ ] Bounds validation proves every reachable logical element is within storage, including non-contiguous and offset views.

---

## R05 — Extract tensor lifecycle into `core/tensor.c`

**Priority:** P0  
**Depends on:** R03, R04  
**Purpose:** Keep tensor lifecycle focused on tensor metadata and ownership rather than storage internals or numerical copying.

### Work

- [ ] Keep tensor construction, destruction, metadata queries, and clone orchestration in this module.
- [ ] Move storage allocation/release details to `storage.c`.
- [ ] Move shape and stride calculations to `shape.c` / `tensor_layout.c`.
- [ ] Keep constructors transactional:
  - validate inputs,
  - allocate temporary metadata,
  - acquire storage,
  - initialize all fields,
  - publish/commit the completed tensor only on success.
- [ ] Ensure a failed constructor leaves the destination in the documented empty state.
- [ ] Make `T_destroy_tensor()` safe for a zero-initialized tensor and safe to call according to its documented lifecycle contract.
- [ ] Implement clone in terms of logical copy semantics, not by blindly copying the entire backing allocation.
- [ ] Keep `T_tensor_share_storage()` explicitly separate from deep clone.
- [ ] Remove duplicate or obsolete helpers once all call sites are migrated.

### Acceptance criteria

- [ ] Constructors, clone, share-storage, destroy, and metadata tests pass.
- [ ] Clone of a view copies only logical elements and produces independent storage.
- [ ] Sharing storage preserves alias visibility and reference lifetime.
- [ ] Failure injection or equivalent allocation-failure tests leave no leaks or half-valid tensors.

---

## R06 — Extract metadata-only view operations into `core/views.c`

**Priority:** P0  
**Depends on:** R03, R04, R05  
**Purpose:** Make view operations device-independent metadata transformations.

### Work

- [ ] Keep transpose, permute, squeeze, squeeze-dimension, unsqueeze, slice, and reshape-view operations in `views.c`.
- [ ] Make all view constructors use one shared internal helper for:
  - validating the source tensor,
  - validating the destination state,
  - allocating shape/stride metadata,
  - retaining source storage,
  - setting offset/rank/numel,
  - validating resulting storage bounds,
  - committing the destination.
- [ ] Remove CPU-only restrictions from metadata-only view operations. A valid CUDA tensor can be transposed/permuted/squeezed by changing metadata; no device kernel is required.
- [ ] Ensure views preserve the source device and dtype.
- [ ] Ensure views preserve storage aliasing and correctly retain storage.
- [ ] Ensure transpose/permute validate dimensions and permutations before modifying destination state.
- [ ] Implement or explicitly defer `T_tensor_slice()`; if implemented, specify start/end/step normalization, negative values policy, step-zero rejection, output length, and offset/stride calculations.
- [ ] Ensure reshape-view only succeeds when the requested shape can be represented by valid strides; otherwise return the documented error without silently copying.
- [ ] Keep copying reshape separate from reshape-view.
- [ ] Remove the public `TensorView` helper if the public `Tensor` view APIs make it redundant, or provide a complete public lifecycle if it is retained.

### Acceptance criteria

- [ ] View operations work on CPU tensors.
- [ ] View operations on CUDA-tagged tensors can be tested at metadata level without dereferencing CUDA memory.
- [ ] Every view retains storage and releases it correctly.
- [ ] Mutations through shared CPU aliases remain visible.
- [ ] Invalid permutations, dimensions, slices, and reshape requests fail safely.
- [ ] Randomized view tests and bounds tests pass.

---

## R07 — Define T08 data-movement contracts

**Priority:** P0  
**Depends on:** R01, R02, R03, R04  
**Purpose:** Specify semantics before writing CPU/CUDA implementations.

### Fill contract

- [ ] Distinguish constructors from in-place mutation:
  - constructors such as `T_tensor_zeros()` / `T_tensor_ones()` create new tensors;
  - `T_tensor_fill()` modifies an existing tensor.
- [ ] Decide whether existing `T_tensor_fill_zeros`, `T_tensor_fill_ones`, and `T_tensor_fill_value` names are retained, renamed, or deprecated. Their current signatures behave as constructors, despite the word “fill”.
- [ ] Define scalar representation for every supported dtype, including complex values.
- [ ] Define whether fill supports non-contiguous views. Preferred contract: fill every logical element, respecting strides.
- [ ] Define whether fill is in-place and therefore visible through aliases.
- [ ] Define supported device behaviour and error behaviour when a backend is unavailable.

### Copy contract

- [ ] Distinguish tensor-level deep copy from internal storage copy.
- [ ] Define source/destination shape and dtype requirements.
- [ ] Define same-device copy and cross-device transfer separately.
- [ ] Define aliasing/overlap behaviour.
- [ ] Define whether non-contiguous source and destination layouts are supported.
- [ ] Define whether copy requires a preallocated destination or creates one.
- [ ] Define error behaviour for incompatible devices and unsupported paths.

### Cast contract

- [ ] Define whether cast creates a new tensor or writes into a supplied destination.
- [ ] Define all supported source/destination dtype pairs.
- [ ] Define float-to-integer rounding/truncation.
- [ ] Define integer narrowing and signed/unsigned overflow behaviour.
- [ ] Define complex-to-real and real-to-complex behaviour.
- [ ] Define NaN/infinity handling when converting to integer types.
- [ ] Do not rely on accidental C casts as the specification.

### Contiguous materialization contract

- [ ] Define materialization as a new allocation containing the same logical values in contiguous layout.
- [ ] Preserve logical shape and dtype.
- [ ] Preserve the source device unless an explicit destination device is requested by a separately named transfer API.
- [ ] Ensure the result does not alias source storage.
- [ ] Define behaviour for already-contiguous inputs: still copy, or return/share? Preferred contract: materialization guarantees independent storage, so it still copies.
- [ ] Define scalar and zero-sized tensor behaviour.

### Acceptance criteria

- [ ] Contracts are recorded in public header comments and the relevant design documentation.
- [ ] CPU and CUDA implementations can be written against the same semantics.
- [ ] No contract relies on implicit host/device transfers.
- [ ] Tests can be written from the contract without referring to implementation details.

---

## R08 — Create the backend execution boundary

**Priority:** P0  
**Depends on:** R03, R07  
**Purpose:** Ensure core code validates and dispatches while CPU/CUDA modules execute device-specific work.

### Work

- [ ] Define a private backend interface in `src/internal/backend.h`.
- [ ] Keep the interface in C-compatible types where practical; isolate CUDA runtime types inside `.cu` implementation files.
- [ ] Define operation entries needed by the next backlog:
  - allocate/release,
  - fill,
  - copy,
  - cast,
  - contiguous materialization.
- [ ] Decide whether materialization is a core logical traversal plus backend element-copy primitive, or a backend operation. Document the chosen boundary and keep it consistent.
- [ ] Add CPU backend implementation/registration.
- [ ] Add a CUDA backend stub or registration only if the build system already supports conditional CUDA compilation. Do not pretend a stub is a working CUDA implementation.
- [ ] Core dispatch must validate tensor state, dtype, shape, device compatibility, and backend availability before calling an implementation.
- [ ] Unsupported CUDA requests in CPU-only builds must return a documented status, not fall through to CPU pointer handling.
- [ ] Do not add implicit transfers to make an operation “work”.
- [ ] Avoid circular dependencies: backend code may use internal metadata definitions, but core should not depend on CUDA-specific implementation details.

### Acceptance criteria

- [ ] Core data-movement wrappers contain no direct CUDA runtime calls.
- [ ] CPU backend code contains no CUDA header dependency.
- [ ] CPU-only configuration builds and tests without CUDA installed.
- [ ] CUDA-enabled configuration can compile the backend boundary when a CUDA toolchain is available.
- [ ] Missing backend implementations return explicit unsupported/not-available errors.

---

## R09 — Implement CPU data-movement primitives

**Priority:** P1  
**Depends on:** R07, R08  
**Purpose:** Move existing CPU data movement out of tensor/view monoliths and complete CPU semantics.

### Work

- [ ] Move the current logical-stride copy algorithm into `src/cpu/copy.c` or split it into focused helpers if needed.
- [ ] Implement CPU fill for every supported dtype.
- [ ] Implement CPU copy for documented contiguous and non-contiguous cases.
- [ ] Implement CPU cast according to R07 conversion rules.
- [ ] Implement CPU contiguous materialization using the common layout/offset rules.
- [ ] Ensure source and destination aliasing/overlap is handled according to the contract.
- [ ] Ensure no pointer arithmetic is performed on NULL for zero-byte tensors.
- [ ] Ensure all loops use checked dimensions/offsets and avoid signed overflow.
- [ ] Keep public wrappers and validation in core; keep actual CPU data access and loops in `src/cpu/`.

### Acceptance criteria

- [ ] CPU fill/copy/cast/materialize tests cover all supported dtypes.
- [ ] Tests cover scalar, empty, contiguous, non-contiguous, offset and transposed tensors.
- [ ] Tests verify alias visibility for in-place fill and storage independence for deep copy/materialization.
- [ ] Conversion tests cover boundary values, negative values, fractions, NaN/infinity where applicable, and complex values.
- [ ] Existing CPU test suite passes.

---

## R10 — Prepare CUDA data-movement implementation

**Priority:** P1  
**Depends on:** R07, R08  
**Purpose:** Make the CUDA implementation path concrete without requiring a CUDA device on every developer machine.

### Work

- [ ] Identify the existing CUDA build integration and confirm which CUDA source files are conditionally compiled.
- [ ] Keep CUDA memory allocation/release in CUDA-specific implementation files.
- [ ] Implement or plan CUDA fill kernels for supported dtypes.
- [ ] Implement or plan device copy kernels and use CUDA copy primitives where semantics permit.
- [ ] Implement or plan cast kernels matching the exact CPU conversion rules.
- [ ] Implement or plan contiguous materialization for non-contiguous device tensors.
- [ ] Define launch-error checking and the initial synchronization policy.
- [ ] Define how asynchronous execution will be added later without changing public operation semantics.
- [ ] Ensure CUDA implementation never passes device pointers to ordinary host `memcpy`, `memset`, or CPU loops.
- [ ] If no NVIDIA GPU is available during this sprint, compile-check CUDA code where possible and mark runtime parity as pending rather than complete.

### Acceptance criteria

- [ ] CUDA-specific code is isolated from public headers and CPU source files.
- [ ] CUDA-enabled build compiles on a configured CUDA toolchain.
- [ ] CPU-only build remains unaffected.
- [ ] Runtime tests are clearly marked as requiring a usable NVIDIA GPU.
- [ ] No CUDA runtime success is claimed based only on compilation.

---

## R11 — Build-system and test reorganization

**Priority:** P0  
**Depends on:** R01–R10 as applicable  
**Purpose:** Ensure the new module boundaries are reflected in builds and tests.

### Work

- [ ] Update CMake source lists for every extracted core module.
- [ ] Keep CPU sources in the CPU target/source group.
- [ ] Keep CUDA sources behind the existing CUDA availability/build option.
- [ ] Ensure ordinary `.c` core files compile as C and do not include CUDA runtime headers.
- [ ] Preserve current test targets and add focused test files only where the suite needs clearer ownership.
- [ ] Add public-header compile tests from an external-consumer perspective.
- [ ] Add CPU-only configure/build/test verification.
- [ ] Add CUDA configure/build verification when a toolchain is available.
- [ ] Add ASan/UBSan verification for CPU tests where supported.
- [ ] Update `docs/STRUCTURE.md` to describe actual modules, not planned modules that do not exist.
- [ ] Update `docs/BACKLOG.md` only after acceptance criteria are verified.

### Acceptance criteria

- [ ] Clean CPU-only configure, build, and test pass.
- [ ] Existing test count and coverage are not accidentally reduced.
- [ ] Sanitizer run passes on supported compiler/toolchain.
- [ ] CUDA build path is conditionally isolated and does not break CPU-only builds.
- [ ] Structure and backlog documentation match the resulting repository.

---

## R12 — Refactor completion review and handoff

**Priority:** P0  
**Depends on:** R01–R11  
**Purpose:** Verify that the refactor genuinely enables the upcoming implementation backlog.

### Work

- [ ] Review all public declarations against definitions.
- [ ] Review all internal declarations against definitions and call sites.
- [ ] Search for duplicate implementations and stale declarations.
- [ ] Search core files for direct host-memory operations that should belong to CPU backend.
- [ ] Search CPU files for CUDA includes/types.
- [ ] Search public headers for internal structs, backend types, or CUDA types.
- [ ] Verify metadata-only views do not access tensor data.
- [ ] Verify data movement reaches a backend through the agreed dispatch seam.
- [ ] Record known limitations, especially lack of CUDA runtime testing if no GPU runner is available.
- [ ] Produce a short refactor report: modules moved, APIs changed, tests run, known defects remaining, and next tasks unblocked.

### Acceptance criteria

- [ ] No known public declaration/implementation mismatch remains without an explicit deferred issue.
- [ ] No known core/backend responsibility violation remains without an explicit rationale.
- [ ] All required tests and build checks pass.
- [ ] T08 can proceed without redesigning tensor lifecycle, storage ownership, or backend dispatch.
- [ ] T09–T11 can add CPU/CUDA kernels without moving code back out of `ten_tensor.c` or `ten_view.c`.

---

# Suggested sprint order

Use this order to minimize simultaneous changes and keep the repository buildable:

| Order | Task | Main deliverable                                |
| ----: | ---- | ----------------------------------------------- |
|     1 | R01  | Clean public `ten_tensor.h`                     |
|     2 | R02  | Correct API mismatches and regression tests     |
|     3 | R03  | `core/storage.c`                                |
|     4 | R04  | `core/shape.c` and `core/tensor_layout.c`       |
|     5 | R05  | Focused `core/tensor.c`                         |
|     6 | R06  | Metadata-only `core/views.c`                    |
|     7 | R07  | Documented T08 contracts                        |
|     8 | R08  | Private backend interface and dispatch seam     |
|     9 | R09  | CPU data-movement implementation                |
|    10 | R10  | CUDA implementation path / compile verification |
|    11 | R11  | Build, tests, and documentation alignment       |
|    12 | R12  | Final review and handoff                        |

## Suggested sprint slices

### Slice A — API and correctness baseline

R01, R02

**Exit gate:** public API is coherent; known declaration/definition bugs are fixed and covered by tests.

### Slice B — Core decomposition

R03, R04, R05, R06

**Exit gate:** storage, shape/layout, tensor lifecycle, and views have separate ownership; CPU tests still pass.

### Slice C — Backend-ready data movement

R07, R08, R09, R10

**Exit gate:** contracts and dispatch are established; CPU data movement works; CUDA path is isolated and its verification status is honest.

### Slice D — Integration and handoff

R11, R12

**Exit gate:** clean builds/tests pass and the next implementation backlog can begin.

---

# Explicitly out of scope for this sprint

Do not let this refactor sprint expand into implementing the whole numerical library.

- Elementwise arithmetic kernels (T09).
- Matrix multiplication kernels (T10).
- Reduction kernels (T11).
- Full CUDA numerical kernel suite (T13).
- Neural-network primitives (T15).
- Autograd.
- Asynchronous streams/events unless required to preserve an already documented API.
- Memory pools, kernel fusion, multi-GPU, or advanced vectorization.
- Unrelated naming/style rewrites.

These remain in their existing implementation backlogs.

# Handoff: next backlog entry criteria

Before starting the next feature sprint, confirm:

- [ ] Public tensor API is stable enough to consume.
- [ ] Storage allocation/release is backend-aware.
- [ ] View operations are metadata-only and device-independent.
- [ ] Data-movement contracts are documented.
- [ ] Core dispatch can select CPU or CUDA without leaking backend details.
- [ ] CPU data movement is tested.
- [ ] CUDA implementation path is isolated; runtime verification status is documented.
- [ ] CMake supports CPU-only and conditional CUDA builds.
- [ ] `docs/STRUCTURE.md` and `docs/BACKLOG.md` reflect the actual code.

**Next feature sprint:** complete the remaining T08/T12 acceptance work, then implement T09 elementwise operations as vertical slices (shared contract → CPU reference → CUDA kernel → parity tests).
