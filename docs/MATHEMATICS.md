# libten — Mathematics and Invariants

## 1. Rank, shape, and indices

For rank `r`, shape is `(d0, d1, ..., d[r-1])`. Each index satisfies `0 <= ik < dk`. A scalar has rank zero and one logical element (the empty product is one). Decide and document support for zero-sized axes.

For positive dimensions, element count is `N = product(dk)`. Byte count is `N * dtype_size`. Both calculations require checked arithmetic. A zero-sized dimension must not cause division-by-zero in stride logic.

## 2. Contiguous row-major strides

The last axis varies fastest:
- `stride[r-1] = 1`
- `stride[k] = stride[k+1] * d[k+1]`

Shape `(2, 3, 4)` has strides `(12, 4, 1)`. Index `(1, 2, 3)` maps to element offset `1*12 + 2*4 + 3 = 23`. Stride multiplication must be overflow-checked.

## 3. General strided addressing

For index tuple `i`, strides `s`, and storage offset `o` (all element units):

`element_offset = o + sum(ik * sk)`

Convert to bytes only after multiplying by dtype size with checked arithmetic. Validate bounds before forming/dereferencing a pointer. Strides and offsets are measured in elements; allocation sizes are measured in bytes.

## 4. Views and aliasing

A view changes shape, strides, or offset while sharing storage. Transpose/permutation reorders shape and strides; slicing changes shape and usually offset/stride; squeeze removes size-one axes; unsqueeze inserts one. Reshape can be a view only when the requested shape is representable by the existing element order and stride rules; otherwise the API must explicitly fail or materialize a copy.

Views keep storage alive. Aliases can observe each other's writes. Document mutability, overlapping input/output behavior, and whether in-place operations are allowed.

## 5. Broadcasting

Align dimensions from the trailing axes. Dimensions are compatible when equal or when one is `1`; the output dimension is the common size or the non-one size.

- `(2, 3)` with `(3)` -> `(2, 3)`
- `(4, 1, 3)` with `(1, 5, 3)` -> `(4, 5, 3)`
- `(2, 3)` with `(2, 4)` -> incompatible

Broadcasting changes iteration, not necessarily storage. Zero-stride views are one implementation option. Writes to repeated broadcast locations must be prohibited or explicitly defined. Do not silently broadcast output buffers or cast dtypes.

## 6. Complex arithmetic

Let `z = a + bi`, `w = c + di`, and `i*i = -1`.

- Addition: `(a+c) + (b+d)i`
- Subtraction: `(a-c) + (b-d)i`
- Multiplication: `(ac-bd) + (ad+bc)i`
- Conjugate: `conj(z) = a - bi`
- Magnitude: `sqrt(a*a + b*b)`

`C64` means two F32 components (typically 8 bytes total); `C128` means two F64 components (typically 16 bytes total). Define component order and alignment. Complex division should avoid naive formulas that overflow/underflow for extreme magnitudes. Specify NaN, infinity, signed-zero, and divide-by-zero behavior. Ordinary complex matmul does not conjugate operands; conjugate transpose is a distinct operation.

## 7. Matrix multiplication

For `A` shape `(M,K)` and `B` shape `(K,N)`, `C` shape is `(M,N)` and:

`C[i,j] = sum(k=0..K-1) A[i,k] * B[k,j]`

The inner dimensions must match. Batched matmul later extends this rule to leading batch dimensions with documented batch broadcasting. Floating-point addition is not associative; CPU/CUDA values may differ slightly, so use dtype-appropriate tolerances rather than universal bitwise equality.

## 8. Reductions and ML primitives

A reduction over axis `a` removes that axis unless `keepdims` retains it with size one. Define empty-reduction behavior and accumulator dtype. Mean is sum divided by the reduced element count; mean of zero elements requires an explicit policy.

Useful higher-level formulas:
- Affine layer: `Y = X W + b`
- Stable softmax: `softmax(x_i) = exp(x_i - m) / sum_j exp(x_j - m)`, where `m = max_j x_j`
- Mean squared error: mean of squared differences with documented reduction and dtype rules.

Autograd is a later layer, but operation contracts should not make future backward implementations unnecessarily difficult.

## 9. Required mathematical tests

Test element counts and overflow, strides and offsets, view indexing against a reference, broadcasting compatibility, real/complex arithmetic, matmul and reductions across supported dtypes, scalar/empty/singleton/non-contiguous cases where supported, invalid requests, and CPU/CUDA parity with documented tolerances.
