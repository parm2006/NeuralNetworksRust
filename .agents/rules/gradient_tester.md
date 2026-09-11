# Autograd Testing Subagent Constraints & Mission

## STRICT SECURITY & INTEGRITY DIRECTIVES
1. **ABSOLUTE ACCESS BAN**: You are **STRICTLY FORBIDDEN** from viewing, opening, reading, grepping, or inspecting `src/tensor.rs`. 
   - Any attempt to read `src/tensor.rs` violates the clean-room testing protocol and will result in invalid, biased tests.
   - You must NOT ask to see `src/tensor.rs`.
2. **SOURCE OF TRUTH**: You MUST ONLY rely on the public API declarations in [tests/TENSOR_API.rs](file:///c:/Users/parth/Projects/NeuralNetworksRust/tests/TENSOR_API.rs) and the standard library traits.
3. **MISSION**: Your job is to create a rigorous, independent test suite in `tests/test_gradients.rs`.

## Testing Methodology
Because you cannot see the internal code, your test suite must rely on mathematically rigorous techniques:

### 1. Analytical Tests
Test simple known operations where the exact derivative can be worked out by hand:
- Addition: $f(x, y) = x + y \implies \nabla_x = 1, \nabla_y = 1$
- Element-wise product: $f(x, y) = x \odot y \implies \nabla_x = y, \nabla_y = x$
- Linear combinations: $f(x, y) = (x \odot y) + x$
- Matrix multiplication: $C = A \cdot B \implies \nabla_A = G \cdot B^T, \nabla_B = A^T \cdot G$
- Transpose: $Y = X^T \implies \nabla_X = G^T$

### 2. Numerical Gradient Checking (Finite Differences)
Implement a finite-difference verification helper in the test file:
$$\frac{\partial f}{\partial x_i} \approx \frac{f(x + \epsilon e_i) - f(x - \epsilon e_i)}{2\epsilon} \quad (\epsilon = 10^{-4})$$
Compare the analytical gradient returned by the autograd engine with this numerical approximation.
Assert that the relative or absolute error is $< 10^{-3}$.

### 3. Edge Cases
- Tensors where `requires_grad` is false (should not accumulate gradients).
- Asymmetric `requires_grad` (e.g. in $A \cdot B$, only $A$ requires grad, or only $B$ requires grad).
- Accumulated gradients across multiple paths / nodes.
- Calling `.zero_grad()` to verify gradient clearing.
- 1D, 2D, and batched 3D matrix multiplication gradient shapes and values.
