# Loss Testing Subagent Constraints & Mission

## STRICT CLEAN-ROOM DIRECTIVES
1. **ABSOLUTE SOURCE CODE ACCESS BAN**: You are **STRICTLY FORBIDDEN** from viewing, opening, reading, grepping, or inspecting ANY files under `src/` (including `src/losses/*`, `src/tensor.rs`, `src/modules/*`, `src/lib.rs`, `src/main.rs`).
   - Any attempt to read `src/` violates the clean-room testing protocol and will result in invalid, biased tests.
   - You must NOT ask to see any `src/` files.
2. **SOURCES OF TRUTH**: You MUST ONLY rely on the public API declarations and mathematical specifications in:
   - [tests/LOSSES_API.md](file:///c:/Users/parth/Projects/NeuralNetworksRust/tests/LOSSES_API.md)
   - [tests/TENSOR_API.md](file:///c:/Users/parth/Projects/NeuralNetworksRust/tests/TENSOR_API.md)
3. **MISSION**: Your job is to create a rigorous, independent black-box test suite in `tests/test_losses.rs`.

## Required Testing Methodology

### 1. Analytical Tests
Test known, hand-calculated examples:
- `MSELoss`: 1D and 2D manual calculations ($L = \frac{1}{N}\sum (\hat{y}_i - y_i)^2$).
- `CrossEntropyLoss`: Hand-calculated LogSumExp loss for known logits and both 1D class indices and 2D one-hot distributions.
- `NLLLoss`: Hand-calculated $-\ln(P_y)$ for known probability vectors.

### 2. Numerical Gradient Checking (Finite Differences)
Implement central finite-difference verification:
$$\frac{\partial L}{\partial x_i} \approx \frac{L(x + \epsilon e_i) - L(x - \epsilon e_i)}{2\epsilon} \quad (\epsilon = 10^{-4})$$
Assert that analytical gradients match numerical gradients within absolute error $< 10^{-3}$.

### 3. Deep Learning Invariance Testing
Verify the fundamental equivalence:
`CrossEntropyLoss(logits, targets) == NLLLoss(Softmax(logits), targets)`
Both forward loss scalars and backward gradients on `logits` must be identical within floating-point tolerance ($10^{-4}$).

### 4. Edge Cases & Panic Assertions
- Batch size = 1 (single sample) vs large batches.
- Single class / binary vs multi-class (e.g. 10 classes).
- Shape mismatch panics using `#[should_panic]`.
- Target class index out of bounds panics using `#[should_panic]`.
