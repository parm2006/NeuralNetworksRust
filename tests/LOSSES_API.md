# Losses API Interface Specification
*This document acts as the C++-style header (.h) specification for the loss functions in NeuralNetworksRust. Test subagents must use this file and tests/TENSOR_API.md as their sole reference for types, methods, behaviors, and autograd specifications. Subagents must NEVER read or inspect any files inside src/.*

```rust
use NeuralNetworksRust::losses::{Loss, MSELoss, NLLLoss, CrossEntropyLoss};
use NeuralNetworksRust::tensor::Tensor;
use NeuralNetworksRust::modules::{Module, Softmax};
```

---

## 1. The `Loss` Trait

All loss functions implement the `Loss` trait:

```rust
pub trait Loss {
    /// Computes the forward loss scalar tensor between predictions and targets,
    /// registering the autograd backward pass closure on predictions.
    ///
    /// - predictions: Model output tensor.
    /// - targets: Ground-truth target tensor.
    /// - Returns: Scalar Tensor with shape [1] containing the reduced mean loss.
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor;
}
```

---

## 2. `MSELoss` (Mean Squared Error)

Computes the element-wise mean squared difference between predictions and targets.

### Constructors
```rust
impl MSELoss {
    pub fn new() -> Self;
}
```

### Input Specifications
- **`predictions`**: Tensor of arbitrary shape (e.g. `[N]`, `[B, D]`, `[B, C, H, W]`).
- **`targets`**: Tensor with the exact same shape as `predictions`.
- **Panics**: Panics with a shape mismatch error if `predictions.shape() != targets.shape()`.

### Mathematical Formulation
- **Forward Reduction (Mean)**:
  $$L = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2$$
  where $N = \text{predictions.numel()}$.
- **Backward Gradient (Vector-Jacobian Product)**:
  $$\frac{\partial L}{\partial \hat{y}_i} = \frac{2}{N} (\hat{y}_i - y_i) \cdot G$$
  where $G = \text{grad}[0]$ is the incoming scalar gradient (usually $1.0$).

---

## 3. `CrossEntropyLoss` (Categorical Cross-Entropy on Logits)

Computes cross-entropy directly on **raw, unnormalized logits** $z \in (-\infty, +\infty)$ (e.g. the output of a `Linear` layer with no subsequent `Softmax`). Combines Softmax and Negative Log-Likelihood in a single, numerically stable step using the Log-Sum-Exp trick.

### Constructors
```rust
impl CrossEntropyLoss {
    pub fn new() -> Self;
}
```

### Input Specifications
- **`predictions`**: Raw logits tensor of shape `[batch_size, num_classes]` (or 1D `[num_classes]` where `batch_size = 1`).
- **`targets`**: Supports two distinct target representations:
  1. **Class Indices (1D)**: Tensor of shape `[batch_size]` where each element is an integer float $y_b \in \{0, 1, \dots, C-1\}$.
  2. **One-Hot / Probability Vectors (2D)**: Tensor of shape `[batch_size, num_classes]` matching `predictions.shape()`.
- **Panics**:
  - Panics if `targets` is neither 1D of length `batch_size` nor matching `predictions.shape()`.
  - Panics if a class index is $\ge \text{num\_classes}$.

### Mathematical Formulation
- **Log-Sum-Exp Trick**:
  $$z_{\max} = \max_{c} z_{b,c}$$
  $$\text{log\_sum\_exp}_b = z_{\max} + \ln \sum_{c=1}^C e^{z_{b,c} - z_{\max}}$$
  $$P_{b,c} = \frac{e^{z_{b,c} - z_{\max}}}{\sum_{k=1}^C e^{z_{b,k} - z_{\max}}}$$
- **Forward Loss**:
  - For class index target $y_b$: $\text{loss}_b = \text{log\_sum\_exp}_b - z_{b, y_b}$
  - For one-hot target $Y_{b}$: $\text{loss}_b = \text{log\_sum\_exp}_b - \sum_{c=1}^C Y_{b,c} z_{b,c}$
  - Reduced mean loss: $L = \frac{1}{B} \sum_{b=1}^B \text{loss}_b$ (returns `Tensor` of shape `[1]`).
- **Backward Gradient**:
  $$\frac{\partial L}{\partial z_{b,c}} = \frac{1}{B} (P_{b,c} - Y_{b,c}) \cdot G$$
  where $Y_{b,c} = 1.0$ if $c == y_b$ else $0.0$.

---

## 4. `NLLLoss` (Negative Log-Likelihood on Probabilities)

Computes negative log-likelihood on **predicted probabilities** $P \in (0, 1)$ (e.g. the output of a `Softmax` layer).

### Constructors
```rust
impl NLLLoss {
    pub fn new() -> Self;
    pub fn with_eps(eps: f32) -> Self; // Default eps is 1e-7
}
```

### Input Specifications
- **`predictions`**: Probability tensor of shape `[batch_size, num_classes]` where elements are in $(0, 1)$ and sum to $1.0$ along the class dimension.
- **`targets`**:
  1. **Class Indices (1D)**: Tensor of shape `[batch_size]` where each element is an integer float $y_b \in \{0, 1, \dots, C-1\}$.
  2. **One-Hot / Distributions (2D)**: Tensor of shape `[batch_size, num_classes]`.
- **Panics**: Same shape and index validation as `CrossEntropyLoss`.

### Mathematical Formulation
- **Forward Loss**:
  - For class index target $y_b$: $\text{loss}_b = -\ln(\max(P_{b, y_b}, \epsilon))$
  - For one-hot target $Y_{b}$: $\text{loss}_b = -\sum_{c=1}^C Y_{b,c} \ln(\max(P_{b,c}, \epsilon))$
  - Reduced mean loss: $L = \frac{1}{B} \sum_{b=1}^B \text{loss}_b$ (returns `Tensor` of shape `[1]`).
- **Backward Gradient**:
  $$\frac{\partial L}{\partial P_{b,c}} = -\frac{1}{B} \frac{Y_{b,c}}{\max(P_{b,c}, \epsilon)} \cdot G$$

---

## 5. Architectural Equivalence Invariant

The test suite must verify the fundamental deep learning invariant:

For any raw logits tensor $z$ and targets $y$:
```rust
// Pipeline A: Unified Cross-Entropy
let loss_ce = CrossEntropyLoss::new().forward(&z_ce, &y);
loss_ce.backward();

// Pipeline B: Softmax layer followed by NLLLoss
let probs = Softmax::new().forward(&z_nll);
let loss_nll = NLLLoss::new().forward(&probs, &y);
loss_nll.backward();
```
- **Forward Invariant**: `loss_ce.data()[0] == loss_nll.data()[0]` (within floating-point tolerance $10^{-4}$).
- **Backward Invariant**: `z_ce.grad().unwrap() == z_nll.grad().unwrap()` (within floating-point tolerance $10^{-4}$).
