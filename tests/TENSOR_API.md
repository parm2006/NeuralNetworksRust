# Tensor API Interface Specification
*This document acts as the C++-style header (.h) for NeuralNetworksRust. Test subagents must use this file as their sole reference for types, methods, and autograd specifications.*

```rust
use std::rc::Rc;
use std::cell::{RefCell, RefMut};
use std::ops::{Add, Mul, BitXor};

// Public Tensor struct
#[derive(Clone)]
pub struct Tensor(pub Rc<RefCell<TensorInternal>>);
```

---

## 1. Constructors & Conversions

```rust
// 0D scalar from f32. Shape: [], Stride: []
impl From<f32> for Tensor;

// 1D vector from Vec<f32>. Shape: [len], Stride: [1]
impl From<Vec<f32>> for Tensor;

// 2D matrix from Vec<Vec<f32>>. Shape: [rows, cols], Stride: [cols, 1]
impl From<Vec<Vec<f32>>> for Tensor;

impl Tensor {
    // Generic constructor: takes data and shape.
    // Panics if data.len() != shape.iter().product()
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self;

    // Random uniform values in [-1.0, 1.0]
    pub fn random(shape: Vec<usize>) -> Self;

    // Zero-initialized tensor
    pub fn zeros(shape: Vec<usize>) -> Self;

    // 2D Kaiming initialization (shape: [in_features, out_features])
    pub fn kaiming(shape: Vec<usize>) -> Self;

    // 2D Xavier initialization (shape: [in_features, out_features])
    pub fn xavier(shape: Vec<usize>) -> Self;
}
```

---

## 2. Autograd & State Accessors

```rust
impl Tensor {
    // Check if tensor accumulates gradients
    pub fn requires_grad(&self) -> bool;

    // Builder method to enable requires_grad
    pub fn with_requires_grad(self) -> Self;

    // In-place setter for requires_grad
    pub fn set_requires_grad(&mut self, boolean_value: bool);

    // Returns accumulated gradient if requires_grad is true and grads exist
    pub fn grad(&self) -> Option<Vec<f32>>;

    // Resets all accumulated gradients to 0.0
    pub fn zero_grad(&self);

    // Total count of elements (product of shape)
    pub fn numel(&self) -> usize;

    // Cloned shape vector
    pub fn shape(&self) -> Vec<usize>;

    // Cloned stride vector
    pub fn stride(&self) -> Vec<usize>;

    // Scalar float value (panics if numel != 1)
    pub fn item(&self) -> f32;
    pub fn item_mut(&self) -> RefMut<'_, f32>;

    // Multi-dimensional index access
    pub fn at(&self, indices: &[usize]) -> f32;
    pub fn at_mut(&self, indices: &[usize]) -> RefMut<'_, f32>;

    // Raw data buffer access & mutation
    pub fn data(&self) -> Vec<f32>;
    pub fn set_data(&self, new_data: Vec<f32>);
}
```

---

## 3. Matrix Transformations

```rust
impl Tensor {
    // Transposes the last two dimensions (rows and cols).
    // Supports 2D, 3D, 4D, etc. tensors (requires shape.len() >= 2).
    // Batch dimensions are preserved untouched.
    pub fn t(&self) -> Self;
}
```

---

## 4. Operator Overloads & Autograd Rules

### Addition (`+`)
- **Supported types**: `&Tensor + &Tensor`, `Tensor + Tensor`, `&Tensor + Tensor`, `Tensor + &Tensor`, and scalar combinations (`&Tensor + f32`, etc.).
- **Autograd Math**:
  - Same shape: $\frac{\partial L}{\partial \text{self}} = G$, $\frac{\partial L}{\partial \text{other}} = G$.
  - Broadcasted shape: the smaller tensor sums/reduces incoming gradients across the broadcasted dimensions.
  - Scalar addition: $\frac{\partial L}{\partial \text{self}} = G$.

### Element-Wise Multiplication (`^`, BitXor)
- **Supported types**: `&Tensor ^ &Tensor`, `Tensor ^ Tensor`, `&Tensor ^ Tensor`, `Tensor ^ &Tensor`.
- **Requirements**: Both operands must have the exact same shape.
- **Autograd Math (Product Rule)**:
  - $\frac{\partial L}{\partial \text{self}} = G \odot \text{other}$
  - $\frac{\partial L}{\partial \text{other}} = G \odot \text{self}$

### Matrix Multiplication (`*`, Mul)
- **Supported types**: `&Tensor * &Tensor`, `Tensor * Tensor`.
- **Requirements**:
  - 1D $\times$ 1D: dot product (returns scalar 1-element tensor `[1]`).
  - 2D $\times$ 2D: $[M, K] \times [K, N] \to [M, N]$.
  - Batched: $[..., M, K] \times [..., K, N] \to [..., M, N]$ with batch broadcasting.
  - Vector promotions: 1D $\times$ 2D and 2D $\times$ 1D.
- **Autograd Math**:
  - $\frac{\partial L}{\partial \text{self}} = G \times \text{other}^T$
  - $\frac{\partial L}{\partial \text{other}} = \text{self}^T \times G$
  - Broadcasted batch dimensions are reduced/summed into the parameter's gradient shape.

### Scalar Multiplication (`*`, Mul)
- **Supported types**: `&Tensor * f32`, `f32 * &Tensor`, `Tensor * f32`, `f32 * Tensor`.
