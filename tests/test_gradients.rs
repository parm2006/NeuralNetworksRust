// tests/test_gradients.rs
// ============================================================================
// Clean-Room Black-Box Autograd & Gradient Test Suite
// Based strictly on public interface specifications in tests/TENSOR_API.md.
// Per AGENTS.md clean-room protocol, no internal implementation files are inspected.
// ============================================================================

use NeuralNetworksRust::tensor::Tensor;

// ============================================================================
// Numerical Gradient Checking & Test Utilities
// ============================================================================

/// Extracts all elements from a tensor into a flat Vec<f32> in row-major order
/// using only public shape() and at() accessors.
fn get_flat_data(t: &Tensor) -> Vec<f32> {
    let shape = t.shape();
    if shape.is_empty() {
        return vec![t.item()];
    }
    let numel = t.numel();
    let mut data = Vec::with_capacity(numel);
    let mut idx = vec![0; shape.len()];
    for _ in 0..numel {
        data.push(t.at(&idx));
        for d in (0..shape.len()).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    data
}

/// Computes the numerical gradient of a scalar function f(x) with respect to
/// each element of tensor x using two-sided central finite differences:
///   d f / d x_i ≈ [ f(x + ε e_i) - f(x - ε e_i) ] / (2 * ε)
fn numerical_gradient<F>(f: F, x: &Tensor, eps: f32) -> Vec<f32>
where
    F: Fn(&Tensor) -> Tensor,
{
    let shape = x.shape();
    let flat_data = get_flat_data(x);
    let n = flat_data.len();
    let mut num_grad = Vec::with_capacity(n);

    for i in 0..n {
        let mut x_plus_data = flat_data.clone();
        x_plus_data[i] += eps;
        let x_plus = Tensor::new(x_plus_data, shape.clone());
        let y_plus = f(&x_plus);

        let mut x_minus_data = flat_data.clone();
        x_minus_data[i] -= eps;
        let x_minus = Tensor::new(x_minus_data, shape.clone());
        let y_minus = f(&x_minus);

        let diff = (y_plus.item() - y_minus.item()) / (2.0 * eps);
        num_grad.push(diff);
    }
    num_grad
}

/// Asserts that actual and expected gradient vectors match within a given absolute tolerance.
fn assert_grads_close(actual: &[f32], expected: &[f32], tol: f32, msg: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: gradient size mismatch (actual {} vs expected {})",
        msg,
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff <= tol,
            "{}: gradient mismatch at element [{}]: actual={}, expected={}, diff={} > tol={}",
            msg,
            i,
            a,
            e,
            diff,
            tol
        );
    }
}

// ============================================================================
// 1. Addition Autograd Tests (+)
// ============================================================================

#[test]
fn test_grad_add_0d_scalar() {
    let a = Tensor::from(3.0).with_requires_grad();
    let b = Tensor::from(7.0).with_requires_grad();

    let c = &a + &b;
    c.backward();

    let grad_a = a.grad().expect("a must have grad");
    let grad_b = b.grad().expect("b must have grad");

    assert_eq!(grad_a, vec![1.0]);
    assert_eq!(grad_b, vec![1.0]);
}

#[test]
fn test_grad_add_1d_same_shape() {
    let a = Tensor::from(vec![1.0, 2.0, 3.0]).with_requires_grad();
    let b = Tensor::from(vec![4.0, 5.0, 6.0]).with_requires_grad();

    // Loss = sum(a + b) via dot product with ones
    let ones = Tensor::from(vec![1.0, 1.0, 1.0]);
    let c = &a + &b;
    let loss = &c * &ones;
    loss.backward();

    let grad_a = a.grad().expect("a must have grad");
    let grad_b = b.grad().expect("b must have grad");

    assert_grads_close(&grad_a, &[1.0, 1.0, 1.0], 1e-4, "grad_a for 1D add");
    assert_grads_close(&grad_b, &[1.0, 1.0, 1.0], 1e-4, "grad_b for 1D add");
}

#[test]
fn test_grad_add_1d_numerical_check() {
    let a_init = Tensor::from(vec![1.5, -2.0, 3.7]).with_requires_grad();
    let b_const = Tensor::from(vec![0.5, 4.2, -1.1]);
    let ones = Tensor::from(vec![1.0, 1.0, 1.0]);

    let f = |x: &Tensor| -> Tensor {
        let sum_ab = x + &b_const;
        &sum_ab * &ones
    };

    let num_grad = numerical_gradient(f, &a_init, 1e-3);

    let loss = f(&a_init);
    loss.backward();
    let analytical_grad = a_init.grad().expect("a_init must have grad");

    assert_grads_close(
        &analytical_grad,
        &num_grad,
        1e-2,
        "1D addition numerical comparison",
    );
}

#[test]
fn test_grad_add_2d_same_shape() {
    let a = Tensor::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).with_requires_grad();
    let b = Tensor::from(vec![vec![5.0, 6.0], vec![7.0, 8.0]]).with_requires_grad();

    let c = &a + &b;
    // Loss = 1_1x2 * (c * 1_2x1) = sum of all 4 elements
    let ones_col = Tensor::from(vec![vec![1.0], vec![1.0]]);
    let ones_row = Tensor::from(vec![vec![1.0, 1.0]]);
    let loss = &ones_row * &(&c * &ones_col);
    loss.backward();

    let grad_a = a.grad().expect("a must have grad");
    let grad_b = b.grad().expect("b must have grad");

    assert_grads_close(&grad_a, &[1.0, 1.0, 1.0, 1.0], 1e-4, "2D add grad a");
    assert_grads_close(&grad_b, &[1.0, 1.0, 1.0, 1.0], 1e-4, "2D add grad b");
}

#[test]
fn test_grad_add_scalar_constant() {
    let a = Tensor::from(vec![2.0, 4.0, 6.0]).with_requires_grad();
    let ones = Tensor::from(vec![1.0, 1.0, 1.0]);

    let c = &a + 10.0;
    let loss = &c * &ones;
    loss.backward();

    let grad_a = a.grad().expect("a must have grad");
    assert_grads_close(&grad_a, &[1.0, 1.0, 1.0], 1e-4, "add scalar constant grad");
}

#[test]
fn test_grad_add_broadcast_reduction() {
    // a: 2x3 matrix, b: 1x3 or 1D vector of length 3
    // In TENSOR_API.md: "Broadcasted shape: the smaller tensor sums/reduces incoming gradients across the broadcasted dimensions."
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).with_requires_grad();
    let b = Tensor::from(vec![10.0, 20.0, 30.0]).with_requires_grad();

    let c = &a + &b;
    let ones_col = Tensor::from(vec![vec![1.0], vec![1.0], vec![1.0]]);
    let ones_row = Tensor::from(vec![vec![1.0, 1.0]]);
    let loss = &ones_row * &(&c * &ones_col);
    loss.backward();

    let grad_a = a.grad().expect("a must have grad");
    let grad_b = b.grad().expect("b must have grad");

    // G is all 1.0 of shape [2, 3].
    // a's shape is [2, 3], so grad_a is 6 ones.
    assert_grads_close(&grad_a, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 1e-4, "grad_a broadcast");
    // b's shape is [3], broadcast across 2 rows, so grad_b is sum across 2 rows = [2.0, 2.0, 2.0].
    assert_grads_close(&grad_b, &[2.0, 2.0, 2.0], 1e-4, "grad_b broadcast reduction");
}

// ============================================================================
// 2. Element-Wise Multiplication Autograd Tests (^, BitXor)
// ============================================================================

#[test]
fn test_grad_elemmul_0d() {
    let a = Tensor::from(4.0).with_requires_grad();
    let b = Tensor::from(5.0).with_requires_grad();

    let c = &a ^ &b;
    c.backward();

    assert_eq!(a.grad().unwrap(), vec![5.0]);
    assert_eq!(b.grad().unwrap(), vec![4.0]);
}

#[test]
fn test_grad_elemmul_1d_product_rule() {
    let a = Tensor::from(vec![2.0, 3.0, 4.0]).with_requires_grad();
    let b = Tensor::from(vec![5.0, 6.0, 7.0]).with_requires_grad();

    let ones = Tensor::from(vec![1.0, 1.0, 1.0]);
    let c = &a ^ &b;
    let loss = &c * &ones;
    loss.backward();

    // d(loss)/da = G * b = 1.0 * [5.0, 6.0, 7.0]
    // d(loss)/db = G * a = 1.0 * [2.0, 3.0, 4.0]
    assert_grads_close(&a.grad().unwrap(), &[5.0, 6.0, 7.0], 1e-4, "grad_a elemmul 1D");
    assert_grads_close(&b.grad().unwrap(), &[2.0, 3.0, 4.0], 1e-4, "grad_b elemmul 1D");
}

#[test]
fn test_grad_elemmul_1d_numerical_check() {
    let a = Tensor::from(vec![1.2, -3.4, 0.5]).with_requires_grad();
    let b_const = Tensor::from(vec![2.1, 4.3, -1.8]);
    let ones = Tensor::from(vec![1.0, 1.0, 1.0]);

    let f = |x: &Tensor| -> Tensor {
        let prod = x ^ &b_const;
        &prod * &ones
    };

    let num_grad = numerical_gradient(f, &a, 1e-3);
    let loss = f(&a);
    loss.backward();

    assert_grads_close(&a.grad().unwrap(), &num_grad, 1e-2, "elemmul 1D numerical");
}

#[test]
fn test_grad_elemmul_self_multiplication() {
    // z = x ^ x => d(loss)/dx = 2 * x (gradient accumulation from both branches)
    let x = Tensor::from(vec![3.0, -4.0, 5.0]).with_requires_grad();
    let ones = Tensor::from(vec![1.0, 1.0, 1.0]);

    let z = &x ^ &x;
    let loss = &z * &ones;
    loss.backward();

    assert_grads_close(
        &x.grad().unwrap(),
        &[6.0, -8.0, 10.0],
        1e-4,
        "elemmul self-multiplication gradient accumulation",
    );
}

#[test]
fn test_grad_composite_elemmul_and_add() {
    // z = (x ^ y) + x => d(loss)/dx = y + 1, d(loss)/dy = x
    let x = Tensor::from(vec![2.0, 3.0]).with_requires_grad();
    let y = Tensor::from(vec![4.0, 5.0]).with_requires_grad();
    let ones = Tensor::from(vec![1.0, 1.0]);

    let xy = &x ^ &y;
    let z = &xy + &x;
    let loss = &z * &ones;
    loss.backward();

    // dx = y + 1 = [5.0, 6.0]
    // dy = x = [2.0, 3.0]
    assert_grads_close(&x.grad().unwrap(), &[5.0, 6.0], 1e-4, "composite dx");
    assert_grads_close(&y.grad().unwrap(), &[2.0, 3.0], 1e-4, "composite dy");
}

// ============================================================================
// 3. Matrix Multiplication Autograd Tests (*, Mul)
// ============================================================================

#[test]
fn test_grad_matmul_1d_dot_product() {
    // 1D * 1D returns a 1-element scalar tensor [1]
    let a = Tensor::from(vec![1.0, 2.0, 3.0]).with_requires_grad();
    let b = Tensor::from(vec![4.0, 5.0, 6.0]).with_requires_grad();

    let loss = &a * &b;
    assert_eq!(loss.shape(), vec![1]);
    loss.backward();

    // d(a*b)/da = b, d(a*b)/db = a
    assert_grads_close(&a.grad().unwrap(), &[4.0, 5.0, 6.0], 1e-4, "1D dot grad a");
    assert_grads_close(&b.grad().unwrap(), &[1.0, 2.0, 3.0], 1e-4, "1D dot grad b");
}

#[test]
fn test_grad_matmul_1d_dot_product_self() {
    // z = x * x => dz/dx = 2 * x
    let x = Tensor::from(vec![2.0, -3.0, 4.0]).with_requires_grad();
    let loss = &x * &x;
    loss.backward();

    assert_grads_close(&x.grad().unwrap(), &[4.0, -6.0, 8.0], 1e-4, "1D dot self");
}

#[test]
fn test_grad_matmul_2d_square() {
    // A: 2x2, B: 2x2. C = A * B.
    // L = sum(C). G = [[1, 1], [1, 1]].
    // dL/dA = G * B^T
    // dL/dB = A^T * G
    let a = Tensor::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).with_requires_grad();
    let b = Tensor::from(vec![vec![5.0, 6.0], vec![7.0, 8.0]]).with_requires_grad();

    let c = &a * &b;
    let ones_col = Tensor::from(vec![vec![1.0], vec![1.0]]);
    let ones_row = Tensor::from(vec![vec![1.0, 1.0]]);
    let loss = &ones_row * &(&c * &ones_col);
    loss.backward();

    // B = [[5, 6], [7, 8]] => B^T = [[5, 7], [6, 8]]
    // G = [[1, 1], [1, 1]]
    // dL/dA = [[1, 1], [1, 1]] * [[5, 7], [6, 8]] = [[11, 15], [11, 15]]
    assert_grads_close(
        &a.grad().unwrap(),
        &[11.0, 15.0, 11.0, 15.0],
        1e-3,
        "matmul 2D grad A",
    );

    // A = [[1, 2], [3, 4]] => A^T = [[1, 3], [2, 4]]
    // dL/dB = [[1, 3], [2, 4]] * [[1, 1], [1, 1]] = [[4, 4], [6, 6]]
    assert_grads_close(
        &b.grad().unwrap(),
        &[4.0, 4.0, 6.0, 6.0],
        1e-3,
        "matmul 2D grad B",
    );
}

#[test]
fn test_grad_matmul_2d_rectangular_numerical_check() {
    // A: 2x3, B: 3x2 => C: 2x2
    let a_init = Tensor::from(vec![
        vec![1.0, -2.0, 3.0],
        vec![-4.0, 5.0, -6.0],
    ]).with_requires_grad();

    let b_const = Tensor::from(vec![
        vec![0.5, 1.5],
        vec![-1.0, 2.0],
        vec![2.5, -0.5],
    ]);

    let ones_col = Tensor::from(vec![vec![1.0], vec![1.0]]);
    let ones_row = Tensor::from(vec![vec![1.0, 1.0]]);

    let f = |x: &Tensor| -> Tensor {
        let prod = x * &b_const;
        &ones_row * &(&prod * &ones_col)
    };

    let num_grad = numerical_gradient(f, &a_init, 1e-3);
    let loss = f(&a_init);
    loss.backward();

    assert_grads_close(
        &a_init.grad().unwrap(),
        &num_grad,
        1e-2,
        "2D rectangular matmul numerical gradient check",
    );
}

#[test]
fn test_grad_matmul_vector_promotion_1d_times_2d() {
    // 1D (size 2) * 2D (2x3) -> 1D (size 3)
    let v = Tensor::from(vec![2.0, 3.0]).with_requires_grad();
    let m = Tensor::from(vec![
        vec![1.0, 4.0, 7.0],
        vec![2.0, 5.0, 8.0],
    ]).with_requires_grad();

    let ones3 = Tensor::from(vec![1.0, 1.0, 1.0]);
    let out = &v * &m;
    assert_eq!(out.shape(), vec![3]);

    let loss = &out * &ones3;
    loss.backward();

    let grad_v = v.grad().expect("v must have grad");
    let grad_m = m.grad().expect("m must have grad");

    // Output shape of grad_v must match original v shape [2]
    assert_eq!(grad_v.len(), 2, "grad_v should have 2 elements");
    // dL/dv_i = sum_j m_ij => v[0]: 1+4+7=12, v[1]: 2+5+8=15
    assert_grads_close(&grad_v, &[12.0, 15.0], 1e-3, "1D x 2D grad_v");

    // Output shape of grad_m must match original m shape [2, 3] (6 elements)
    assert_eq!(grad_m.len(), 6, "grad_m should have 6 elements");
    // dL/dm_ij = v_i * G_j = v_i * 1.0
    // row 0: [2, 2, 2], row 1: [3, 3, 3]
    assert_grads_close(&grad_m, &[2.0, 2.0, 2.0, 3.0, 3.0, 3.0], 1e-3, "1D x 2D grad_m");
}

#[test]
fn test_grad_matmul_vector_promotion_2d_times_1d() {
    // 2D (3x2) * 1D (size 2) -> 1D (size 3)
    let m = Tensor::from(vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
        vec![5.0, 6.0],
    ]).with_requires_grad();
    let v = Tensor::from(vec![10.0, 20.0]).with_requires_grad();

    let ones3 = Tensor::from(vec![1.0, 1.0, 1.0]);
    let out = &m * &v;
    assert_eq!(out.shape(), vec![3]);

    let loss = &out * &ones3;
    loss.backward();

    let grad_v = v.grad().expect("v must have grad");
    let grad_m = m.grad().expect("m must have grad");

    // grad_v shape must be [2]: sum_i m_ij = [1+3+5, 2+4+6] = [9.0, 12.0]
    assert_grads_close(&grad_v, &[9.0, 12.0], 1e-3, "2D x 1D grad_v");
    // grad_m shape must be [3, 2]: v repeated along rows = [10, 20, 10, 20, 10, 20]
    assert_grads_close(&grad_m, &[10.0, 20.0, 10.0, 20.0, 10.0, 20.0], 1e-3, "2D x 1D grad_m");
}

#[test]
fn test_grad_scalar_multiplication() {
    let a = Tensor::from(vec![1.0, -2.0, 3.0]).with_requires_grad();
    let ones = Tensor::from(vec![1.0, 1.0, 1.0]);

    // Right scalar multiplication: &Tensor * f32
    let c = &a * 4.0_f32;
    let loss = &c * &ones;
    loss.backward();

    assert_grads_close(&a.grad().unwrap(), &[4.0, 4.0, 4.0], 1e-4, "scalar mul right");

    // Left scalar multiplication: f32 * &Tensor
    a.zero_grad();
    let c2 = 2.5_f32 * &a;
    let loss2 = &c2 * &ones;
    loss2.backward();

    assert_grads_close(&a.grad().unwrap(), &[2.5, 2.5, 2.5], 1e-4, "scalar mul left");
}

// ============================================================================
// 4. Matrix Transformation Tests (Transpose)
// ============================================================================

#[test]
fn test_grad_transpose_2d() {
    // Y = X^T => dL/dX = (dL/dY)^T
    let x = Tensor::from(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
    ]).with_requires_grad();

    let y = x.t();
    assert_eq!(y.shape(), vec![3, 2]);

    let ones_col = Tensor::from(vec![vec![1.0], vec![1.0]]);
    let ones_row = Tensor::from(vec![vec![1.0, 1.0, 1.0]]);
    let loss = &ones_row * &(&y * &ones_col);
    loss.backward();

    let grad_x = x.grad().expect("x must have grad");
    assert_grads_close(&grad_x, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 1e-4, "transpose grad");
}

// ============================================================================
// 5. Engine State & Gradient Accumulation Tests
// ============================================================================

#[test]
fn test_grad_accumulation_across_backward_calls() {
    let x = Tensor::from(5.0).with_requires_grad();

    // First forward + backward: loss = 2 * x => grad = 2.0
    let loss1 = &x * 2.0_f32;
    loss1.backward();
    assert_eq!(x.grad().unwrap(), vec![2.0]);

    // Second forward + backward without zero_grad: loss = 3 * x => grad accumulates + 3.0 = 5.0
    let loss2 = &x * 3.0_f32;
    loss2.backward();
    assert_eq!(x.grad().unwrap(), vec![5.0]);

    // Calling zero_grad resets back to 0.0
    x.zero_grad();
    assert_eq!(x.grad().unwrap(), vec![0.0]);
}

#[test]
fn test_grad_requires_grad_flag_gating() {
    // When requires_grad is false, tensor does not accumulate gradient (grad remains None)
    let a = Tensor::from(3.0).with_requires_grad();
    let b = Tensor::from(4.0); // requires_grad is false by default

    assert!(a.requires_grad());
    assert!(!b.requires_grad());

    let c = &a + &b;
    c.backward();

    assert!(a.grad().is_some());
    assert!(b.grad().is_none(), "non-requiring tensor should have grad = None");
}

// ============================================================================
// 6. Multi-Layer MLP Forward Step Integration Test
// ============================================================================

#[test]
fn test_grad_mlp_linear_and_elemmul_step() {
    // Simulates a single feedforward layer step:
    //   Y = (X * W) + b
    //   Loss = sum(Y ^ Y)
    // Checking that backpropagation flows through matmul, add, and element-wise square simultaneously.

    let x = Tensor::from(vec![vec![1.0, -1.0], vec![0.5, 2.0]]).with_requires_grad();
    let w = Tensor::from(vec![vec![0.2, -0.4], vec![0.8, 0.1]]).with_requires_grad();
    let b = Tensor::from(vec![0.1, -0.2]).with_requires_grad();

    let ones_col = Tensor::from(vec![vec![1.0], vec![1.0]]);
    let ones_row = Tensor::from(vec![vec![1.0, 1.0]]);

    // Numerical gradient check for weights W
    let f_w = |w_test: &Tensor| -> Tensor {
        let x_const = Tensor::from(vec![vec![1.0, -1.0], vec![0.5, 2.0]]);
        let b_const = Tensor::from(vec![0.1, -0.2]);
        let y = &(&x_const * w_test) + &b_const;
        let y_sq = &y ^ &y;
        &ones_row * &(&y_sq * &ones_col)
    };

    let num_grad_w = numerical_gradient(f_w, &w, 1e-3);

    // Analytical forward + backward
    let y = &(&x * &w) + &b;
    let y_sq = &y ^ &y;
    let loss = &ones_row * &(&y_sq * &ones_col);
    loss.backward();

    let analytical_grad_w = w.grad().expect("w must have grad");
    assert_grads_close(
        &analytical_grad_w,
        &num_grad_w,
        1e-2,
        "MLP layer backward numerical vs analytical for weights",
    );
}
