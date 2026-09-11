// tests/test_add.rs
use NeuralNetworksRust::tensor::Tensor;

// =========================================================================
// Helper assertion functions to inspect tensor shape and elements
// =========================================================================

fn assert_tensor_0d(t: &Tensor, expected: f32) {
    assert_eq!(t.shape().as_slice(), &[] as &[usize], "0D tensor shape must be []");
    assert_eq!(t.item(), expected, "0D tensor value mismatch");
}

fn assert_tensor_1d(t: &Tensor, expected: &[f32]) {
    assert_eq!(
        t.shape().as_slice(),
        &[expected.len()],
        "1D tensor shape mismatch"
    );
    for (i, &val) in expected.iter().enumerate() {
        assert_eq!(
            t.at(&[i]),
            val,
            "1D tensor element mismatch at index [{}]",
            i
        );
    }
}

fn assert_tensor_2d(t: &Tensor, expected: &[&[f32]]) {
    let rows = expected.len();
    let cols = if rows > 0 { expected[0].len() } else { 0 };
    assert_eq!(
        t.shape().as_slice(),
        &[rows, cols],
        "2D tensor shape mismatch"
    );
    for i in 0..rows {
        for j in 0..cols {
            assert_eq!(
                t.at(&[i, j]),
                expected[i][j],
                "2D tensor element mismatch at index [{}, {}]",
                i,
                j
            );
        }
    }
}

// =========================================================================
// 1. 0D Tensor Addition Tests
// =========================================================================

#[test]
fn test_0d_plus_0d_owned() {
    let a = Tensor::from(3.5);
    let b = Tensor::from(4.5);
    let res = a + b;
    assert_tensor_0d(&res, 8.0);
}

#[test]
fn test_0d_plus_0d_ref() {
    let a = Tensor::from(12.0);
    let b = Tensor::from(8.0);
    let res = &a + &b;
    assert_tensor_0d(&res, 20.0);
}

// 0D + f32 scalar (Left)
#[test]
fn test_0d_plus_f32_scalar_left_owned() {
    let a = Tensor::from(10.0);
    let res = a + 5.5_f32;
    assert_tensor_0d(&res, 15.5);
}

#[test]
fn test_0d_plus_f32_scalar_left_ref() {
    let a = Tensor::from(10.0);
    let res = &a + 5.5_f32;
    assert_tensor_0d(&res, 15.5);
}

// f32 scalar + 0D (Right)
#[test]
fn test_f32_scalar_plus_0d_right_owned() {
    let a = Tensor::from(10.0);
    let res = 5.5_f32 + a;
    assert_tensor_0d(&res, 15.5);
}

#[test]
fn test_f32_scalar_plus_0d_right_ref() {
    let a = Tensor::from(10.0);
    let res = 5.5_f32 + &a;
    assert_tensor_0d(&res, 15.5);
}

// =========================================================================
// 2. 1D Tensor Addition Tests
// =========================================================================

#[test]
fn test_1d_plus_1d_owned() {
    let a = Tensor::from(vec![1.0, 2.0, 3.0]);
    let b = Tensor::from(vec![10.0, 20.0, 30.0]);
    let res = a + b;
    assert_tensor_1d(&res, &[11.0, 22.0, 33.0]);
}

#[test]
fn test_1d_plus_1d_ref() {
    let a = Tensor::from(vec![1.0, 2.0, 3.0]);
    let b = Tensor::from(vec![10.0, 20.0, 30.0]);
    let res = &a + &b;
    assert_tensor_1d(&res, &[11.0, 22.0, 33.0]);
}

#[test]
fn test_1d_plus_1d_commutativity() {
    let a = Tensor::from(vec![1.5, 2.5, 3.5]);
    let b = Tensor::from(vec![4.0, 5.0, 6.0]);
    let res1 = &a + &b;
    let res2 = &b + &a;
    assert_tensor_1d(&res1, &[5.5, 7.5, 9.5]);
    assert_tensor_1d(&res2, &[5.5, 7.5, 9.5]);
}

#[test]
#[should_panic]
fn test_1d_plus_1d_dimension_mismatch_panics() {
    let a = Tensor::from(vec![1.0, 2.0]);
    let b = Tensor::from(vec![1.0, 2.0, 3.0]);
    let _ = a + b;
}

// 1D + f32 scalar (Left)
#[test]
fn test_1d_plus_f32_scalar_left_owned() {
    let a = Tensor::from(vec![1.0, 2.0, 3.0]);
    let res = a + 10.0_f32;
    assert_tensor_1d(&res, &[11.0, 12.0, 13.0]);
}

#[test]
fn test_1d_plus_f32_scalar_left_ref() {
    let a = Tensor::from(vec![1.0, 2.0, 3.0]);
    let res = &a + 10.0_f32;
    assert_tensor_1d(&res, &[11.0, 12.0, 13.0]);
}

// f32 scalar + 1D (Right)
#[test]
fn test_f32_scalar_plus_1d_right_owned() {
    let a = Tensor::from(vec![1.0, 2.0, 3.0]);
    let res = 10.0_f32 + a;
    assert_tensor_1d(&res, &[11.0, 12.0, 13.0]);
}

#[test]
fn test_f32_scalar_plus_1d_right_ref() {
    let a = Tensor::from(vec![1.0, 2.0, 3.0]);
    let res = 10.0_f32 + &a;
    assert_tensor_1d(&res, &[11.0, 12.0, 13.0]);
}

// =========================================================================
// 3. 2D Tensor Addition Tests
// =========================================================================

#[test]
fn test_2d_plus_2d_same_shape_owned() {
    let a = Tensor::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let b = Tensor::from(vec![vec![10.0, 20.0], vec![30.0, 40.0]]);
    let res = a + b;
    assert_tensor_2d(&res, &[&[11.0, 22.0], &[33.0, 44.0]]);
}

#[test]
fn test_2d_plus_2d_same_shape_ref() {
    let a = Tensor::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let b = Tensor::from(vec![vec![10.0, 20.0], vec![30.0, 40.0]]);
    let res = &a + &b;
    assert_tensor_2d(&res, &[&[11.0, 22.0], &[33.0, 44.0]]);
}

#[test]
fn test_2d_plus_2d_commutativity() {
    let a = Tensor::from(vec![vec![1.0, 5.0], vec![2.0, 6.0]]);
    let b = Tensor::from(vec![vec![3.0, 7.0], vec![4.0, 8.0]]);
    let res1 = &a + &b;
    let res2 = &b + &a;
    assert_tensor_2d(&res1, &[&[4.0, 12.0], &[6.0, 14.0]]);
    assert_tensor_2d(&res2, &[&[4.0, 12.0], &[6.0, 14.0]]);
}

#[test]
#[should_panic]
fn test_2d_plus_2d_incompatible_shapes_panics_different_rows() {
    let a = Tensor::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]); // [2, 2]
    let b = Tensor::from(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]); // [3, 2]
    let _ = a + b;
}

#[test]
#[should_panic]
fn test_2d_plus_2d_incompatible_shapes_panics_different_cols() {
    let a = Tensor::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]); // [2, 2]
    let b = Tensor::from(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]); // [2, 3]
    let _ = a + b;
}

// 2D + f32 scalar (Left)
#[test]
fn test_2d_plus_f32_scalar_left_owned() {
    let a = Tensor::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let res = a + 10.0_f32;
    assert_tensor_2d(&res, &[&[11.0, 12.0], &[13.0, 14.0]]);
}

#[test]
fn test_2d_plus_f32_scalar_left_ref() {
    let a = Tensor::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let res = &a + 10.0_f32;
    assert_tensor_2d(&res, &[&[11.0, 12.0], &[13.0, 14.0]]);
}

// f32 scalar + 2D (Right)
#[test]
fn test_f32_scalar_plus_2d_right_owned() {
    let a = Tensor::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let res = 10.0_f32 + a;
    assert_tensor_2d(&res, &[&[11.0, 12.0], &[13.0, 14.0]]);
}

#[test]
fn test_f32_scalar_plus_2d_right_ref() {
    let a = Tensor::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let res = 10.0_f32 + &a;
    assert_tensor_2d(&res, &[&[11.0, 12.0], &[13.0, 14.0]]);
}

// =========================================================================
// 4. Numerical Edge Cases (Identity & Negative values)
// =========================================================================

#[test]
fn test_add_zero_identity() {
    let a_0d = Tensor::from(42.0);
    let res_0d = &a_0d + 0.0_f32;
    assert_tensor_0d(&res_0d, 42.0);

    let a_1d = Tensor::from(vec![1.0, -2.0, 3.0]);
    let res_1d = &a_1d + 0.0_f32;
    assert_tensor_1d(&res_1d, &[1.0, -2.0, 3.0]);

    let a_2d = Tensor::from(vec![vec![1.0, -2.0], vec![3.0, -4.0]]);
    let res_2d = &a_2d + 0.0_f32;
    assert_tensor_2d(&res_2d, &[&[1.0, -2.0], &[3.0, -4.0]]);
}

#[test]
fn test_add_negative_and_cancellation() {
    let a = Tensor::from(vec![5.0, -10.0, 2.5]);
    let b = Tensor::from(vec![-5.0, 10.0, -2.5]);
    let res = &a + &b;
    assert_tensor_1d(&res, &[0.0, 0.0, 0.0]);
}
