// tests/test_elemmul.rs
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

fn assert_tensor_nd(t: &Tensor, expected_shape: &[usize], expected_flat: &[f32]) {
    assert_eq!(
        t.shape().as_slice(),
        expected_shape,
        "N-D tensor shape mismatch"
    );
    let expected_len = if expected_shape.is_empty() {
        1
    } else {
        expected_shape.iter().product()
    };
    assert_eq!(
        expected_flat.len(),
        expected_len,
        "Expected flat length does not match expected shape"
    );

    let mut coords = vec![0; expected_shape.len()];
    for (flat_idx, &val) in expected_flat.iter().enumerate() {
        let mut rem = flat_idx;
        for d in (0..expected_shape.len()).rev() {
            coords[d] = rem % expected_shape[d];
            rem /= expected_shape[d];
        }
        assert_eq!(
            t.at(&coords),
            val,
            "N-D tensor element mismatch at coords {:?}",
            coords
        );
    }
}

// =========================================================================
// 1. 0D (Scalar) Element-Wise Multiplication Tests (^)
// =========================================================================

#[test]
fn test_0d_elemmul_0d_ref() {
    let a = Tensor::from(2.5);
    let b = Tensor::from(4.0);
    let res = &a ^ &b;
    assert_tensor_0d(&res, 10.0);
}

#[test]
fn test_0d_elemmul_0d_owned() {
    let a = Tensor::from(3.0);
    let b = Tensor::from(4.0);
    let res = a ^ b;
    assert_tensor_0d(&res, 12.0);
}

#[test]
fn test_0d_elemmul_0d_ref_owned() {
    let a = Tensor::from(3.0);
    let b = Tensor::from(4.0);
    let res = &a ^ b;
    assert_tensor_0d(&res, 12.0);
}

#[test]
fn test_0d_elemmul_0d_owned_ref() {
    let a = Tensor::from(3.0);
    let b = Tensor::from(4.0);
    let res = a ^ &b;
    assert_tensor_0d(&res, 12.0);
}

// =========================================================================
// 2. 1D Element-Wise Multiplication Tests (^)
// =========================================================================

#[test]
fn test_1d_elemmul_1d_ref() {
    let a = Tensor::from(vec![2.0, -3.0, 4.0]);
    let b = Tensor::from(vec![5.0, 2.0, -0.5]);
    let res = &a ^ &b;
    assert_tensor_1d(&res, &[10.0, -6.0, -2.0]);
}

#[test]
fn test_1d_elemmul_1d_owned() {
    let a = Tensor::from(vec![1.0, 2.0, 3.0]);
    let b = Tensor::from(vec![4.0, 5.0, 6.0]);
    let res = a ^ b;
    assert_tensor_1d(&res, &[4.0, 10.0, 18.0]);
}

#[test]
fn test_1d_elemmul_1d_ref_owned() {
    let a = Tensor::from(vec![1.0, 2.0, 3.0]);
    let b = Tensor::from(vec![4.0, 5.0, 6.0]);
    let res = &a ^ b;
    assert_tensor_1d(&res, &[4.0, 10.0, 18.0]);
}

#[test]
fn test_1d_elemmul_1d_owned_ref() {
    let a = Tensor::from(vec![1.0, 2.0, 3.0]);
    let b = Tensor::from(vec![4.0, 5.0, 6.0]);
    let res = a ^ &b;
    assert_tensor_1d(&res, &[4.0, 10.0, 18.0]);
}

#[test]
fn test_1d_elemmul_1d_commutativity() {
    let a = Tensor::from(vec![1.5, 2.5, 3.5]);
    let b = Tensor::from(vec![4.0, 5.0, 6.0]);
    let res1 = &a ^ &b;
    let res2 = &b ^ &a;
    assert_tensor_1d(&res1, &[6.0, 12.5, 21.0]);
    assert_tensor_1d(&res2, &[6.0, 12.5, 21.0]);
}

#[test]
#[should_panic(expected = "Error! Can not perform element-wise multiplication on tensors of different shapes")]
fn test_1d_elemmul_1d_dimension_mismatch_panics() {
    let a = Tensor::from(vec![1.0, 2.0]);
    let b = Tensor::from(vec![1.0, 2.0, 3.0]);
    let _ = a ^ b;
}

// =========================================================================
// 3. 2D Element-Wise Multiplication Tests (^)
// =========================================================================

#[test]
fn test_2d_elemmul_2d_same_shape_ref() {
    let a = Tensor::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let b = Tensor::from(vec![vec![10.0, 20.0], vec![30.0, 40.0]]);
    let res = &a ^ &b;
    assert_tensor_2d(&res, &[&[10.0, 40.0], &[90.0, 160.0]]);
}

#[test]
fn test_2d_elemmul_2d_same_shape_owned() {
    let a = Tensor::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let b = Tensor::from(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
    let res = a ^ b;
    assert_tensor_2d(&res, &[&[5.0, 12.0], &[21.0, 32.0]]);
}

#[test]
fn test_2d_elemmul_2d_same_shape_ref_owned() {
    let a = Tensor::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let b = Tensor::from(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
    let res = &a ^ b;
    assert_tensor_2d(&res, &[&[5.0, 12.0], &[21.0, 32.0]]);
}

#[test]
fn test_2d_elemmul_2d_same_shape_owned_ref() {
    let a = Tensor::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let b = Tensor::from(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
    let res = a ^ &b;
    assert_tensor_2d(&res, &[&[5.0, 12.0], &[21.0, 32.0]]);
}

#[test]
fn test_2d_elemmul_2d_commutativity() {
    let a = Tensor::from(vec![vec![1.0, 5.0], vec![2.0, 6.0]]);
    let b = Tensor::from(vec![vec![3.0, 7.0], vec![4.0, 8.0]]);
    let res1 = &a ^ &b;
    let res2 = &b ^ &a;
    assert_tensor_2d(&res1, &[&[3.0, 35.0], &[8.0, 48.0]]);
    assert_tensor_2d(&res2, &[&[3.0, 35.0], &[8.0, 48.0]]);
}

#[test]
#[should_panic(expected = "Error! Can not perform element-wise multiplication on tensors of different shapes")]
fn test_2d_elemmul_2d_incompatible_shapes_panics_different_rows() {
    let a = Tensor::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]); // [2, 2]
    let b = Tensor::from(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]); // [3, 2]
    let _ = a ^ b;
}

#[test]
#[should_panic(expected = "Error! Can not perform element-wise multiplication on tensors of different shapes")]
fn test_2d_elemmul_2d_incompatible_shapes_panics_different_cols() {
    let a = Tensor::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]); // [2, 2]
    let b = Tensor::from(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]); // [2, 3]
    let _ = a ^ b;
}

// =========================================================================
// 4. Higher-Dimensional (3D & 4D) Element-Wise Multiplication Tests (^)
// =========================================================================

#[test]
fn test_3d_elemmul_3d_ref() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3]);
    let b = Tensor::new(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0], vec![1, 2, 3]);
    let res = &a ^ &b;

    let expected = vec![2.0, 6.0, 12.0, 20.0, 30.0, 42.0];
    assert_tensor_nd(&res, &[1, 2, 3], &expected);
}

#[test]
fn test_3d_elemmul_3d_owned() {
    let data_a: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let data_b: Vec<f32> = vec![2.0; 12];
    let a = Tensor::new(data_a, vec![2, 2, 3]);
    let b = Tensor::new(data_b, vec![2, 2, 3]);
    let res = a ^ b;

    let expected: Vec<f32> = (1..=12).map(|x| (x * 2) as f32).collect();
    assert_tensor_nd(&res, &[2, 2, 3], &expected);
}

#[test]
fn test_3d_elemmul_3d_ref_owned() {
    let data_a: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let data_b: Vec<f32> = vec![2.0; 12];
    let a = Tensor::new(data_a, vec![2, 2, 3]);
    let b = Tensor::new(data_b, vec![2, 2, 3]);
    let res = &a ^ b;

    let expected: Vec<f32> = (1..=12).map(|x| (x * 2) as f32).collect();
    assert_tensor_nd(&res, &[2, 2, 3], &expected);
}

#[test]
#[should_panic(expected = "Error! Can not perform element-wise multiplication on tensors of different shapes")]
fn test_3d_elemmul_3d_shape_mismatch_panics() {
    let a = Tensor::new(vec![1.0; 12], vec![2, 2, 3]);
    let b = Tensor::new(vec![1.0; 12], vec![2, 3, 2]);
    let _ = a ^ b;
}

#[test]
fn test_4d_elemmul_4d_ref() {
    let data_a: Vec<f32> = (0..16).map(|x| x as f32).collect();
    let data_b: Vec<f32> = (0..16).map(|x| (x * 2) as f32).collect();
    let a = Tensor::new(data_a.clone(), vec![2, 2, 2, 2]);
    let b = Tensor::new(data_b.clone(), vec![2, 2, 2, 2]);
    let res = &a ^ &b;

    let expected: Vec<f32> = data_a.iter().zip(&data_b).map(|(x, y)| x * y).collect();
    assert_tensor_nd(&res, &[2, 2, 2, 2], &expected);
}

#[test]
fn test_4d_elemmul_4d_owned() {
    let data_a: Vec<f32> = (0..16).map(|x| x as f32).collect();
    let data_b: Vec<f32> = (0..16).map(|x| (x * 2) as f32).collect();
    let a = Tensor::new(data_a.clone(), vec![2, 2, 2, 2]);
    let b = Tensor::new(data_b.clone(), vec![2, 2, 2, 2]);
    let res = a ^ b;

    let expected: Vec<f32> = data_a.iter().zip(&data_b).map(|(x, y)| x * y).collect();
    assert_tensor_nd(&res, &[2, 2, 2, 2], &expected);
}

#[test]
fn test_chaining_elemmul() {
    // (A ^ B) ^ C chaining temporaries
    let a = Tensor::from(vec![2.0, 3.0]);
    let b = Tensor::from(vec![4.0, 5.0]);
    let c = Tensor::from(vec![2.0, 2.0]);
    let res = (a ^ b) ^ c;
    assert_tensor_1d(&res, &[16.0, 30.0]);
}

// =========================================================================
// 5. Numerical Edge Cases (Identity, Zero, Signs)
// =========================================================================

#[test]
fn test_elemmul_ones_identity() {
    let a_1d = Tensor::from(vec![1.0, -2.0, 3.5]);
    let ones = Tensor::from(vec![1.0, 1.0, 1.0]);
    let res = &a_1d ^ &ones;
    assert_tensor_1d(&res, &[1.0, -2.0, 3.5]);

    let a_2d = Tensor::from(vec![vec![1.0, -2.0], vec![3.0, -4.0]]);
    let ones_2d = Tensor::from(vec![vec![1.0, 1.0], vec![1.0, 1.0]]);
    let res_2d = &a_2d ^ &ones_2d;
    assert_tensor_2d(&res_2d, &[&[1.0, -2.0], &[3.0, -4.0]]);
}

#[test]
fn test_elemmul_zero_property() {
    let a_1d = Tensor::from(vec![10.0, -20.0, 30.0]);
    let zeros = Tensor::from(vec![0.0, 0.0, 0.0]);
    let res = &a_1d ^ &zeros;
    assert_tensor_1d(&res, &[0.0, 0.0, 0.0]);
}

#[test]
fn test_elemmul_negative_numbers() {
    let a = Tensor::from(vec![-2.0, 3.0, -4.0]);
    let b = Tensor::from(vec![-5.0, -6.0, 7.0]);
    let res = &a ^ &b;
    assert_tensor_1d(&res, &[10.0, -18.0, -28.0]);
}
