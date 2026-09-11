// tests/test_matmul.rs
use NeuralNetworksRust::tensor::Tensor;

// =========================================================================
// Helper assertion functions to inspect tensor shape and elements
// =========================================================================

fn assert_dot_product(t: &Tensor, expected: f32) {
    assert!(
        t.shape().as_slice() == &[1] || t.shape().as_slice() == &[] as &[usize],
        "Dot product shape expected to be [1] or [], got {:?}",
        t.shape()
    );
    assert_eq!(t.item(), expected, "Dot product value mismatch");
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

    // Verify each element by calculating coordinates
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
// 0. 0D Matmul Error Tests
// =========================================================================

#[test]
#[should_panic(expected = "Error! Can not perform matmul on 0D tensors")]
fn test_0d_matmul_panics() {
    let a = Tensor::from(2.0);
    let b = Tensor::from(3.0);
    let _ = &a * &b;
}

// =========================================================================
// 1. 1D x 1D Matrix Multiplication Tests (Vector Dot Product)
// =========================================================================

#[test]
fn test_1d_matmul_1d_owned() {
    let a = Tensor::from(vec![1.0, 2.0, 3.0]);
    let b = Tensor::from(vec![4.0, 5.0, 6.0]);
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32.0
    let res = a * b;
    assert_dot_product(&res, 32.0);
}

#[test]
fn test_1d_matmul_1d_ref() {
    let a = Tensor::from(vec![2.0, -3.0, 4.0]);
    let b = Tensor::from(vec![5.0, 2.0, -1.0]);
    // 2*5 + (-3)*2 + 4*(-1) = 10 - 6 - 4 = 0.0
    let res = &a * &b;
    assert_dot_product(&res, 0.0);
}

#[test]
#[should_panic(expected = "Error! Can not perform matmul on tensors of different shapes")]
fn test_1d_matmul_1d_dimension_mismatch_panics() {
    let a = Tensor::from(vec![1.0, 2.0]);
    let b = Tensor::from(vec![1.0, 2.0, 3.0]);
    let _ = a * b;
}

// =========================================================================
// 2. 1D x 2D Matrix Multiplication Tests (Vector x Matrix -> [N])
// =========================================================================

#[test]
fn test_1d_matmul_2d_owned() {
    // [K] = [2]
    let a = Tensor::from(vec![1.0, 2.0]);
    // [K, N] = [2, 3]
    let b = Tensor::from(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
    ]);
    // [1, 2] * [[1, 2, 3], [4, 5, 6]] = [1*1+2*4, 1*2+2*5, 1*3+2*6] = [9.0, 12.0, 15.0]
    let res = a * b;
    assert_tensor_1d(&res, &[9.0, 12.0, 15.0]);
}

#[test]
fn test_1d_matmul_2d_ref() {
    let a = Tensor::from(vec![2.0, 3.0]);
    let b = Tensor::from(vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
    ]);
    let res = &a * &b;
    assert_tensor_1d(&res, &[2.0, 3.0]);
}

#[test]
#[should_panic(expected = "Error! Can not perform matmul on tensors of different shapes")]
fn test_1d_matmul_2d_dimension_mismatch_panics() {
    let a = Tensor::from(vec![1.0, 2.0, 3.0]); // len 3
    let b = Tensor::from(vec![                 // [2, 3] -> rows = 2 != 3
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
    ]);
    let _ = a * b;
}

// =========================================================================
// 3. 2D x 1D Matrix Multiplication Tests (Matrix x Vector -> [M])
// =========================================================================

#[test]
fn test_2d_matmul_1d_owned() {
    // [M, K] = [2, 3]
    let a = Tensor::from(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
    ]);
    // [K] = [3]
    let b = Tensor::from(vec![1.0, 2.0, 3.0]);
    // Row 0: 1*1 + 2*2 + 3*3 = 14.0
    // Row 1: 4*1 + 5*2 + 6*3 = 32.0
    let res = a * b;
    assert_tensor_1d(&res, &[14.0, 32.0]);
}

#[test]
fn test_2d_matmul_1d_ref() {
    let a = Tensor::from(vec![
        vec![2.0, 0.0],
        vec![0.0, 3.0],
    ]);
    let b = Tensor::from(vec![4.0, 5.0]);
    let res = &a * &b;
    assert_tensor_1d(&res, &[8.0, 15.0]);
}

#[test]
#[should_panic(expected = "Error! Can not perform matmul on tensors of different shapes")]
fn test_2d_matmul_1d_dimension_mismatch_panics() {
    let a = Tensor::from(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
    ]); // cols = 3
    let b = Tensor::from(vec![1.0, 2.0]); // len = 2 != 3
    let _ = a * b;
}

// =========================================================================
// 4. 2D x 2D Matrix Multiplication Tests (Matrix x Matrix -> [M, N])
// =========================================================================

#[test]
fn test_2d_matmul_2d_square_owned() {
    let a = Tensor::from(vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
    ]);
    let b = Tensor::from(vec![
        vec![5.0, 6.0],
        vec![7.0, 8.0],
    ]);
    // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    let res = a * b;
    assert_tensor_2d(&res, &[&[19.0, 22.0], &[43.0, 50.0]]);
}

#[test]
fn test_2d_matmul_2d_rectangular_ref() {
    // [2, 3] * [3, 2] -> [2, 2]
    let a = Tensor::from(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
    ]);
    let b = Tensor::from(vec![
        vec![7.0, 8.0],
        vec![9.0, 1.0],
        vec![2.0, 3.0],
    ]);
    // Row 0: [1*7+2*9+3*2, 1*8+2*1+3*3] = [31.0, 19.0]
    // Row 1: [4*7+5*9+6*2, 4*8+5*1+6*3] = [85.0, 55.0]
    let res = &a * &b;
    assert_tensor_2d(&res, &[&[31.0, 19.0], &[85.0, 55.0]]);
}

#[test]
fn test_2d_matmul_2d_rectangular_tall_by_wide() {
    // [3, 2] * [2, 4] -> [3, 4]
    let a = Tensor::from(vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
        vec![5.0, 6.0],
    ]);
    let b = Tensor::from(vec![
        vec![1.0, 0.0, 2.0, 1.0],
        vec![0.0, 1.0, 1.0, 2.0],
    ]);
    let res = &a * &b;
    assert_tensor_2d(&res, &[
        &[1.0, 2.0, 4.0, 5.0],
        &[3.0, 4.0, 10.0, 11.0],
        &[5.0, 6.0, 16.0, 17.0],
    ]);
}

#[test]
fn test_2d_matmul_identity_property() {
    let a = Tensor::from(vec![
        vec![2.5, -1.0],
        vec![3.0, 4.5],
    ]);
    let identity = Tensor::from(vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
    ]);
    let res = &a * &identity;
    assert_tensor_2d(&res, &[&[2.5, -1.0], &[3.0, 4.5]]);
}

#[test]
#[should_panic(expected = "Error! Can not perform matmul on tensors of different shapes")]
fn test_2d_matmul_2d_inner_dim_mismatch_panics() {
    let a = Tensor::from(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
    ]); // [2, 3]
    let b = Tensor::from(vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
    ]); // [2, 2] -> cols of A (3) != rows of B (2)
    let _ = a * b;
}

// =========================================================================
// 5. Batched Matmul Tests: 3D x 2D (Broadcasting 2D across 3D batch)
// =========================================================================

#[test]
fn test_3d_matmul_2d_broadcast_owned() {
    // Shape [2, 2, 3] * [3, 2] -> [2, 2, 2]
    // A: 2 batches of (2x3)
    let a_data = vec![
        // Batch 0
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        // Batch 1
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
    ];
    let a = Tensor::new(a_data, vec![2, 2, 3]);

    // B: single (3x2)
    let b = Tensor::from(vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ]);

    let res = a * b;
    let expected = vec![
        // Batch 0: [1*1+3*1, 2*1+3*1], [4*1+6*1, 5*1+6*1]
        4.0, 5.0, 10.0, 11.0,
        // Batch 1: [7*1+9*1, 8*1+9*1], [10*1+12*1, 11*1+12*1]
        16.0, 17.0, 22.0, 23.0,
    ];
    assert_tensor_nd(&res, &[2, 2, 2], &expected);
}

#[test]
fn test_3d_matmul_2d_broadcast_ref() {
    let a = Tensor::new(vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
    ], vec![2, 2, 3]);

    let b = Tensor::from(vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ]);

    let res = &a * &b;
    let expected = vec![
        4.0, 5.0, 10.0, 11.0,
        16.0, 17.0, 22.0, 23.0,
    ];
    assert_tensor_nd(&res, &[2, 2, 2], &expected);
}

// =========================================================================
// 6. Batched Matmul Tests: 2D x 3D (Broadcasting 2D across 3D batch)
// =========================================================================

#[test]
fn test_2d_matmul_3d_broadcast_ref() {
    // Shape [2, 3] * [2, 3, 2] -> [2, 2, 2]
    let a = Tensor::from(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
    ]);

    let b_data = vec![
        // Batch 0
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        // Batch 1
        2.0, 0.0,
        0.0, 2.0,
        1.0, 1.0,
    ];
    let b = Tensor::new(b_data, vec![2, 3, 2]);

    let res = &a * &b;
    let expected = vec![
        // Batch 0: A * B[0]
        4.0, 5.0, 10.0, 11.0,
        // Batch 1: A * B[1]
        // [1*2+3*1, 2*2+3*1] = [5.0, 7.0]
        // [4*2+6*1, 5*2+6*1] = [14.0, 16.0]
        5.0, 7.0, 14.0, 16.0,
    ];
    assert_tensor_nd(&res, &[2, 2, 2], &expected);
}

// =========================================================================
// 7. Batched Matmul Tests: 3D x 3D (Batch x Batch)
// =========================================================================

#[test]
fn test_3d_matmul_3d_same_batch_owned() {
    // [2, 2, 2] * [2, 2, 2] -> [2, 2, 2]
    let a_data = vec![
        // Batch 0
        1.0, 2.0,
        3.0, 4.0,
        // Batch 1
        2.0, 0.0,
        1.0, 3.0,
    ];
    let b_data = vec![
        // Batch 0
        5.0, 6.0,
        7.0, 8.0,
        // Batch 1
        1.0, 2.0,
        3.0, 4.0,
    ];
    let a = Tensor::new(a_data, vec![2, 2, 2]);
    let b = Tensor::new(b_data, vec![2, 2, 2]);

    let res = a * b;
    let expected = vec![
        // Batch 0: [[1, 2], [3, 4]] * [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        19.0, 22.0, 43.0, 50.0,
        // Batch 1: [[2, 0], [1, 3]] * [[1, 2], [3, 4]] = [[2, 4], [10, 14]]
        2.0, 4.0, 10.0, 14.0,
    ];
    assert_tensor_nd(&res, &[2, 2, 2], &expected);
}

#[test]
#[should_panic(expected = "Error! Can not perform matmul on tensors of different shapes")]
fn test_3d_matmul_3d_batch_mismatch_panics() {
    // Batch 2 vs Batch 3
    let a = Tensor::new(vec![1.0; 8], vec![2, 2, 2]);
    let b = Tensor::new(vec![1.0; 12], vec![3, 2, 2]);
    let _ = a * b;
}

// =========================================================================
// 8. Higher-Dimensional (4D) Matmul Tests (Multi-head Attention Style)
// =========================================================================

#[test]
fn test_4d_matmul_4d_ref() {
    // Shape [2, 2, 2, 2] * [2, 2, 2, 2] -> [2, 2, 2, 2]
    // 4 matrices of 2x2, each multiplied by 2x2 identity
    let data_a: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let identity_2x2 = vec![
        1.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 1.0,
    ];
    let a = Tensor::new(data_a.clone(), vec![2, 2, 2, 2]);
    let b = Tensor::new(identity_2x2, vec![2, 2, 2, 2]);

    let res = &a * &b;
    // Multiplying by identity leaves each matrix untouched
    assert_tensor_nd(&res, &[2, 2, 2, 2], &data_a);
}

#[test]
fn test_4d_matmul_2d_broadcast() {
    // Shape [2, 2, 2, 2] * [2, 2] -> [2, 2, 2, 2]
    let data_a: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let a = Tensor::new(data_a.clone(), vec![2, 2, 2, 2]);
    let identity = Tensor::from(vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
    ]);

    let res = &a * &identity;
    assert_tensor_nd(&res, &[2, 2, 2, 2], &data_a);
}

// =========================================================================
// 9. Scalar Multiplication Tests (*)
// =========================================================================

#[test]
fn test_scalar_mul_tensor_owned() {
    let a = Tensor::from(vec![1.0, 2.0, 3.0]);
    let res = a * 2.0_f32;
    assert_tensor_1d(&res, &[2.0, 4.0, 6.0]);

    let b = Tensor::from(vec![1.0, 2.0, 3.0]);
    let res2 = 3.0_f32 * b;
    assert_tensor_1d(&res2, &[3.0, 6.0, 9.0]);
}

#[test]
fn test_scalar_mul_tensor_ref() {
    let a = Tensor::from(vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
    ]);
    let res = &a * 2.0_f32;
    assert_tensor_2d(&res, &[&[2.0, 4.0], &[6.0, 8.0]]);

    let res2 = 0.5_f32 * &a;
    assert_tensor_2d(&res2, &[&[0.5, 1.0], &[1.5, 2.0]]);
}

// =========================================================================
// 10. Numerical Properties & Edge Cases
// =========================================================================

#[test]
fn test_matmul_zero_matrix() {
    let a = Tensor::from(vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
    ]);
    let zeros = Tensor::from(vec![
        vec![0.0, 0.0],
        vec![0.0, 0.0],
    ]);
    let res = &a * &zeros;
    assert_tensor_2d(&res, &[&[0.0, 0.0], &[0.0, 0.0]]);
}

#[test]
fn test_matmul_non_commutative() {
    let a = Tensor::from(vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
    ]);
    let b = Tensor::from(vec![
        vec![0.0, 1.0],
        vec![1.0, 0.0],
    ]);
    let ab = &a * &b; // [[2, 1], [4, 3]]
    let ba = &b * &a; // [[3, 4], [1, 2]]

    assert_tensor_2d(&ab, &[&[2.0, 1.0], &[4.0, 3.0]]);
    assert_tensor_2d(&ba, &[&[3.0, 4.0], &[1.0, 2.0]]);
}
