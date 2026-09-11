// tests/test_tensor.rs
use NeuralNetworksRust::tensor::Tensor;

// ==========================================
// 0D (Scalar) Tests
// ==========================================

#[test]
fn test_0d_scalar_item_and_item_mut() {
    let t = Tensor::from(42.0);
    assert_eq!(t.item(), 42.0);
    assert_eq!(t.shape(), vec![]);
    assert_eq!(t.stride(), vec![]);

    *t.item_mut() = 99.0;
    assert_eq!(t.item(), 99.0);
    assert_eq!(t.at(&[]), 99.0);
}

#[test]
#[should_panic(expected = "Expected 0 indices, got 1")]
fn test_0d_scalar_wrong_indices_count() {
    let t = Tensor::from(42.0);
    t.at(&[0]);
}

// ==========================================
// 1D (Vector) Tests
// ==========================================

#[test]
fn test_1d_tensor_indexing_and_mutation() {
    let t = Tensor::from(vec![10.0, 20.0, 30.0]);
    assert_eq!(t.shape(), vec![3]);
    assert_eq!(t.stride(), vec![1]);
    assert_eq!(t.at(&[1]), 20.0);

    *t.at_mut(&[1]) = 25.0;
    assert_eq!(t.at(&[1]), 25.0);
}

#[test]
#[should_panic(expected = "Expected 1 element, got 3")]
fn test_1d_item_fails_when_multiple_elements() {
    let t = Tensor::from(vec![1.0, 2.0, 3.0]);
    t.item();
}

#[test]
fn test_1d_single_element_item() {
    let t = Tensor::from(vec![7.5]);
    assert_eq!(t.item(), 7.5);
}

// ==========================================
// 2D (Matrix) Tests
// ==========================================

#[test]
fn test_2d_tensor_indexing() {
    let x = Tensor::from(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    assert_eq!(x.at(&[0, 0]), 1.0);
    assert_eq!(x.at(&[0, 2]), 3.0);
    assert_eq!(x.at(&[1, 2]), 6.0);

    *x.at_mut(&[0, 0]) = 10.0;
    assert_eq!(x.at(&[0, 0]), 10.0);
    *x.at_mut(&[1, 1]) = 50.0;
    assert_eq!(x.at(&[1, 1]), 50.0);
}

#[test]
#[should_panic(expected = "Index out of bounds")]
fn test_2d_out_of_bounds_row() {
    let x = Tensor::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    x.at(&[2, 0]); // Row 2 is out of bounds for 2x2
}

#[test]
#[should_panic(expected = "Index out of bounds")]
fn test_2d_out_of_bounds_col() {
    let x = Tensor::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    x.at(&[0, 2]); // Col 2 is out of bounds for 2x2
}

#[test]
#[should_panic(expected = "Expected 2 indices, got 3")]
fn test_2d_wrong_dim_count() {
    let x = Tensor::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    x.at(&[0, 0, 0]);
}

// ==========================================
// 3D Tensor Tests
// ==========================================

#[test]
fn test_3d_stride_calculation() {
    // 2 x 3 x 4 -> 24 elements total
    let x = Tensor::new(vec![0.0; 24], vec![2, 3, 4]);
    assert_eq!(x.shape(), vec![2, 3, 4]);
    assert_eq!(x.stride(), vec![12, 4, 1]);
}

#[test]
fn test_3d_indexing_and_mutation() {
    // Fill with 0.0 .. 23.0 sequentially
    let data: Vec<f32> = (0..24).map(|n| n as f32).collect();
    let x = Tensor::new(data, vec![2, 3, 4]);

    // flat_index = i * 12 + j * 4 + k * 1
    // (0, 0, 0) -> 0
    assert_eq!(x.at(&[0, 0, 0]), 0.0);
    // (0, 1, 2) -> 0*12 + 1*4 + 2*1 = 6
    assert_eq!(x.at(&[0, 1, 2]), 6.0);
    // (1, 2, 3) -> 1*12 + 2*4 + 3*1 = 23
    assert_eq!(x.at(&[1, 2, 3]), 23.0);

    // Mutate an element and ensure neighbors are untouched
    *x.at_mut(&[0, 1, 2]) = 999.0;
    assert_eq!(x.at(&[0, 1, 2]), 999.0);
    assert_eq!(x.at(&[0, 1, 1]), 5.0);
    assert_eq!(x.at(&[0, 1, 3]), 7.0);
}

#[test]
#[should_panic(expected = "Index out of bounds")]
fn test_3d_index_out_of_bounds() {
    let x = Tensor::new(vec![0.0; 24], vec![2, 3, 4]);
    x.at(&[2, 0, 0]); // Dim 0 out of bounds
}

#[test]
#[should_panic(expected = "Expected 3 indices, got 2")]
fn test_3d_too_few_indices() {
    let x = Tensor::new(vec![0.0; 24], vec![2, 3, 4]);
    x.at(&[0, 0]);
}

// ==========================================
// 4D Tensor Tests (e.g. Batch x Channel x Height x Width)
// ==========================================

#[test]
fn test_4d_stride_calculation() {
    // 2 x 3 x 4 x 5 -> 120 elements
    let x = Tensor::new(vec![0.0; 120], vec![2, 3, 4, 5]);
    assert_eq!(x.shape(), vec![2, 3, 4, 5]);
    // strides: [3*4*5, 4*5, 5, 1] = [60, 20, 5, 1]
    assert_eq!(x.stride(), vec![60, 20, 5, 1]);
}

#[test]
fn test_4d_indexing_and_mutation() {
    let data: Vec<f32> = (0..120).map(|n| n as f32).collect();
    let x = Tensor::new(data, vec![2, 3, 4, 5]);

    // (1, 2, 3, 4) -> 1*60 + 2*20 + 3*5 + 4*1 = 60 + 40 + 15 + 4 = 119
    assert_eq!(x.at(&[1, 2, 3, 4]), 119.0);

    *x.at_mut(&[1, 2, 3, 4]) = -42.0;
    assert_eq!(x.at(&[1, 2, 3, 4]), -42.0);
}

#[test]
#[should_panic(expected = "Index out of bounds")]
fn test_4d_out_of_bounds() {
    let x = Tensor::new(vec![0.0; 120], vec![2, 3, 4, 5]);
    x.at(&[0, 3, 0, 0]); // Dim 1 is size 3, index 3 is out of bounds
}

// ==========================================
// Display / Formatting Tests
// ==========================================

#[test]
fn test_display_output_formatting() {
    let t_0d = Tensor::from(42.0);
    assert_eq!(format!("{}", t_0d), "42");

    let t_1d = Tensor::from(vec![1.0, 2.0, 3.0]);
    assert_eq!(format!("{}", t_1d), "[1,2,3]");

    let t_2d = Tensor::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    assert_eq!(format!("{}", t_2d), "[[1,2],\n [3,4]]");
}
