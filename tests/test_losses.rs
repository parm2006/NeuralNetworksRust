use NeuralNetworksRust::losses::{CrossEntropyLoss, Loss, MSELoss, NLLLoss};
use NeuralNetworksRust::modules::{Module, Softmax};
use NeuralNetworksRust::tensor::Tensor;

const EPSILON: f32 = 1e-4;
const TOLERANCE: f32 = 1e-3;

// =========================================================================
// 1. Analytical Forward & Backward Verification
// =========================================================================

#[test]
fn test_mse_loss_forward_1d_analytical() {
    let criterion = MSELoss::new();
    let preds = Tensor::new(vec![2.0, 4.0, -1.0, 3.0], vec![4]).with_requires_grad();
    let targets = Tensor::new(vec![1.0, 5.0, -1.0, 1.0], vec![4]);

    let loss = criterion.forward(&preds, &targets);
    assert_eq!(loss.shape(), vec![1]);

    // diffs: [1.0, -1.0, 0.0, 2.0]
    // sq diffs: [1.0, 1.0, 0.0, 4.0] -> sum = 6.0, N = 4 -> mean = 1.5
    let val = loss.item();
    assert!(
        (val - 1.5).abs() < 1e-5,
        "Expected MSE 1.5, got {}",
        val
    );

    loss.backward();
    // dL/d(preds) = 2/N * (preds - targets) = 2/4 * [1.0, -1.0, 0.0, 2.0] = [0.5, -0.5, 0.0, 1.0]
    let grads = preds.grad().expect("Gradients should be present");
    let expected_grads = vec![0.5, -0.5, 0.0, 1.0];
    for (i, (&g, &exp)) in grads.iter().zip(expected_grads.iter()).enumerate() {
        assert!(
            (g - exp).abs() < 1e-5,
            "Gradient mismatch at {}: got {}, expected {}",
            i,
            g,
            exp
        );
    }
}

#[test]
fn test_mse_loss_forward_2d_analytical() {
    let criterion = MSELoss::new();
    let preds = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).with_requires_grad();
    let targets = Tensor::new(vec![1.5, 2.5, 2.0, 5.0], vec![2, 2]);

    let loss = criterion.forward(&preds, &targets);
    assert_eq!(loss.shape(), vec![1]);

    // diffs: [-0.5, -0.5, 1.0, -1.0] -> sq sum = 0.25 + 0.25 + 1.0 + 1.0 = 2.5 -> mean = 0.625
    let val = loss.item();
    assert!(
        (val - 0.625).abs() < 1e-5,
        "Expected MSE 0.625, got {}",
        val
    );

    loss.backward();
    // dL/d(preds) = 2/4 * [-0.5, -0.5, 1.0, -1.0] = [-0.25, -0.25, 0.5, -0.5]
    let grads = preds.grad().expect("Gradients should be present");
    let expected_grads = vec![-0.25, -0.25, 0.5, -0.5];
    for (i, (&g, &exp)) in grads.iter().zip(expected_grads.iter()).enumerate() {
        assert!(
            (g - exp).abs() < 1e-5,
            "Gradient mismatch at {}: got {}, expected {}",
            i,
            g,
            exp
        );
    }
}

#[test]
fn test_mse_loss_forward_3d_arbitrary_shape() {
    let criterion = MSELoss::new();
    // Shape [2, 1, 3] -> numel = 6
    let preds = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 1, 3]).with_requires_grad();
    let targets = Tensor::new(vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0], vec![2, 1, 3]);

    let loss = criterion.forward(&preds, &targets);
    assert_eq!(loss.shape(), vec![1]);

    // diffs: [0.0, 1.0, 2.0, 2.0, 3.0, 4.0]
    // sq sum: 0 + 1 + 4 + 4 + 9 + 16 = 34.0 -> mean = 34 / 6 = 5.666667
    let val = loss.item();
    let expected_loss = 34.0 / 6.0;
    assert!(
        (val - expected_loss).abs() < 1e-5,
        "Expected MSE {}, got {}",
        expected_loss,
        val
    );

    loss.backward();
    // dL/d(preds) = 2/6 * [0, 1, 2, 2, 3, 4] = 1/3 * [0, 1, 2, 2, 3, 4]
    let grads = preds.grad().expect("Gradients should be present");
    let diffs = vec![0.0, 1.0, 2.0, 2.0, 3.0, 4.0];
    for (i, (&g, &d)) in grads.iter().zip(diffs.iter()).enumerate() {
        let expected_g = (2.0 / 6.0) * d;
        assert!(
            (g - expected_g).abs() < 1e-5,
            "Gradient mismatch at {}: got {}, expected {}",
            i,
            g,
            expected_g
        );
    }
}

#[test]
fn test_mse_loss_zero_when_identical() {
    let criterion = MSELoss::new();
    let preds = Tensor::new(vec![3.14, -2.71, 0.0], vec![3]).with_requires_grad();
    let targets = Tensor::new(vec![3.14, -2.71, 0.0], vec![3]);

    let loss = criterion.forward(&preds, &targets);
    assert!((loss.item() - 0.0).abs() < 1e-7);

    loss.backward();
    let grads = preds.grad().expect("Gradients should be present");
    for g in grads {
        assert!((g - 0.0).abs() < 1e-7);
    }
}

#[test]
fn test_cross_entropy_forward_1d_logits_class_index() {
    let criterion = CrossEntropyLoss::new();
    // Logits: [0.0, 1.0, 2.0], target: class 2
    let logits = Tensor::new(vec![0.0, 1.0, 2.0], vec![3]).with_requires_grad();
    let targets = Tensor::new(vec![2.0], vec![1]);

    let loss = criterion.forward(&logits, &targets);
    assert_eq!(loss.shape(), vec![1]);

    // max = 2.0
    // sum_exp = exp(-2) + exp(-1) + exp(0) = 0.13533528 + 0.36787944 + 1.0 = 1.5032147
    // log_sum_exp = 2.0 + ln(1.5032147) = 2.40760596
    // loss = log_sum_exp - logits[2] = 2.40760596 - 2.0 = 0.40760596
    let val = loss.item();
    let expected_loss = 0.40760596;
    assert!(
        (val - expected_loss).abs() < 1e-5,
        "Expected CE loss {}, got {}",
        expected_loss,
        val
    );

    loss.backward();
    // grads = P - Y = [exp(-2), exp(-1), 1.0] / 1.5032147 - [0, 0, 1]
    // = [0.09003057, 0.24472847, 0.66524096 - 1.0] = [0.09003057, 0.24472847, -0.33475904]
    let grads = logits.grad().expect("Gradients should be present");
    let expected_grads = vec![0.09003057, 0.24472847, -0.33475904];
    for (i, (&g, &exp)) in grads.iter().zip(expected_grads.iter()).enumerate() {
        assert!(
            (g - exp).abs() < 1e-5,
            "CE grad mismatch at {}: got {}, expected {}",
            i,
            g,
            exp
        );
    }
}

#[test]
fn test_cross_entropy_forward_2d_logits_class_indices() {
    let criterion = CrossEntropyLoss::new();
    // Batch size 2, 3 classes
    let logits_data = vec![
        1.0, 2.0, 3.0, // Batch 0, target class 2
        4.0, 2.0, 0.0, // Batch 1, target class 0
    ];
    let logits = Tensor::new(logits_data, vec![2, 3]).with_requires_grad();
    let targets = Tensor::new(vec![2.0, 0.0], vec![2]);

    let loss = criterion.forward(&logits, &targets);
    assert_eq!(loss.shape(), vec![1]);

    // Batch 0:
    // max = 3.0, sum_exp = exp(-2) + exp(-1) + 1 = 1.5032147
    // log_sum_exp_0 = 3.0 + ln(1.5032147) = 3.40760596
    // loss_0 = 3.40760596 - 3.0 = 0.40760596
    // P_0 = [0.09003057, 0.24472847, 0.66524096]
    //
    // Batch 1:
    // max = 4.0, sum_exp = 1 + exp(-2) + exp(-4) = 1.0 + 0.13533528 + 0.01831564 = 1.1536509
    // log_sum_exp_1 = 4.0 + ln(1.1536509) = 4.1429645
    // loss_1 = 4.1429645 - 4.0 = 0.1429645
    // P_1 = [0.8668133, 0.1173104, 0.01587623]
    //
    // Mean loss = (0.40760596 + 0.1429645) / 2 = 0.27528523
    let expected_loss = 0.27526882;
    let val = loss.item();
    assert!(
        (val - expected_loss).abs() < 1e-4,
        "Expected CE loss {}, got {}",
        expected_loss,
        val
    );

    loss.backward();
    // grads = 1/B * (P - Y)
    // Row 0: 0.5 * [0.09003057, 0.24472847, -0.33475904] = [0.04501528, 0.12236424, -0.16737952]
    // Row 1: 0.5 * [-0.1331867, 0.1173104, 0.01587623] = [-0.06659335, 0.0586552, 0.00793812]
    let grads = logits.grad().expect("Gradients should be present");
    let expected_grads = vec![
        0.04501528, 0.12236424, -0.16737952,
        -0.06659335, 0.0586552, 0.00793812,
    ];
    for (i, (&g, &exp)) in grads.iter().zip(expected_grads.iter()).enumerate() {
        assert!(
            (g - exp).abs() < 1e-4,
            "CE batch grad mismatch at {}: got {}, expected {}",
            i,
            g,
            exp
        );
    }
}

#[test]
fn test_cross_entropy_forward_2d_one_hot_targets() {
    let criterion = CrossEntropyLoss::new();
    let logits_data = vec![
        1.0, 2.0, 3.0,
        4.0, 2.0, 0.0,
    ];
    let logits = Tensor::new(logits_data, vec![2, 3]).with_requires_grad();
    // One-hot representation of targets [2, 0]
    let targets = Tensor::new(
        vec![
            0.0, 0.0, 1.0,
            1.0, 0.0, 0.0,
        ],
        vec![2, 3],
    );

    let loss = criterion.forward(&logits, &targets);
    let expected_loss = 0.27526882;
    assert!(
        (loss.item() - expected_loss).abs() < 1e-4,
        "Expected CE one-hot loss {}, got {}",
        expected_loss,
        loss.item()
    );

    loss.backward();
    let grads = logits.grad().expect("Gradients should be present");
    let expected_grads = vec![
        0.04501528, 0.12236424, -0.16737952,
        -0.06659335, 0.0586552, 0.00793812,
    ];
    for (i, (&g, &exp)) in grads.iter().zip(expected_grads.iter()).enumerate() {
        assert!(
            (g - exp).abs() < 1e-4,
            "CE one-hot grad mismatch at {}: got {}, expected {}",
            i,
            g,
            exp
        );
    }
}

#[test]
fn test_cross_entropy_forward_2d_soft_distribution_targets() {
    let criterion = CrossEntropyLoss::new();
    let logits_data = vec![
        1.0, 2.0, 3.0,
        4.0, 2.0, 0.0,
    ];
    let logits = Tensor::new(logits_data, vec![2, 3]).with_requires_grad();
    // Soft target distributions: sum to 1.0 per row
    let targets = Tensor::new(
        vec![
            0.2, 0.3, 0.5,
            0.8, 0.1, 0.1,
        ],
        vec![2, 3],
    );

    let loss = criterion.forward(&logits, &targets);

    // Row 0:
    // log_sum_exp_0 = 3.40760596
    // sum(Y * z) = 0.2*1.0 + 0.3*2.0 + 0.5*3.0 = 0.2 + 0.6 + 1.5 = 2.3
    // loss_0 = 3.40760596 - 2.3 = 1.10760596
    //
    // Row 1:
    // log_sum_exp_1 = 4.1429645
    // sum(Y * z) = 0.8*4.0 + 0.1*2.0 + 0.1*0.0 = 3.2 + 0.2 = 3.4
    // loss_1 = 4.1429645 - 3.4 = 0.7429645
    //
    // Mean loss = (1.10760596 + 0.7429645) / 2 = 0.92528523
    let expected_loss = 0.92526877;
    assert!(
        (loss.item() - expected_loss).abs() < 1e-4,
        "Expected CE soft targets loss {}, got {}",
        expected_loss,
        loss.item()
    );

    loss.backward();
    // grads = 1/2 * (P - Y)
    // Row 0: 0.5 * ([0.09003057, 0.24472847, 0.66524096] - [0.2, 0.3, 0.5])
    //      = 0.5 * [-0.10996943, -0.05527153, 0.16524096]
    //      = [-0.05498471, -0.02763577, 0.08262048]
    // Row 1: 0.5 * ([0.8668133, 0.1173104, 0.01587623] - [0.8, 0.1, 0.1])
    //      = 0.5 * [0.0668133, 0.0173104, -0.08412377]
    //      = [0.03340665, 0.0086552, -0.04206188]
    let grads = logits.grad().expect("Gradients should be present");
    let expected_grads = vec![
        -0.05498471, -0.02763577, 0.08262048,
        0.03340665, 0.0086552, -0.04206188,
    ];
    for (i, (&g, &exp)) in grads.iter().zip(expected_grads.iter()).enumerate() {
        assert!(
            (g - exp).abs() < 1e-4,
            "CE soft distribution grad mismatch at {}: got {}, expected {}",
            i,
            g,
            exp
        );
    }
}

#[test]
fn test_cross_entropy_numerical_stability_large_logits() {
    let criterion = CrossEntropyLoss::new();
    // Large logits would cause exp(z) to overflow to Inf without Log-Sum-Exp trick
    let logits = Tensor::new(vec![1000.0, 1002.0, 999.0], vec![1, 3]).with_requires_grad();
    let targets = Tensor::new(vec![1.0], vec![1]);

    let loss = criterion.forward(&logits, &targets);
    let val = loss.item();
    assert!(!val.is_nan() && !val.is_infinite(), "Loss must be finite: got {}", val);

    // max = 1002.0
    // sum_exp = exp(-2) + exp(0) + exp(-3) = 0.13533528 + 1.0 + 0.04978707 = 1.1851224
    // log_sum_exp = 1002.0 + ln(1.1851224) = 1002.1698457
    // loss = 1002.1698457 - 1002.0 = 0.1698457
    let expected_loss = 0.1698457;
    assert!(
        (val - expected_loss).abs() < 1e-4,
        "Expected stable CE loss {}, got {}",
        expected_loss,
        val
    );

    loss.backward();
    let grads = logits.grad().expect("Gradients should be computed");
    for &g in &grads {
        assert!(!g.is_nan() && !g.is_infinite(), "Grad must be finite: got {}", g);
    }
}

#[test]
fn test_nll_loss_forward_class_indices() {
    let criterion = NLLLoss::new();
    // Probabilities summing to 1 per row
    let probs_data = vec![
        0.1, 0.7, 0.2, // Row 0, target class 1
        0.3, 0.3, 0.4, // Row 1, target class 2
    ];
    let probs = Tensor::new(probs_data, vec![2, 3]).with_requires_grad();
    let targets = Tensor::new(vec![1.0, 2.0], vec![2]);

    let loss = criterion.forward(&probs, &targets);
    assert_eq!(loss.shape(), vec![1]);

    // Row 0 loss: -ln(0.7) = 0.35667494
    // Row 1 loss: -ln(0.4) = 0.91629073
    // Mean loss: (0.35667494 + 0.91629073) / 2 = 0.63648284
    let expected_loss = 0.63648284;
    let val = loss.item();
    assert!(
        (val - expected_loss).abs() < 1e-5,
        "Expected NLL loss {}, got {}",
        expected_loss,
        val
    );

    loss.backward();
    // dL/d(P_{b,c}) = -1/B * Y_{b,c} / P_{b,c}
    // Row 0 (target 1): -1/2 * [0, 1/0.7, 0] = [0.0, -0.7142857, 0.0]
    // Row 1 (target 2): -1/2 * [0, 0, 1/0.4] = [0.0, 0.0, -1.25]
    let grads = probs.grad().expect("Gradients should be computed");
    let expected_grads = vec![
        0.0, -0.7142857, 0.0,
        0.0, 0.0, -1.25,
    ];
    for (i, (&g, &exp)) in grads.iter().zip(expected_grads.iter()).enumerate() {
        assert!(
            (g - exp).abs() < 1e-4,
            "NLL grad mismatch at {}: got {}, expected {}",
            i,
            g,
            exp
        );
    }
}

#[test]
fn test_nll_loss_forward_one_hot_targets() {
    let criterion = NLLLoss::new();
    let probs_data = vec![
        0.1, 0.7, 0.2,
        0.3, 0.3, 0.4,
    ];
    let probs = Tensor::new(probs_data, vec![2, 3]).with_requires_grad();
    let targets = Tensor::new(
        vec![
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ],
        vec![2, 3],
    );

    let loss = criterion.forward(&probs, &targets);
    let expected_loss = 0.63648284;
    assert!(
        (loss.item() - expected_loss).abs() < 1e-5,
        "Expected NLL one-hot loss {}, got {}",
        expected_loss,
        loss.item()
    );

    loss.backward();
    let grads = probs.grad().expect("Gradients should be computed");
    let expected_grads = vec![
        0.0, -0.7142857, 0.0,
        0.0, 0.0, -1.25,
    ];
    for (i, (&g, &exp)) in grads.iter().zip(expected_grads.iter()).enumerate() {
        assert!(
            (g - exp).abs() < 1e-4,
            "NLL one-hot grad mismatch at {}: got {}, expected {}",
            i,
            g,
            exp
        );
    }
}

#[test]
fn test_nll_loss_forward_soft_distribution_targets() {
    let criterion = NLLLoss::new();
    let probs_data = vec![
        0.2, 0.5, 0.3,
        0.7, 0.1, 0.2,
    ];
    let probs = Tensor::new(probs_data, vec![2, 3]).with_requires_grad();
    let targets = Tensor::new(
        vec![
            0.1, 0.6, 0.3,
            0.5, 0.3, 0.2,
        ],
        vec![2, 3],
    );

    let loss = criterion.forward(&probs, &targets);

    // Row 0 loss: -(0.1*ln(0.2) + 0.6*ln(0.5) + 0.3*ln(0.3))
    // ln(0.2) = -1.60943791 -> 0.1 * (-1.60943791) = -0.16094379
    // ln(0.5) = -0.69314718 -> 0.6 * (-0.69314718) = -0.41588831
    // ln(0.3) = -1.20397280 -> 0.3 * (-1.20397280) = -0.36119184
    // sum = -0.93802394 -> loss_0 = 0.93802394
    //
    // Row 1 loss: -(0.5*ln(0.7) + 0.3*ln(0.1) + 0.2*ln(0.2))
    // ln(0.7) = -0.35667494 -> 0.5 * (-0.35667494) = -0.17833747
    // ln(0.1) = -2.30258509 -> 0.3 * (-2.30258509) = -0.69077553
    // ln(0.2) = -1.60943791 -> 0.2 * (-1.60943791) = -0.32188758
    // sum = -1.19100058 -> loss_1 = 1.19100058
    //
    // Mean loss: (0.93802394 + 1.19100058) / 2 = 1.06451226
    let expected_loss = 1.06451226;
    assert!(
        (loss.item() - expected_loss).abs() < 1e-4,
        "Expected NLL soft loss {}, got {}",
        expected_loss,
        loss.item()
    );

    loss.backward();
    // dL/d(P_{b,c}) = -1/B * Y_{b,c} / P_{b,c}
    // Row 0: -0.5 * [0.1/0.2, 0.6/0.5, 0.3/0.3] = -0.5 * [0.5, 1.2, 1.0] = [-0.25, -0.6, -0.5]
    // Row 1: -0.5 * [0.5/0.7, 0.3/0.1, 0.2/0.2] = -0.5 * [0.7142857, 3.0, 1.0] = [-0.35714286, -1.5, -0.5]
    let grads = probs.grad().expect("Gradients should be computed");
    let expected_grads = vec![
        -0.25, -0.6, -0.5,
        -0.35714286, -1.5, -0.5,
    ];
    for (i, (&g, &exp)) in grads.iter().zip(expected_grads.iter()).enumerate() {
        assert!(
            (g - exp).abs() < 1e-4,
            "NLL soft grad mismatch at {}: got {}, expected {}",
            i,
            g,
            exp
        );
    }
}

#[test]
fn test_nll_loss_with_custom_eps() {
    let criterion = NLLLoss::with_eps(1e-4);
    // Even if prob is 0.0, eps should prevent -Inf
    let probs = Tensor::new(vec![0.0, 1.0], vec![1, 2]);
    let targets = Tensor::new(vec![0.0], vec![1]);

    let loss = criterion.forward(&probs, &targets);
    let val = loss.item();
    assert!(!val.is_infinite() && !val.is_nan());
    // -ln(1e-4) = ln(10000) ~= 9.21034
    assert!((val - 9.21034).abs() < 1e-3);
}

// =========================================================================
// 2. Finite Differences Numerical Gradient Verification
// =========================================================================

fn check_numerical_gradient<F>(
    f: F,
    initial_data: &[f32],
    shape: &[usize],
    analytical_grads: &[f32],
    eps: f32,
    tol: f32,
) where
    F: Fn(&Tensor) -> Tensor,
{
    for i in 0..initial_data.len() {
        let mut data_plus = initial_data.to_vec();
        data_plus[i] += eps;
        let loss_plus = f(&Tensor::new(data_plus, shape.to_vec())).item();

        let mut data_minus = initial_data.to_vec();
        data_minus[i] -= eps;
        let loss_minus = f(&Tensor::new(data_minus, shape.to_vec())).item();

        let num_grad = (loss_plus - loss_minus) / (2.0 * eps);
        let diff = (analytical_grads[i] - num_grad).abs();
        assert!(
            diff < tol,
            "Gradient mismatch at index {}: analytical={}, numerical={}, diff={} (tol={})",
            i,
            analytical_grads[i],
            num_grad,
            diff,
            tol
        );
    }
}

#[test]
fn test_mse_loss_numerical_gradient_1d() {
    let criterion = MSELoss::new();
    let initial_data = vec![1.2, -0.5, 2.8, 0.3];
    let targets = Tensor::new(vec![1.0, 0.0, 3.0, 0.0], vec![4]);

    let preds = Tensor::new(initial_data.clone(), vec![4]).with_requires_grad();
    let loss = criterion.forward(&preds, &targets);
    loss.backward();

    let analytical_grads = preds.grad().expect("Gradients should be computed");
    check_numerical_gradient(
        |x| criterion.forward(x, &targets),
        &initial_data,
        &[4],
        &analytical_grads,
        EPSILON,
        TOLERANCE,
    );
}

#[test]
fn test_mse_loss_numerical_gradient_2d() {
    let criterion = MSELoss::new();
    let initial_data = vec![0.5, -1.2, 3.4, 2.1, -0.8, 1.0];
    let shape = vec![2, 3];
    let targets = Tensor::new(vec![0.0, -1.0, 3.0, 2.0, -1.0, 1.5], shape.clone());

    let preds = Tensor::new(initial_data.clone(), shape.clone()).with_requires_grad();
    let loss = criterion.forward(&preds, &targets);
    loss.backward();

    let analytical_grads = preds.grad().expect("Gradients should be computed");
    check_numerical_gradient(
        |x| criterion.forward(x, &targets),
        &initial_data,
        &shape,
        &analytical_grads,
        EPSILON,
        TOLERANCE,
    );
}

#[test]
fn test_cross_entropy_numerical_gradient_1d_logits() {
    let criterion = CrossEntropyLoss::new();
    let initial_logits = vec![0.8, -1.4, 2.3, 0.1];
    let shape = vec![4];
    let targets = Tensor::new(vec![2.0], vec![1]);

    let logits = Tensor::new(initial_logits.clone(), shape.clone()).with_requires_grad();
    let loss = criterion.forward(&logits, &targets);
    loss.backward();

    let analytical_grads = logits.grad().expect("Gradients should be computed");
    check_numerical_gradient(
        |x| criterion.forward(x, &targets),
        &initial_logits,
        &shape,
        &analytical_grads,
        EPSILON,
        TOLERANCE,
    );
}

#[test]
fn test_cross_entropy_numerical_gradient_batch_class_indices() {
    let criterion = CrossEntropyLoss::new();
    let initial_logits = vec![1.5, -0.8, 2.1, 0.4, 0.1, 1.9, -0.3, 0.7];
    let shape = vec![2, 4];
    let targets = Tensor::new(vec![2.0, 1.0], vec![2]);

    let logits = Tensor::new(initial_logits.clone(), shape.clone()).with_requires_grad();
    let loss = criterion.forward(&logits, &targets);
    loss.backward();

    let analytical_grads = logits.grad().expect("Gradients should be computed");
    check_numerical_gradient(
        |x| criterion.forward(x, &targets),
        &initial_logits,
        &shape,
        &analytical_grads,
        EPSILON,
        TOLERANCE,
    );
}

#[test]
fn test_cross_entropy_numerical_gradient_batch_one_hot() {
    let criterion = CrossEntropyLoss::new();
    let initial_logits = vec![2.0, -1.0, 0.5, 1.2, 0.0, 3.1];
    let shape = vec![2, 3];
    let targets = Tensor::new(
        vec![
            0.0, 0.0, 1.0,
            1.0, 0.0, 0.0,
        ],
        shape.clone(),
    );

    let logits = Tensor::new(initial_logits.clone(), shape.clone()).with_requires_grad();
    let loss = criterion.forward(&logits, &targets);
    loss.backward();

    let analytical_grads = logits.grad().expect("Gradients should be computed");
    check_numerical_gradient(
        |x| criterion.forward(x, &targets),
        &initial_logits,
        &shape,
        &analytical_grads,
        EPSILON,
        TOLERANCE,
    );
}

#[test]
fn test_cross_entropy_numerical_gradient_batch_soft_targets() {
    let criterion = CrossEntropyLoss::new();
    let initial_logits = vec![1.1, 0.9, -0.5, 0.2, 2.5, -1.0];
    let shape = vec![2, 3];
    let targets = Tensor::new(
        vec![
            0.3, 0.5, 0.2,
            0.1, 0.7, 0.2,
        ],
        shape.clone(),
    );

    let logits = Tensor::new(initial_logits.clone(), shape.clone()).with_requires_grad();
    let loss = criterion.forward(&logits, &targets);
    loss.backward();

    let analytical_grads = logits.grad().expect("Gradients should be computed");
    check_numerical_gradient(
        |x| criterion.forward(x, &targets),
        &initial_logits,
        &shape,
        &analytical_grads,
        EPSILON,
        TOLERANCE,
    );
}

#[test]
fn test_nll_loss_numerical_gradient_class_indices() {
    let criterion = NLLLoss::new();
    // Probabilities kept away from 0 and 1 so perturbations stay well within bounds
    let initial_probs = vec![0.2, 0.5, 0.3, 0.4, 0.2, 0.4];
    let shape = vec![2, 3];
    let targets = Tensor::new(vec![1.0, 0.0], vec![2]);

    let probs = Tensor::new(initial_probs.clone(), shape.clone()).with_requires_grad();
    let loss = criterion.forward(&probs, &targets);
    loss.backward();

    let analytical_grads = probs.grad().expect("Gradients should be computed");
    check_numerical_gradient(
        |x| criterion.forward(x, &targets),
        &initial_probs,
        &shape,
        &analytical_grads,
        EPSILON,
        TOLERANCE,
    );
}

#[test]
fn test_nll_loss_numerical_gradient_one_hot() {
    let criterion = NLLLoss::new();
    let initial_probs = vec![0.15, 0.65, 0.20, 0.50, 0.25, 0.25];
    let shape = vec![2, 3];
    let targets = Tensor::new(
        vec![
            0.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
        ],
        shape.clone(),
    );

    let probs = Tensor::new(initial_probs.clone(), shape.clone()).with_requires_grad();
    let loss = criterion.forward(&probs, &targets);
    loss.backward();

    let analytical_grads = probs.grad().expect("Gradients should be computed");
    check_numerical_gradient(
        |x| criterion.forward(x, &targets),
        &initial_probs,
        &shape,
        &analytical_grads,
        EPSILON,
        TOLERANCE,
    );
}

#[test]
fn test_nll_loss_numerical_gradient_soft_targets() {
    let criterion = NLLLoss::new();
    let initial_probs = vec![0.3, 0.4, 0.3, 0.2, 0.5, 0.3];
    let shape = vec![2, 3];
    let targets = Tensor::new(
        vec![
            0.2, 0.6, 0.2,
            0.4, 0.4, 0.2,
        ],
        shape.clone(),
    );

    let probs = Tensor::new(initial_probs.clone(), shape.clone()).with_requires_grad();
    let loss = criterion.forward(&probs, &targets);
    loss.backward();

    let analytical_grads = probs.grad().expect("Gradients should be computed");
    check_numerical_gradient(
        |x| criterion.forward(x, &targets),
        &initial_probs,
        &shape,
        &analytical_grads,
        EPSILON,
        TOLERANCE,
    );
}

// =========================================================================
// 3. Architectural Equivalence Invariance
// =========================================================================

#[test]
fn test_invariance_single_sample_class_index() {
    let logits_data = vec![2.5, -0.5, 1.0, 0.2];
    let shape = vec![1, 4];
    let targets_data = vec![2.0];

    // Pipeline A: CrossEntropyLoss directly on logits
    let z_ce = Tensor::new(logits_data.clone(), shape.clone()).with_requires_grad();
    let y_ce = Tensor::new(targets_data.clone(), vec![1]);
    let loss_ce = CrossEntropyLoss::new().forward(&z_ce, &y_ce);
    loss_ce.backward();

    // Pipeline B: Softmax followed by NLLLoss
    let z_nll = Tensor::new(logits_data.clone(), shape.clone()).with_requires_grad();
    let y_nll = Tensor::new(targets_data.clone(), vec![1]);
    let probs = Softmax::new().forward(&z_nll);
    let loss_nll = NLLLoss::new().forward(&probs, &y_nll);
    loss_nll.backward();

    // Forward scalar invariant
    let ce_val = loss_ce.item();
    let nll_val = loss_nll.item();
    assert!(
        (ce_val - nll_val).abs() < 1e-4,
        "Forward loss mismatch (B=1): CE={}, NLL={}",
        ce_val,
        nll_val
    );

    // Backward gradient invariant on logits z
    let grad_ce = z_ce.grad().expect("CE gradients");
    let grad_nll = z_nll.grad().expect("NLL gradients");
    assert_eq!(grad_ce.len(), grad_nll.len());
    for (i, (&g1, &g2)) in grad_ce.iter().zip(grad_nll.iter()).enumerate() {
        assert!(
            (g1 - g2).abs() < 1e-4,
            "Gradient mismatch at index {}: CE={}, NLL={}",
            i,
            g1,
            g2
        );
    }
}

#[test]
fn test_invariance_multi_sample_class_index() {
    let logits_data = vec![
        2.0, 1.0, 0.1, -1.0,
        0.5, 3.0, 1.2, 0.0,
        -0.5, 0.0, 2.2, 1.1,
    ];
    let shape = vec![3, 4];
    let targets_data = vec![1.0, 2.0, 3.0];

    let z_ce = Tensor::new(logits_data.clone(), shape.clone()).with_requires_grad();
    let y_ce = Tensor::new(targets_data.clone(), vec![3]);
    let loss_ce = CrossEntropyLoss::new().forward(&z_ce, &y_ce);
    loss_ce.backward();

    let z_nll = Tensor::new(logits_data.clone(), shape.clone()).with_requires_grad();
    let y_nll = Tensor::new(targets_data.clone(), vec![3]);
    let probs = Softmax::new().forward(&z_nll);
    let loss_nll = NLLLoss::new().forward(&probs, &y_nll);
    loss_nll.backward();

    let ce_val = loss_ce.item();
    let nll_val = loss_nll.item();
    assert!(
        (ce_val - nll_val).abs() < 1e-4,
        "Forward loss mismatch: CE={}, NLL={}",
        ce_val,
        nll_val
    );

    let grad_ce = z_ce.grad().expect("CE gradients");
    let grad_nll = z_nll.grad().expect("NLL gradients");
    for (i, (&g1, &g2)) in grad_ce.iter().zip(grad_nll.iter()).enumerate() {
        assert!(
            (g1 - g2).abs() < 1e-4,
            "Gradient mismatch at index {}: CE={}, NLL={}",
            i,
            g1,
            g2
        );
    }
}

#[test]
fn test_invariance_multi_sample_one_hot() {
    let logits_data = vec![
        1.5, -0.5, 0.8,
        0.2, 2.1, -1.0,
        -0.2, 0.4, 1.8,
        0.0, 1.2, 0.5,
    ];
    let shape = vec![4, 3];
    let targets_data = vec![
        0.0, 0.0, 1.0,
        0.0, 1.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 0.0, 1.0,
    ];

    let z_ce = Tensor::new(logits_data.clone(), shape.clone()).with_requires_grad();
    let y_ce = Tensor::new(targets_data.clone(), shape.clone());
    let loss_ce = CrossEntropyLoss::new().forward(&z_ce, &y_ce);
    loss_ce.backward();

    let z_nll = Tensor::new(logits_data.clone(), shape.clone()).with_requires_grad();
    let y_nll = Tensor::new(targets_data.clone(), shape.clone());
    let probs = Softmax::new().forward(&z_nll);
    let loss_nll = NLLLoss::new().forward(&probs, &y_nll);
    loss_nll.backward();

    let ce_val = loss_ce.item();
    let nll_val = loss_nll.item();
    assert!(
        (ce_val - nll_val).abs() < 1e-4,
        "Forward loss mismatch (one-hot): CE={}, NLL={}",
        ce_val,
        nll_val
    );

    let grad_ce = z_ce.grad().expect("CE gradients");
    let grad_nll = z_nll.grad().expect("NLL gradients");
    for (i, (&g1, &g2)) in grad_ce.iter().zip(grad_nll.iter()).enumerate() {
        assert!(
            (g1 - g2).abs() < 1e-4,
            "One-hot gradient mismatch at index {}: CE={}, NLL={}",
            i,
            g1,
            g2
        );
    }
}

// =========================================================================
// 4. Edge Cases & #[should_panic] Tests
// =========================================================================

#[test]
#[should_panic]
fn test_mse_loss_shape_mismatch_1d_should_panic() {
    let criterion = MSELoss::new();
    let preds = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let targets = Tensor::new(vec![1.0, 2.0], vec![2]);
    let _ = criterion.forward(&preds, &targets);
}

#[test]
#[should_panic]
fn test_mse_loss_shape_mismatch_2d_should_panic() {
    let criterion = MSELoss::new();
    let preds = Tensor::new(vec![1.0; 6], vec![2, 3]);
    let targets = Tensor::new(vec![1.0; 6], vec![3, 2]);
    let _ = criterion.forward(&preds, &targets);
}

#[test]
#[should_panic]
fn test_cross_entropy_target_batch_mismatch_should_panic() {
    let criterion = CrossEntropyLoss::new();
    // Batch size 3, but target length 2
    let logits = Tensor::new(vec![1.0; 9], vec![3, 3]);
    let targets = Tensor::new(vec![0.0, 1.0], vec![2]);
    let _ = criterion.forward(&logits, &targets);
}

#[test]
#[should_panic]
fn test_cross_entropy_one_hot_shape_mismatch_should_panic() {
    let criterion = CrossEntropyLoss::new();
    // Logits [2, 3], but one-hot targets [2, 4]
    let logits = Tensor::new(vec![1.0; 6], vec![2, 3]);
    let targets = Tensor::new(vec![0.0; 8], vec![2, 4]);
    let _ = criterion.forward(&logits, &targets);
}

#[test]
#[should_panic]
fn test_cross_entropy_class_index_out_of_bounds_should_panic() {
    let criterion = CrossEntropyLoss::new();
    // Num classes is 3 (indices 0, 1, 2), but target class is 3.0
    let logits = Tensor::new(vec![1.0; 6], vec![2, 3]);
    let targets = Tensor::new(vec![0.0, 3.0], vec![2]);
    let _ = criterion.forward(&logits, &targets);
}

#[test]
#[should_panic]
fn test_cross_entropy_class_index_far_out_of_bounds_should_panic() {
    let criterion = CrossEntropyLoss::new();
    let logits = Tensor::new(vec![0.5; 4], vec![1, 4]);
    let targets = Tensor::new(vec![99.0], vec![1]);
    let _ = criterion.forward(&logits, &targets);
}

#[test]
#[should_panic]
fn test_nll_loss_target_batch_mismatch_should_panic() {
    let criterion = NLLLoss::new();
    let probs = Tensor::new(vec![0.5; 6], vec![2, 3]);
    let targets = Tensor::new(vec![0.0], vec![1]);
    let _ = criterion.forward(&probs, &targets);
}

#[test]
#[should_panic]
fn test_nll_loss_one_hot_shape_mismatch_should_panic() {
    let criterion = NLLLoss::new();
    let probs = Tensor::new(vec![0.5; 6], vec![2, 3]);
    let targets = Tensor::new(vec![0.5; 4], vec![2, 2]);
    let _ = criterion.forward(&probs, &targets);
}

#[test]
#[should_panic]
fn test_nll_loss_class_index_out_of_bounds_should_panic() {
    let criterion = NLLLoss::new();
    let probs = Tensor::new(vec![0.33; 6], vec![2, 3]);
    let targets = Tensor::new(vec![0.0, 4.0], vec![2]);
    let _ = criterion.forward(&probs, &targets);
}
