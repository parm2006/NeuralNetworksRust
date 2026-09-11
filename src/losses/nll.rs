use super::Loss;
use crate::tensor::Tensor;

/// Negative Log-Likelihood (NLL) Loss on predicted probabilities.
///
/// Expects inputs to be probabilities in (0, 1) (e.g. the output of a `Softmax` layer).
/// For each sample, computes: -ln(p_target).
///
/// If your network does NOT have a Softmax layer and outputs raw logits,
/// use `CrossEntropyLoss` instead.
pub struct NLLLoss {
    pub eps: f32,
}

impl NLLLoss {
    pub fn new() -> Self {
        Self { eps: 1e-7 }
    }

    pub fn with_eps(eps: f32) -> Self {
        Self { eps }
    }
}

impl Loss for NLLLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let p_shape = predictions.shape();
        let p_data = predictions.data();

        let (batch_size, num_classes) = if p_shape.len() <= 1 {
            (1, p_data.len())
        } else {
            let c = *p_shape.last().unwrap_or(&1);
            (predictions.numel() / c, c)
        };

        let t_data = targets.data();
        let is_class_index = targets.numel() == batch_size;

        if !is_class_index {
            assert_eq!(
                predictions.shape(),
                targets.shape(),
                "Shape mismatch: targets must either be 1D class indices of length {} or match predictions shape {:?}, but got {:?}",
                batch_size,
                predictions.shape(),
                targets.shape()
            );
        }

        let eps = self.eps;
        let mut total_loss = 0.0;

        for b in 0..batch_size {
            let row_p = &p_data[b * num_classes..(b + 1) * num_classes];

            if is_class_index {
                let target_idx = t_data[b].round() as usize;
                assert!(
                    target_idx < num_classes,
                    "Target class index {} out of bounds for num_classes {}",
                    target_idx,
                    num_classes
                );
                let p = row_p[target_idx].max(eps);
                total_loss += -p.ln();
            } else {
                let row_t = &t_data[b * num_classes..(b + 1) * num_classes];
                for (&p, &t) in row_p.iter().zip(row_t.iter()) {
                    if t > 0.0 {
                        total_loss += -t * p.max(eps).ln();
                    }
                }
            }
        }

        let mean_loss = total_loss / (batch_size as f32);
        let mut output = Tensor::new(vec![mean_loss], vec![1]);

        let pred_clone = predictions.clone();
        let p_saved = p_data;
        let t_saved = t_data;

        output.set_autograd(
            vec![predictions.clone()],
            Box::new(move |grad| {
                let g = grad[0];
                let factor = g / (batch_size as f32);
                let mut in_grad = vec![0.0; p_saved.len()];

                for b in 0..batch_size {
                    let offset = b * num_classes;
                    if is_class_index {
                        let target_idx = t_saved[b].round() as usize;
                        let p = p_saved[offset + target_idx].max(eps);
                        // d/dp [-ln(p)] = -1/p
                        in_grad[offset + target_idx] = -factor / p;
                    } else {
                        for c in 0..num_classes {
                            let t = t_saved[offset + c];
                            if t > 0.0 {
                                let p = p_saved[offset + c].max(eps);
                                in_grad[offset + c] = -factor * t / p;
                            }
                        }
                    }
                }

                pred_clone.add_grad(&in_grad);
            }),
        );

        output
    }
}
