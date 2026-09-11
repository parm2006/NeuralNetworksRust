use crate::tensor::Tensor;
use super::optimizer::Optimizer;

/// AdamW optimizer with decoupled weight decay (Loshchilov & Hutter, 2017).
///
/// Updates model parameters using bias-corrected 1st and 2nd moment estimates,
/// with weight decay applied directly to the weights rather than gradients:
/// ```text
/// m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
/// v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
/// m_hat = m_t / (1 - beta1^t)
/// v_hat = v_t / (1 - beta2^t)
/// w_t = w_{t-1} - lr * weight_decay * w_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
/// ```
pub struct AdamW {
    pub params: Vec<Tensor>,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step_count: usize,
    pub m: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

impl AdamW {
    /// Creates a new AdamW optimizer with standard PyTorch defaults:
    /// `beta1 = 0.9`, `beta2 = 0.999`, `eps = 1e-8`, `weight_decay = 0.01`.
    pub fn new(parameters: Vec<(String, Tensor)>, lr: f32) -> Self {
        Self::with_hyperparams(parameters, lr, 0.9, 0.999, 1e-8, 0.01)
    }

    /// Creates a new AdamW optimizer with custom hyperparameters.
    pub fn with_hyperparams(
        parameters: Vec<(String, Tensor)>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) -> Self {
        let params: Vec<Tensor> = parameters.into_iter().map(|(_, p)| p).collect();
        let m = params.iter().map(|p| vec![0.0; p.numel()]).collect();
        let v = params.iter().map(|p| vec![0.0; p.numel()]).collect();
        Self {
            params,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step_count: 0,
            m,
            v,
        }
    }
}

impl Optimizer for AdamW {
    /// Performs a single optimization step, updating all tracked parameters.
    fn step(&mut self) {
        self.step_count += 1;
        let t = self.step_count as i32;
        let bias_correction1 = 1.0 - self.beta1.powi(t);
        let bias_correction2 = 1.0 - self.beta2.powi(t);

        for (i, p) in self.params.iter().enumerate() {
            if let Some(grad) = p.grad() {
                let mut data = p.data();
                let m = &mut self.m[i];
                let v = &mut self.v[i];

                for j in 0..data.len() {
                    let g = grad[j];

                    // Update biased 1st and 2nd moment estimates
                    m[j] = self.beta1 * m[j] + (1.0 - self.beta1) * g;
                    v[j] = self.beta2 * v[j] + (1.0 - self.beta2) * g * g;

                    // Bias-corrected moment estimates
                    let m_hat = m[j] / bias_correction1;
                    let v_hat = v[j] / bias_correction2;

                    // Decoupled weight decay
                    data[j] -= self.lr * self.weight_decay * data[j];

                    // Gradient step
                    data[j] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
                }
                p.set_data(data);
            }
        }
    }

    /// Resets all parameter gradient buffers to zero.
    fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
}
