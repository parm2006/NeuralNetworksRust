use crate::tensor::Tensor;
use super::optimizer::Optimizer;

/// Stochastic Gradient Descent (SGD) optimizer.
///
/// Updates model parameters in-place using their accumulated gradients:
/// `weight = weight - lr * grad`
pub struct SGD {
    pub params: Vec<Tensor>,
    pub lr: f32,
}

impl SGD {
    /// Creates a new SGD optimizer from model parameters.
    pub fn new(parameters: Vec<(String, Tensor)>, lr: f32) -> Self {
        let params = parameters.into_iter().map(|(_, p)| p).collect();
        Self { params, lr }
    }
}

impl Optimizer for SGD {
    /// Performs a single optimization step, updating all parameters with gradients.
    fn step(&mut self) {
        for p in &self.params {
            if let Some(grad) = p.grad() {
                let mut data = p.data();
                for (w, &g) in data.iter_mut().zip(grad.iter()) {
                    *w -= self.lr * g;
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
