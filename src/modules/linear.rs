use super::Module;
use crate::tensor::Tensor;

pub struct Linear {
    pub weights: Tensor,
    pub bias: Tensor,
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let weights = Tensor::kaiming(vec![in_dim, out_dim]).with_requires_grad();
        let bias = Tensor::zeros(vec![out_dim]).with_requires_grad();

        Self { weights, bias }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Tensor {
        x * &self.weights + &self.bias
    }

    fn direct_parameters(&self) -> Vec<(String, Tensor)> {
        vec![
            ("weights".to_string(), self.weights.clone()),
            ("bias".to_string(), self.bias.clone()),
        ]
    }
}
