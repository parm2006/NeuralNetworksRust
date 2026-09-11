use super::{Loss, NLLLoss};
use crate::modules::{Module, Softmax};
use crate::tensor::Tensor;

pub struct CrossEntropyLoss {
    softmax: Softmax,
    nll_loss: NLLLoss,
}

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self {
            softmax: Softmax::new(),
            nll_loss: NLLLoss::new(),
        }
    }
}

impl Loss for CrossEntropyLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let softmax_output = self.softmax.forward(predictions);
        self.nll_loss.forward(&softmax_output, targets)
    }
}
