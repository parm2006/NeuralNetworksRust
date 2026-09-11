use crate::tensor::Tensor;

pub trait Loss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor;
}
