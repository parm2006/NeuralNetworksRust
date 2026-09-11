use super::Loss;
use crate::tensor::Tensor;

pub struct MSELoss {}

impl MSELoss {
    pub fn new() -> Self {
        Self {}
    }
}

impl Loss for MSELoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        assert_eq!(
            predictions.shape(),
            targets.shape(),
            "Shape mismatch: predictions shape {:?} does not match targets shape {:?}",
            predictions.shape(),
            targets.shape()
        );

        let p_data = predictions.data();
        let t_data = targets.data();
        let n = p_data.len() as f32;

        let mut loss_sum = 0.0;
        for (&p, &t) in p_data.iter().zip(t_data.iter()) {
            let diff = p - t;
            loss_sum += diff * diff;
        }
        let mean_loss = loss_sum / n;

        let mut output = Tensor::new(vec![mean_loss], vec![1]);
        let pred_clone = predictions.clone();
        let p_saved = p_data;
        let t_saved = t_data;

        output.set_autograd(
            vec![predictions.clone()],
            Box::new(move |grad| {
                let g = grad[0];
                let factor = 2.0 * g / n;
                let mut in_grad = Vec::with_capacity(p_saved.len());
                for (&p, &t) in p_saved.iter().zip(t_saved.iter()) {
                    in_grad.push(factor * (p - t));
                }
                pred_clone.add_grad(&in_grad);
            }),
        );

        output
    }
}
