use super::Module;
use crate::tensor::Tensor;

pub struct Sigmoid {}

impl Sigmoid {
    pub fn new() -> Self {
        Self {}
    }
}

impl Module for Sigmoid {
    fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        let data = input.data();

        // Numerically stable sigmoid: 1 / (1 + exp(-x))
        let sigmoided: Vec<f32> = data
            .iter()
            .map(|&x| {
                if x >= 0.0 {
                    1.0 / (1.0 + (-x).exp())
                } else {
                    let ez = x.exp();
                    ez / (1.0 + ez)
                }
            })
            .collect();

        let mut output = Tensor::new(sigmoided.clone(), shape);
        let input_clone = input.clone();

        output.set_autograd(
            vec![input.clone()],
            Box::new(move |grad| {
                let mut ingrad = Vec::with_capacity(grad.len());
                // Derivative of sigmoid: sigma(x) * (1 - sigma(x))
                for (&g, &s) in grad.iter().zip(sigmoided.iter()) {
                    ingrad.push(g * s * (1.0 - s));
                }
                input_clone.add_grad(&ingrad);
            }),
        );

        output
    }
}
