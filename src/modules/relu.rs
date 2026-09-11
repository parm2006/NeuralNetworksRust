use super::Module;
use crate::tensor::Tensor;

pub struct Relu {}

impl Relu {
    pub fn new() -> Self {
        Self {}
    }
}

impl Module for Relu {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut outputdata = input.data();
        for i in outputdata.iter_mut() {
            *i = (*i).max(0.0);
        }
        let mut output = Tensor::new(outputdata.clone(), input.shape());

        let input_clone = input.clone();
        output.set_autograd(
            vec![input.clone()],
            Box::new(move |grad| {
                let mut in_grad = grad.to_vec();
                for (g, &out) in in_grad.iter_mut().zip(outputdata.iter()) {
                    if out == 0.0 {
                        *g = 0.0;
                    }
                }
                input_clone.add_grad(&in_grad);
            }),
        );
        output
    }
}
