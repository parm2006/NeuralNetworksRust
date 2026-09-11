use super::Module;
use crate::tensor::Tensor;


pub struct Softmax{}

impl Softmax{
    pub fn new() -> Self{
        Self {}
    }
}

impl Module for Softmax{
    fn forward(&self, input: &Tensor) -> Tensor{
        let expsum = input.data().exp().sum(1,false);
        let mut output = Tensor::new(input.data().clone().div(&expsum), input.shape());

        output.set_autograd(
            vec![input.clone()],
            Box::new(move |grad|{
                let mut in_grad = grad.to_vec()
                let dotp: f32 = grad.iter().zip(output._data.iter()).map(|g,s| g*s).sum();
                for (g, i) in in_grad.iter_mut().zip(output._data.iter()){
                    *g = i * (*g - dot);
                }
                input.add_grad(&in_grad);
            })
        )
        output

    }
}