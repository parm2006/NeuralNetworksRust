use super::Module;
use crate::tensor::Tensor;

pub struct Flatten {
    startdim : usize,
    enddim : Option<usize>,
}

impl Flatten{
    pub fn new(startdim: usize, enddim: Option<usize>) -> Self {
        Self { startdim, enddim }
    }
}

impl Module for Flatten {    
    fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        if shape.is_empty(){
            return Tensor::from(input.data());
        }
        if shape.len() <= self.startdim{
            panic!("Shape length must be greater than startdim");
        }
        
        let end = self.enddim.unwrap_or(shape.len()-1);
        let mut newshape = shape[..self.startdim].to_vec();
        let flat:usize = shape[self.startdim..=end].iter().product();
        newshape.push(flat);
        newshape.extend_from_slice(&shape[end+1..]);
        let mut output = Tensor::new(input.data(), newshape);
        let in_clone = input.clone();
        output.set_autograd(
            vec![input.clone()],
            Box::new(move |grad| {
                in_clone.add_grad(grad);
            }),
        );
        output
    }
}