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
        
        

    }
}