use NeuralNetworksRust::{tensor::Tensor, Module, Flatten, Linear, Relu};




pub struct NeuralNet{
    pub flatten: Flatten,
    pub linear1: Linear,
    pub relu: Relu,
    pub linear2: Linear,
}

impl NeuralNet{
    pub fn new() -> Self {
        Self{
            flatten: Flatten::new(1,None),
            linear1: Linear::new(28*28, 128),
            relu: Relu::new(),
            linear2: Linear::new(128, 10),
        }
    }
}

impl Module for NeuralNet{
    fn forward(&self, input: &Tensor) -> Tensor{
        let x = self.flatten.forward(input);
        let x = self.linear1.forward(&x);
        let x = self.relu.forward(&x);
        let x = self.linear2.forward(&x);
        x
    }

    fn modules(&self) -> Vec<(String, &dyn Module)> {
        vec![
            ("flatten".to_string(), &self.flatten),
            ("linear1".to_string(), &self.linear1),
            ("relu".to_string(), &self.relu),
            ("linear2".to_string(), &self.linear2),
        ]
    }
}



fn main() {
    let model = NeuralNet::new();
    let X = Tensor::random(vec![4,28,28]);
    let Y = model.forward(&X);
    println!("{}",Y);
    assert_eq!(Y.shape(), [4,10]);
    

}
