use super::Module;
use crate::tensor::Tensor;

/// A Sequential container that chains multiple `Module` layers together.
///
/// Layers are executed in the exact order they are passed in.
/// `Sequential` automatically handles recursive parameter tracking (`parameters()`),
/// gradient clearing (`zero_grad()`), and binary checkpointing (`save()` / `load()`).
///
/// # Usage Example
/// ```rust
/// use NeuralNetworksRust::{Flatten, Linear, Relu, Sequential, Tensor, Module};
///
/// // Build a 4-layer MNIST classifier in 6 lines of code:
/// let model = Sequential::new(vec![
///     Box::new(Flatten::new(1, None)),
///     Box::new(Linear::new(28 * 28, 128)),
///     Box::new(Relu::new()),
///     Box::new(Linear::new(128, 10)),
/// ]);
///
/// // Forward pass:
/// let x = Tensor::random(vec![4, 28, 28]);
/// let y = model.forward(&x);
/// assert_eq!(y.shape(), vec![4, 10]);
///
/// // All layer weights and biases are tracked automatically:
/// for (name, param) in model.parameters() {
///     println!("Parameter: {} -> shape: {:?}", name, param.shape());
///     // e.g. "1.weights" -> [784, 128], "1.bias" -> [128], etc.
/// }
/// ```
pub struct Sequential {
    pub layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    /// Creates a new `Sequential` container with a list of boxed `Module` layers.
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Self { layers }
    }

    /// Appends a new layer to the end of the sequential container.
    pub fn add(&mut self, layer: Box<dyn Module>) {
        self.layers.push(layer);
    }
}

impl Module for Sequential {
    /// Passes the input tensor sequentially through each layer in the container:
    /// output = layer_n(...(layer_1(layer_0(input))))
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut current = input.clone();
        for layer in &self.layers {
            current = layer.forward(&current);
        }
        current
    }

    /// Automatically registers each sub-module with its numerical index ("0", "1", ...),
    /// enabling recursive parameter tracking and state_dict serialization.
    fn modules(&self) -> Vec<(String, &dyn Module)> {
        self.layers
            .iter()
            .enumerate()
            .map(|(i, m)| (i.to_string(), m.as_ref()))
            .collect()
    }
}
