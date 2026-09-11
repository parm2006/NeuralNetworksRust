/// Base trait for all optimizers.
pub trait Optimizer {
    /// Performs a single optimization step (parameter update).
    fn step(&mut self);

    /// Resets all tracked parameter gradients to zero.
    fn zero_grad(&self);
}
