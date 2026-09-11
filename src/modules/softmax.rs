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
        let shape = input.shape();
        let c = *shape.last().unwrap_or(&1);
        let data = input.data();

        let mut softmaxed = Vec::with_capacity(data.len());

        for chunk in data.chunks(c){
            let maxval = chunk.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
            let start = softmaxed.len();
            let mut sum = 0.0;

            for &x in chunk {
                let e= (x-maxval).exp();
                sum+=e;
                softmaxed.push(e);
            }
            for val in &mut softmaxed[start..]{
                *val/=sum;
            }
        }
        
        
        let mut output = Tensor::new(softmaxed.clone(), shape);
        let input_clone = input.clone();


        output.set_autograd(
            vec![input.clone()],
            Box::new(move |grad| {
                let mut ingrad = Vec::with_capacity(grad.len());
                for (g_chunk, s_chunk) in grad.chunks(c).zip(softmaxed.chunks(c)) {
                    let dot:f32 = g_chunk.iter().zip(s_chunk.iter()).map(|(&g,&s)| g*s).sum();
                    for (&g,&s) in g_chunk.iter().zip(s_chunk.iter()){
                        ingrad.push(s*(g-dot));
                    }
                }
                input_clone.add_grad(&ingrad);
            })
        );

        output
        

    }
}

