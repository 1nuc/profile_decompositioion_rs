use burn::{config::Config, module::Module, nn::{Linear, LinearConfig, Lstm, LstmConfig}, prelude::Backend, *};
use polars::prelude::last;

#[derive(Config, Debug)]
struct NucLstmConfig{
    input_size: usize,
    output_size: usize,
    hidden_size: usize,
    num_layers: usize,
    dropout: f32,
}
impl NucLstmConfig{
    pub fn init<B: Backend>(&self, device: B::Device) -> NucLstm<B>{
        let lstm=LstmConfig::new(self.input_size, self.hidden_size, false).with_batch_first(true);
        let linear=LinearConfig::new(self.hidden_size, self.output_size);
        NucLstm{
           model: lstm.init(&device),
           output_model: linear.init(&device)
        }
    }
}

#[derive(Module, Debug)]
struct NucLstm<B :Backend>{
    model: Lstm<B>,
    output_model: Linear<B>,
}

impl <B: Backend>NucLstm<B>{
    fn forward(&self, input: Tensor<B,3>) -> Tensor<B, 2>{
        let (output,_) =self.model.forward(input, None);
        let [batch_size, seq_length, hidden_size]=output.dims();
        let last_output=output.narrow(1, seq_length-1, 1).reshape([batch_size, hidden_size]);
        self.output_model.forward(last_output)
    }
}
