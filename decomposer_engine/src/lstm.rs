use burn::{nn::{Linear, Lstm}, prelude::Backend, *};

struct NucLstmConfig{
    input_size: usize,
    output_size: usize,
    hidden_size: usize,
    num_layers: usize,
    dropout: f32,
}

struct NucLstm<B> where  B: Backend{
    model: Lstm<B>,
    output_model: Linear<B>,
}

impl <B: Backend>NucLstm<B>{
    fn forward(){

    }
}
