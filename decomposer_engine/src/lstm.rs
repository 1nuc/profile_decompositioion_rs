use burn::{backend::{Autodiff, Wgpu}, config::Config, data::dataloader::DataLoaderBuilder, module::Module, nn::{Linear, LinearConfig, Lstm, LstmConfig}, optim::AdamWConfig, prelude::Backend, train::{Learner, SupervisedTraining}, *};
use polars::prelude::last;

use crate::{Actions, data_engine::Nrel, preprocessor_engine::Preprocessor};

#[derive(Clone)]

pub struct Mybatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> Mybatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Config, Debug)]
pub struct NucLstmConfig{
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
    fn train(){
        type Mybackend=Wgpu<f32, i32>;
        let data_source=Nrel::init();
        let mut data=data_source.data;
        let encoded_data=data.encode_categoricals();
        let preprocessor=Preprocessor::new(encoded_data.clone(), 42, 0.3);
        let (mut x_train, mut x_test, mut y_train, y_test)=preprocessor.split_x_y();
        let device=Default::default();
        let tensor_train=Tensor::<Mybackend,2>::from_data(x_train.to_ndarry().as_slice().unwrap(), &device);
        let tensor_test=Tensor::<Mybackend,2>::from_data(x_test.to_ndarry().as_slice().unwrap(), &device);
        let model=NucLstmConfig::new(preprocessor.x_labels_size, preprocessor.y_labels_size, 20, 2, 0.0).init::<Mybackend>(device.clone());
        for i in 0..50{
           let preds=model.forward(tensor_train); 
        }
    }
}
