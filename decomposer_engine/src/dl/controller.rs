//TODO: Define the logic for training
//TODO: Define the logic for testing and validation
//TODO: Must make a function to split the dat into train, test and split
//TODO: steps:
//1. One struct that contains the data that will be red through the data engine
//2. define the default implementation for the controller, the data backend, and many more
//3. method for training and predicting
//4. catch the method for the metrics

use burn::{Tensor, backend::{Autodiff, Wgpu, wgpu::WgpuDevice}, module::{Module}, optim::AdamWConfig, prelude::Backend, tensor::backend::AutodiffBackend};
use polars::frame::DataFrame;
use crate::{EagerActions, dl::{inference::Inference, models::{lstm::NucLstmConfig, stacked_lstm::StackedLstmConfig}, training::NrelConfig}};

pub struct Controller{
    pub train_data: DataFrame,
    pub test_data: DataFrame,
    pub val_data: DataFrame,
}

impl Controller{

    pub fn new(data: DataFrame) -> Self{
        let (train_data, val_data, test_data)=data.train_val_test_spli();
        Self{
            train_data,
            test_data,
            val_data,
        }
    }

    pub fn lstm_simulation(&self){
        type Mybackend= Autodiff<Wgpu>;
        let device=WgpuDevice::DiscreteGpu(0);
        self.train_lstm::<Mybackend>(device.clone());
        self.infer_lstm::<Mybackend>(device);
    }

    pub fn train_lstm<B: AutodiffBackend>(&self,device: B::Device){
        let model=NucLstmConfig::default();
        let model_config=NrelConfig::new(model,AdamWConfig::new().with_weight_decay(1e-4));
        model_config.train::<B>(self.train_data.clone(), self.val_data.clone(), "lstm_artifact", device);
    }

    pub fn infer_lstm<B: AutodiffBackend>(&self,device: B::Device){
        Inference::inference::<B>("lstm_artifact", self.test_data.clone(), device);
    }
}
