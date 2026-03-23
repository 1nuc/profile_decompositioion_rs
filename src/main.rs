use decomposer_engine::{Actions, EagerActions, data_engine::*, dl::{models::lstm::{self, NucLstmConfig}, training::NrelConfig}, preprocessor_engine::Preprocessor, xgb::Xgb}; 
use ndarray::{Array3, s};
use polars::prelude::*;
use tap::Conv;
use burn::{Tensor, backend::{self, Autodiff, Wgpu, wgpu::WgpuDevice}, optim::AdamWConfig, prelude::Backend, tensor::{Int, TensorData}, train};

fn main(){
    let data_source=Nrel::init();
    let mut data=data_source.data;
    let encoded_data=data.clone().encode_categoricals().encode_categoricals().return_time_sequenced().collect().unwrap();
    let data_fraction=encoded_data.clone().sample_frac(
        &Series::new("fraction".into(),[1]), false, false, Some(42)).unwrap();
    let validation_size=encoded_data.clone().height() as f32 * 0.3;
    let train_size=encoded_data.clone().height() as f32 * 0.7;
    let train_data=encoded_data.clone().head(Some(train_size as usize));
    let test_data=encoded_data.clone().tail(Some(validation_size as usize));
    type Mybackend= Autodiff<Wgpu>;
    let device=WgpuDevice::DiscreteGpu(0);
    let model=NucLstmConfig::default();
    let model_config=NrelConfig::new(model,AdamWConfig::new().with_weight_decay(1e-4));
    model_config.train::<Mybackend>(train_data, test_data, "artifact_dir", device);
    // let train_data=data_fraction.tail(length)
    // let preprocessor=Preprocessor::new(encoded_data.clone(), 42, 0.3);
    // let (mut x_train, mut x_test, mut y_train, y_test)=preprocessor.split_x_y();
    // let d_train=x_train.to_matrix(true);
    // let d_test=x_test.to_matrix(true);
    // let mut xgb=Xgb::new(d_train, d_test);
    // let mean=xgb.train(y_train, y_test).evaluate();
    // println!("r2 is: {:?}", mean);
}

