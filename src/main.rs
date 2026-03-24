use decomposer_engine::{Actions, EagerActions, data_engine::*, dl::{controller::Controller, models::lstm::{self, NucLstmConfig}, training::NrelConfig}, preprocessor_engine::Preprocessor, xgb::Xgb}; 
use ndarray::{Array3, s};
use polars::prelude::*;
use tap::Conv;
use burn::{Tensor, backend::{self, Autodiff, Wgpu, wgpu::WgpuDevice}, optim::AdamWConfig, prelude::Backend, tensor::{Int, TensorData}, train};

fn main(){
    let data_source=Nrel::init();
    let mut data=data_source.data;
    let encoded_data=data.clone().encode_categoricals().standard_scalar().return_time_sequenced().collect().unwrap();
    let control=Controller::new(encoded_data);
    control.lstm_simulation();
    // let preprocessor=Preprocessor::new(encoded_data.clone(), 42, 0.3);
    // let (mut x_train, mut x_test, mut y_train, y_test)=preprocessor.split_x_y();
    // let d_train=x_train.to_matrix(true);
    // let d_test=x_test.to_matrix(true);
    // let mut xgb=Xgb::new(d_train, d_test);
    // let mean=xgb.train(y_train, y_test).evaluate();
    // println!("r2 is: {:?}", mean);
}

