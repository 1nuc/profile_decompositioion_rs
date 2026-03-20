use decomposer_engine::{Actions, EagerActions, data_engine::*, lstm::*, preprocessor_engine::Preprocessor, xgb::Xgb}; 
use ndarray::{Array3, s};
use polars::prelude::*;
use tap::Conv;
use burn::{Tensor, backend::{self, wgpu::WgpuDevice}, prelude::Backend, tensor::TensorData};

fn main(){
    let data_source=Nrel::init();
    let mut data=data_source.data;
    let encoded_data=data.clone().encode_categoricals().return_time_sequenced();
    let d=encoded_data.collect().unwrap();
    let binding=d.clone();
    let cols=binding.return_y_columns();
    let samples=d.height();
    let sequenced_data=d.select_sequence(cols.clone(), samples);
    let arr=sequenced_data.slice(s![3,..,..]).to_owned().into_raw_vec_and_offset().0.chunks(cols.len()).map(|x| x.to_vec()).collect::<Vec<Vec<f32>>>();
    let device=WgpuDevice::default();
    type MyBackend=backend::Wgpu<f32,i32>;
    let tensor=Tensor::<MyBackend, 2>::from_data(TensorData::new(arr,[1,96,cols.len()]), &device);
    println!("{:?}",tensor);
    // let preprocessor=Preprocessor::new(encoded_data.clone(), 42, 0.3);
    // let (mut x_train, mut x_test, mut y_train, y_test)=preprocessor.split_x_y();
    // let d_train=x_train.to_matrix(true);
    // let d_test=x_test.to_matrix(true);
    // let mut xgb=Xgb::new(d_train, d_test);
    // let mean=xgb.train(y_train, y_test).evaluate();
    // println!("r2 is: {:?}", mean);
}

