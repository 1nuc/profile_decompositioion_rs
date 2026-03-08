use decomposer_engine::{Actions, data_engine::*}; 
use ndarray::prelude::*;
use linfa_preprocessing::linear_scaling::*;
use polars::prelude::*;
use tap::Conv;
fn main() {

    let data_source=Nrel::init();
    let mut data=data_source.data;
    let encoded_data=data.encode_categoricals();
    let t =encoded_data.clone().collect().unwrap().to_ndarray::<Float32Type>(Default::default()).expect("Error in converting to an array");
    println!("")
}
