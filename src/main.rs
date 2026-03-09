use std::{fs::File, ops::Sub};

use decomposer_engine::{Actions, data_engine::*}; 
use ndarray::Array2;
use polars::prelude::*;
use smartcore::{api::{Transformer, UnsupervisedEstimator}, linalg::basic::matrix::DenseMatrix, preprocessing::numerical::StandardScaler};
use tap::Conv;
fn main() {

    let data_source=Nrel::init();
    let mut data=data_source.data;
    let encoded_data=data.encode_categoricals();
    // let std_dev=encoded_data.clone().with_columns([col("*").std(1)]); //according to the library 1 is the
    //                                                          //standard
    // let mean: Vec<f32>=encoded_data.clone().with_columns([col("*").mean()]).collect().unwrap().clone().get(0).unwrap();
    //
    // let transformed_data=encoded_data.with_columns([col(PlSmallStr::from_static("*")).sub(lit(mean))]);
    // // println!("{:?}", mean.unwrap().get(0));

    let standard_scalar_native=encoded_data.clone().with_columns([(col("*") - col("*").mean()) / (col("*").std(1))]);
    let min_max_native=encoded_data.clone().with_columns([
        when(col("*").max().eq(col("*").min())
            ).then(0).otherwise((col("*") - col("*").min()) / (col("*").max() - col("*").min()))]);
    print!("{:?}", min_max_native.collect().unwrap());
}

    // let t: Array2<f32>=encoded_data.clone().collect().unwrap().to_ndarray::<Float32Type>(IndexOrder::C).expect("Error in converting to an array");
    // TODO: Convert polars to a vector
    // let ncols=t.ncols();
    // let c_t=t.clone().into_raw_vec_and_offset().0;
    // let vec_t=c_t.chunks(ncols).map(|x| x.to_vec()).collect::<Vec<Vec<f32>>>();
    // let dense_file=DenseMatrix::from_2d_vec(&vec_t).unwrap();
    // let transformer =StandardScaler::fit(&dense_file, Default::default()).unwrap();
    // let transformed_data=transformer.transform(&dense_file);
