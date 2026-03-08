use decomposer_engine::{Actions, data_engine::*}; 
use ndarray::Array2;
use polars::prelude::*;
use smartcore::{api::{Transformer, UnsupervisedEstimator}, linalg::basic::matrix::DenseMatrix, preprocessing::numerical::StandardScaler};
use tap::Conv;
fn main() {

    let data_source=Nrel::init();
    let mut data=data_source.data;
    let encoded_data=data.encode_categoricals();
    println!("{:?}",transformed_data);
}

    // let t: Array2<f32>=encoded_data.clone().collect().unwrap().to_ndarray::<Float32Type>(IndexOrder::C).expect("Error in converting to an array");
    // let ncols=t.ncols();
    // let c_t=t.clone().into_raw_vec_and_offset().0;
    // let vec_t=c_t.chunks(ncols).map(|x| x.to_vec()).collect::<Vec<Vec<f32>>>();
    // let dense_file=DenseMatrix::from_2d_vec(&vec_t).unwrap();
    // let transformer =StandardScaler::fit(&dense_file, Default::default()).unwrap();
    // let transformed_data=transformer.transform(&dense_file);
