use std::{fs::File, ops::Sub};

use decomposer_engine::{Actions, data_engine::*}; 
use ndarray::Array2;
use polars::prelude::*;
use tap::Conv;
fn main() {

    let data_source=Nrel::init();
    let mut data=data_source.data;
    let mut encoded_data=data.encode_categoricals();
    let total_rows= encoded_data.collect().unwrap().height();
    let n_rows=total_rows * 0.2 as usize;
    let arr: Vec<usize>= (0..total_rows).collect();
    let t_arr=&arr[..n_rows];
    let r=ChunkedArray::from_slice("new".into(), t_arr);
    let train_t=encoded_data.collect().unwrap().take(&r);
}

    // let t: Array2<f32>=encoded_data.clone().collect().unwrap().to_ndarray::<Float32Type>(IndexOrder::C).expect("Error in converting to an array");
    // TODO: Convert polars to a vector
    // let ncols=t.ncols();
    // let c_t=t.clone().into_raw_vec_and_offset().0;
    // let vec_t=c_t.chunks(ncols).map(|x| x.to_vec()).collect::<Vec<Vec<f32>>>();
    // let dense_file=DenseMatrix::from_2d_vec(&vec_t).unwrap();
    // let transformer =StandardScaler::fit(&dense_file, Default::default()).unwrap();
    // let transformed_data=transformer.transform(&dense_file);
