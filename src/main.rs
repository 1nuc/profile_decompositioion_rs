use std::{fs::File, ops::Sub};

use decomposer_engine::{Actions, data_engine::*, preprocessor_engine::Preprocessor}; 
use ndarray::Array2;
use polars::prelude::*;
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};
use tap::Conv;
fn main() {

    //TODO: Finish the train test and split function in polars Done
    //TODO: train the xgboost
    //TODO: Extract x and y features
    let data_source=Nrel::init();
    let mut data=data_source.data;
    let mut encoded_data=data.encode_categoricals();
    let preprocessor=Preprocessor::new(encoded_data.clone(), 42, 0.3);
    let (x_train, x_test, y_train, y_test)=preprocessor.split_x_y();
    println!("{:?}", x_train.collect().unwrap());
    println!("{:?}", x_test.collect().unwrap());
    println!("{:?}", y_train.collect().unwrap());
    println!("{:?}", y_test.collect().unwrap());
    // println!("The size of the full set of the data: {:?}", total_rows);
    // println!("the size of the testing set is: {:?}", test_rows);
    // println!("{:?}", test_t);
    //
    // println!("the size of the testing set is: {:?}", training_rows);
    // println!("{:?}", train_t);
}

    // let t: Array2<f32>=encoded_data.clone().collect().unwrap().to_ndarray::<Float32Type>(IndexOrder::C).expect("Error in converting to an array");
    // TODO: Convert polars to a vector
    // let ncols=t.ncols();
    // let c_t=t.clone().into_raw_vec_and_offset().0;
    // let vec_t=c_t.chunks(ncols).map(|x| x.to_vec()).collect::<Vec<Vec<f32>>>();
    // let dense_file=DenseMatrix::from_2d_vec(&vec_t).unwrap();
    // let transformer =StandardScaler::fit(&dense_file, Default::default()).unwrap();
    // let transformed_data=transformer.transform(&dense_file);
