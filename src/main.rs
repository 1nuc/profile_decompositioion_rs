use decomposer_engine::{Actions, data_engine::*, preprocessor_engine::Preprocessor}; 
use smartcore::{linalg::basic::{arrays::Array2, matrix::DenseMatrix}, metrics::r2, xgboost::{self, XGRegressor, XGRegressorParameters}};
use ndarray::Array2;
use polars::prelude::*;
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};
use tap::Conv;
fn main() {

    //TODO: Finish the train test and split function in polars Done
    //TODO: train the xgboost
    //TODO: Extract x and y features Done
    let data_source=Nrel::init();
    let mut data=data_source.data;
    let mut encoded_data=data.encode_categoricals();
    let preprocessor=Preprocessor::new(encoded_data.clone(), 42, 0.3);
    let (x_train, x_test, y_train, y_test)=preprocessor.split_x_y();
    let t: Array2<f32>=x_train.clone().collect().unwrap().to_ndarray::<Float32Type>(IndexOrder::C).expect("Error in converting to an array");
    // TODO: Convert polars to a vector
    let ncols=t.ncols();
    let c_t=t.clone().into_raw_vec_and_offset().0;
    let vec_t=c_t.chunks(ncols).map(|x| x.to_vec()).collect::<Vec<Vec<f32>>>();
    let x_train_=DenseMatrix::from_2d_vec(&vec_t).unwrap();

    let t_y: Array2<f32>=y_train.clone().collect().unwrap().to_ndarray::<Float32Type>(IndexOrder::C).expect("Error in converting to an array");
    // TODO: Convert polars to a vector
    let ncols_y=t_y.ncols();
    let c_y=t_y.clone().into_raw_vec_and_offset().0;
    let vec_y=c_y.chunks(ncols_y).map(|x| x.to_vec()).collect::<Vec<Vec<f32>>>();
    let y_train_=DenseMatrix::from_2d_vec(&vec_y).unwrap();

    let parameters=XGRegressorParameters::default().with_n_estimators(50).with_max_depth(3).with_learning_rate(0.1);

    let model=XGRegressor::fit(&x_train_, &c_y, parameters).expect("Error with the model");

    let t_t: Array2<f32>=x_test.clone().collect().unwrap().to_ndarray::<Float32Type>(IndexOrder::C).expect("Error in converting to an array");
    let ncols=t_t.ncols();
    let t_t_v=t_t.clone().into_raw_vec_and_offset().0;
    let vec_test=t_t_v.chunks(ncols).map(|x| x.to_vec()).collect::<Vec<Vec<f32>>>();
    let x_test=DenseMatrix::from_2d_vec(&vec_test).unwrap();

    let predicted=model.predict(&x_test).unwrap();


    let y_predict=DenseMatrix::from(&predicted);

    let y_t: Array2<f32>=y_test.clone().collect().unwrap().to_ndarray::<Float32Type>(IndexOrder::C).expect("Error in converting to an array");
    let ncols=y_t.ncols();
    let y_t_v=y_t.clone().into_raw_vec_and_offset().0;
    let vec_y_test=y_t_v.chunks(ncols).map(|x| x.to_vec()).collect::<Vec<Vec<f32>>>();
    let y_test=DenseMatrix::from_2d_vec(&vec_y_test).unwrap();

    let r_score=r2(&y_test, &predicted);


}

    // let dense_file=DenseMatrix::from_2d_vec(&vec_t).unwrap();
    // let transformer =StandardScaler::fit(&dense_file, Default::default()).unwrap();
    // let transformed_data=transformer.transform(&dense_file);
