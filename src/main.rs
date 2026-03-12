use decomposer_engine::{Actions, data_engine::*, preprocessor_engine::Preprocessor}; 
use polars::prelude::*;
use smartcore::{ensemble::random_forest_regressor::{self, RandomForestRegressor}, linalg::basic::{arrays::{Array, Array2}, matrix::DenseMatrix}, linear::linear_regression::LinearRegression, metrics::r2, xgboost::{XGRegressor, XGRegressorParameters}};
use tap::Conv;
fn main() {

    let data_source=Nrel::init();
    let mut data=data_source.data;
    let encoded_data=data.encode_categoricals().collect().unwrap().sample_n(&Series::from_any_values(5000).unwrap(), true, true, Some(6)).unwrap().lazy();
    let preprocessor=Preprocessor::new(encoded_data.clone(), 42, 0.3);
    let (x_train, x_test, y_train, y_test)=preprocessor.split_x_y();
    let y_train=y_train.select([col("out.electricity.AC.energy_consumption..kwh")]);
    let y_test=y_test.select([col("out.electricity.AC.energy_consumption..kwh")]);
    let x_train=DenseMatrix::from_2d_vec(&x_train.to_2d_vec()).expect("Error").transpose();
    let y_train=y_train.to_1d_vec();
    assert_eq!(x_train.shape().0, y_train.len());
    println!("{:?}", x_train.shape().0);
    println!("{:?}", y_train.len());
    let parameters=XGRegressorParameters::default().with_learning_rate(0.1).with_max_depth(4);
    let model=XGRegressor::fit(&x_train,&y_train, parameters).expect("Error in the model");
    let x_test=DenseMatrix::from_2d_vec(&x_test.to_2d_vec()).unwrap().transpose();
    let predicted=model.predict(&x_test).unwrap();
    let r2=r2(&y_test.to_1d_vec(), &predicted);
    println!("{:?}", r2);
}

    // let dense_file=DenseMatrix::from_2d_vec(&vec_t).unwrap();
    // let transformer =StandardScaler::fit(&dense_file, Default::default()).unwrap();
    // let transformed_data=transformer.transform(&dense_file);
