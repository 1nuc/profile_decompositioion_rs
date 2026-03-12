use decomposer_engine::{Actions, data_engine::*, preprocessor_engine::Preprocessor}; 
use polars::prelude::*;
use xgb::{parameters::{BoosterParametersBuilder, learning::{self, LearningTaskParametersBuilder}}, *};
use tap::Conv;
fn main() {

    let data_source=Nrel::init();
    let mut data=data_source.data;
    let encoded_data=data.encode_categoricals();
    let preprocessor=Preprocessor::new(encoded_data.clone(), 42, 0.3);
    let (x_train, x_test, y_train, y_test)=preprocessor.split_x_y();
    // let y_train=y_train.select([col("out.electricity.AC.energy_consumption..kwh")]);
    // let y_test=y_test.select([col("out.electricity.AC.energy_consumption..kwh")]);
    let train_n=x_train.clone().collect().unwrap().height();
    let test_n=x_test.clone().collect().unwrap().height();
    let mut d_train=DMatrix::from_dense(&x_train.to_1d_vec(), train_n).unwrap();
    d_train.set_labels(&y_train.to_1d_vec()).unwrap();

    let mut d_test=DMatrix::from_dense(&x_test.to_1d_vec(), test_n).unwrap();
    d_test.set_labels(&y_test.to_1d_vec()).unwrap();
    
    let learning_param=LearningTaskParametersBuilder::default().objective(parameters::learning::Objective::RegLinear).build().unwrap();
    let eval_set=&[(&d_train, "train"), (&d_test, "test")];

    let booster_param=BoosterParametersBuilder::default().threads(None).learning_params(learning_param).build().unwrap();
    let parameters=parameters::TrainingParametersBuilder::default().
        dtrain(&d_train).booster_params(booster_param).boost_rounds(100).build().unwrap();

    let bst=Booster::train(&parameters).unwrap();
    println!("{:?}", bst.predict(&d_test).unwrap());

    // assert_eq!(x_train.shape().0, y_train.len());
    // println!("{:?}", x_train.shape().0);
    // println!("{:?}", y_train.len());
    // let parameters=XGRegressorParameters::default().with_learning_rate(0.1).with_max_depth(4);
    // let model=XGRegressor::fit(&x_train,&y_train, parameters).expect("Error in the model");
    // let x_test=DenseMatrix::from_2d_vec(&x_test.to_2d_vec()).unwrap().transpose();
    // let predicted=model.predict(&x_test).unwrap();
    // let r2=r2(&y_test.to_1d_vec(), &predicted);
    // println!("{:?}", r2);
}

    // let dense_file=DenseMatrix::from_2d_vec(&vec_t).unwrap();
    // let transformer =StandardScaler::fit(&dense_file, Default::default()).unwrap();
    // let transformed_data=transformer.transform(&dense_file);
