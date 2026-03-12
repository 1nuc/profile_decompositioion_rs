use decomposer_engine::{Actions, data_engine::*, preprocessor_engine::Preprocessor}; 
use polars::prelude::*;
use xgb::{parameters::{BoosterParametersBuilder, learning::{self, LearningTaskParametersBuilder}, linear::LinearBoosterParametersBuilder, tree::{Predictor, TreeBoosterParametersBuilder}}, *};
use tap::Conv;
fn main() {

    let data_source=Nrel::init();
    let mut data=data_source.data;
    let encoded_data=data.encode_categoricals();
    let preprocessor=Preprocessor::new(encoded_data.clone(), 42, 0.3);
    let (mut x_train, mut x_test, y_train, y_test)=preprocessor.split_x_y();
    let train_n=x_train.clone().collect().unwrap().height();
    let test_n=x_test.clone().collect().unwrap().height();
    let mut d_train=DMatrix::from_dense(&x_train.standard_scalar().to_1d_vec(), train_n).unwrap();
    d_train.set_labels(&y_train.to_1d_vec()).unwrap();

    let mut d_test=DMatrix::from_dense(&x_test.standard_scalar().to_1d_vec(), test_n).unwrap();
    d_test.set_labels(&y_test.to_1d_vec()).unwrap();
    
    let tree_param=TreeBoosterParametersBuilder::default().max_depth(7).subsample(0.7).eta(0.1).build().unwrap();
    let learning_param=LearningTaskParametersBuilder::default().objective(parameters::learning::Objective::RegLinear).build().unwrap();
    let eval_set=&[(&d_train, "train"), (&d_test, "test")];
    let booster_param=BoosterParametersBuilder::default().booster_type(
        parameters::BoosterType::Tree(tree_param)).threads(None).learning_params(learning_param).build().unwrap();
    let parameters=parameters::TrainingParametersBuilder::default().
        dtrain(&d_train).booster_params(booster_param).
        boost_rounds(100).evaluation_sets(Some(eval_set)).build().unwrap();

    let bst=Booster::train(&parameters).unwrap();
    let predicted=bst.predict(&d_test).unwrap();
    println!("{:?}", predicted);
}

    // let dense_file=DenseMatrix::from_2d_vec(&vec_t).unwrap();
    // let transformer =StandardScaler::fit(&dense_file, Default::default()).unwrap();
    // let transformed_data=transformer.transform(&dense_file);
