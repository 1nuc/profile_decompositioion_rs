use decomposer_engine::{Actions, data_engine::*, preprocessor_engine::Preprocessor}; 
use polars::prelude::*;
use xgboost::{parameters::{BoosterParametersBuilder, TrainingParametersBuilder, learning::{self, EvaluationMetric, LearningTaskParametersBuilder}, tree::TreeBoosterParametersBuilder}, *};
use tap::Conv;

fn main() {
    let data_source=Nrel::init();
    let mut data=data_source.data;
    let encoded_data=data.encode_categoricals();
    let preprocessor=Preprocessor::new(encoded_data.clone(), 42, 0.3);
    let (mut x_train, mut x_test, y_train, y_test)=preprocessor.split_x_y();
    let y_train=y_train.select([col("out.electricity.AC.energy_consumption..kwh")]);
    let y_test=y_test.select([col("out.electricity.AC.energy_consumption..kwh")]);
    let train_n=x_train.clone().collect().unwrap().height();
    let test_n=x_test.clone().collect().unwrap().height();
    let mut d_train=DMatrix::from_dense(&x_train.standard_scalar().to_1d_vec(), train_n).unwrap();
    d_train.set_labels(&y_train.to_1d_vec()).unwrap();
    let mut d_test=DMatrix::from_dense(&x_test.standard_scalar().to_1d_vec(), test_n).unwrap();
    d_test.set_labels(&y_test.to_1d_vec()).unwrap();

    let tree_param=TreeBoosterParametersBuilder::default().eta(0.1).subsample(0.7).build().unwrap();
    let learning_param=LearningTaskParametersBuilder::default().eval_metrics(
        learning::Metrics::Custom(vec![EvaluationMetric::MAE])).objective(parameters::learning::Objective::RegLinear).build().unwrap();
    let eval_set=&[(&d_train, "train"), (&d_test, "test")];
    let booster_param=BoosterParametersBuilder::default().booster_type(
        parameters::BoosterType::Tree(tree_param)).threads(None).learning_params(learning_param).build().unwrap();
    let parameters=parameters::TrainingParametersBuilder::default().
        dtrain(&d_train).booster_params(booster_param).
        boost_rounds(100).build().unwrap();

    let bst=Booster::train(&parameters).unwrap();
    let preds=bst.predict(&d_test).unwrap();
    let y_true = d_test.get_labels().unwrap();
    let mean = y_true.iter().sum::<f32>() / y_true.len() as f32;
    let ss_tot: f32 = y_true.iter().map(|y| (y - mean).powi(2)).sum();
    let ss_res: f32 = y_true.iter().zip(preds.iter()).map(|(y, p)| (y - p).powi(2)).sum();
    let r2 = 1.0 - ss_res / ss_tot;
    println!("R2: {}", r2);
}

    // let dense_file=DenseMatrix::from_2d_vec(&vec_t).unwrap();
    // let transformer =StandardScaler::fit(&dense_file, Default::default()).unwrap();
    // let transformed_data=transformer.transform(&dense_file);
