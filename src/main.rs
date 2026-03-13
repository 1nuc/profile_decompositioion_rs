use decomposer_engine::{Actions, data_engine::*, preprocessor_engine::Preprocessor, xgboost::Xgb}; 
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
    let d_train=x_train.to_matrix(true);
    let d_test=x_test.to_matrix(true);
    let mut xgb=Xgb::new(d_train, d_test);
    xgb.set_y_train(y_train.to_1d_vec());
    xgb.set_y_test(y_test.to_1d_vec());
    let param=xgb.set_training_param();

    let bst=Booster::train(&param).unwrap();
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
