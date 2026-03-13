use decomposer_engine::{Actions, data_engine::*, preprocessor_engine::Preprocessor, xgboost::Xgb}; 
use polars::prelude::*;
use tap::Conv;

fn main() {
    let data_source=Nrel::init();
    let mut data=data_source.data;
    let encoded_data=data.encode_categoricals();
    let preprocessor=Preprocessor::new(encoded_data.clone(), 42, 0.3);
    let (mut x_train, mut x_test, mut y_train, y_test)=preprocessor.split_x_y();
    let mut cols=y_train.clone().collect_schema().unwrap().iter_names().map(|x|x.as_str().to_string()).collect::<Vec<String>>();

    let d_train=x_train.to_matrix(true);
    let d_test=x_test.to_matrix(true);
    let mut xgb=Xgb::new(d_train, d_test);
    let r2=xgb.apply_modelling(y_train, y_test);
}

