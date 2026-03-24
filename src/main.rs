use decomposer_engine::{Actions, EagerActions, data_engine::*, dl::{controller::Controller, models::lstm::{self, NucLstmConfig}, training::NrelConfig}, preprocessor_engine::Preprocessor, xgb::Xgb}; 
use polars::prelude::*;

fn main(){
    let data_source=Nrel::init();
    let data=data_source.data;
    let mut encoded_data=data.clone().encode_categoricals();
    let s=encoded_data.clone().collect().unwrap();
    let y_columns=s.return_y_columns();
    let modelling_data=encoded_data.standard_scalar(y_columns.clone()).return_time_sequenced().collect().unwrap();
    let control=Controller::new(modelling_data);
    control.lstm_simulation();
    // let preprocessor=Preprocessor::new(encoded_data.clone(), 42, 0.3);
    // let (mut x_train, mut x_test, mut y_train, y_test)=preprocessor.split_x_y();
    // let d_train=x_train.to_matrix(true);
    // let d_test=x_test.to_matrix(true);
    // let mut xgb=Xgb::new(d_train, d_test);
    // let mean=xgb.train(y_train, y_test).evaluate();
    // println!("r2 is: {:?}", mean);
}

