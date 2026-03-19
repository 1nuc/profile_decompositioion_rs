use decomposer_engine::{Actions, EagerActions, data_engine::*, lstm::*, preprocessor_engine::Preprocessor, xgb::Xgb}; 
use polars::prelude::*;
use tap::Conv;

fn main() {
    let data_source=Nrel::init();
    let mut data=data_source.data;
    let encoded_data=data.encode_categoricals().return_time_sequenced();
    let d=encoded_data.collect().unwrap();
    let cols=d.return_x_columns();
    println!("{:?}", d.select_sequence(cols).get_column_names());
    // let preprocessor=Preprocessor::new(encoded_data.clone(), 42, 0.3);
    // let (mut x_train, mut x_test, mut y_train, y_test)=preprocessor.split_x_y();
    // let d_train=x_train.to_matrix(true);
    // let d_test=x_test.to_matrix(true);
    // let mut xgb=Xgb::new(d_train, d_test);
    // let mean=xgb.train(y_train, y_test).evaluate();
    // println!("r2 is: {:?}", mean);
}

