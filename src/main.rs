use decomposer_engine::{Actions, data_engine::*, preprocessor_engine::Preprocessor, xgb::Xgb, lstm::*}; 
use polars::prelude::*;
use tap::Conv;

fn main() {
    let data_source=Nrel::init();
    let mut data=data_source.data;
    let encoded_data=data.encode_categoricals();
    let options=DynamicGroupOptions{
        index_column: PlSmallStr::from_str("timestamp"),
        every: Duration::parse("1d"),
        period: Duration::parse("1d"),
        offset: Duration::parse("1d"),
        ..Default::default()
    };
    let a= encoded_data.sort(
        vec![PlSmallStr::from_str("timestamp")], Default::default()
        ).group_by_dynamic(col("timestamp"), [], options).agg([col("*")]).collect().unwrap();
    println!("{:?}", a);

    // let preprocessor=Preprocessor::new(encoded_data.clone(), 42, 0.3);
    
    // let (mut x_train, mut x_test, mut y_train, y_test)=preprocessor.split_x_y();
    // let d_train=x_train.to_matrix(true);
    // let d_test=x_test.to_matrix(true);
    // let mut xgb=Xgb::new(d_train, d_test);
    // let mean=xgb.train(y_train, y_test).evaluate();
    // println!("r2 is: {:?}", mean);
}

