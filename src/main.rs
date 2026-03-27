use std::{ffi::OsString, fs::{read_dir}};

use decomposer_engine::{Actions, EagerActions, data_engine::*, dl::controller::Controller, xgb};

fn main(){
    // let data_source=Nrel::init();
    // let data=data_source.data;
    // let mut encoded_data=data.clone().encode_categoricals();
    let dir=read_dir("../../datasets").unwrap();
    
    let file_names=dir.map(|x| x.unwrap().file_name()
        ).map(|x| x.to_str().unwrap().to_owned()).collect::<Vec<String>>();
    println!("{:?}", file_names.len());
    // --- Xgboost Model
    // xgb::Xgb::runner(encoded_data);

    // ---- Deep learning Models
    // let s=encoded_data.clone().collect().unwrap();
    // let y_columns=s.return_y_columns();
    // let modelling_data=encoded_data.standard_scalar(y_columns.clone()).return_time_sequenced().collect().unwrap();
    // let control=Controller::new(modelling_data);
    // control.lstm_simulation();
}

