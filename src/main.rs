use std::{fs::read_dir, path::PathBuf};

use decomposer_engine::{Actions, EagerActions, data_engine::*, dl::controller::{self, Controller}, xgb};

fn main(){
    let dir=read_dir("../../datasets/").unwrap();
    let mut files=dir.map(|x| x.unwrap().path()
        ).collect::<Vec<PathBuf>>();
    let name="123";
    let building=format!("{name}-28.parquet").as_str();
    let file=files.pop();
    println!("{:?}", file.unwrap().file_name());
    // controller::run_train();
    // controller::process_chunks();
    // let data_source=Nrel::init();
    // let data=data_source.data;
    // let mut encoded_data=data.clone().encode_categoricals();
    //
    // // Deep Learning Model
    // let s=encoded_data.clone().collect().unwrap();
    // let y_columns=s.return_y_columns();
    // let modelling_data=encoded_data.standard_scalar(y_columns.clone()).return_time_sequenced().collect().unwrap();
    // let control=Controller::new(modelling_data);
    // control.lstm_simulation();
    // --- Xgboost Model
    // xgb::Xgb::runner(encoded_data);

}

