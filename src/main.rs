use std::{fs::{self, copy, create_dir, read_dir}, intrinsics::breakpoint, path::{Path, PathBuf}};

use decomposer_engine::{Actions, EagerActions, data_engine::*, dl::controller::Controller, xgb};

fn main(){
    // let data_source=Nrel::init();
    // let data=data_source.data;
    // let mut encoded_data=data.clone().encode_categoricals();
    let dir=read_dir("../../datasets").unwrap();
    
    let mut files=dir.map(|x| x.unwrap().path()
        ).collect::<Vec<PathBuf>>();


    let input_path=Path::new("input");
    if !input_path.exists(){
        let input_lib=create_dir(input_path).unwrap();
    }
    // Join the file names first
    // then copy the content of the files there

    files.chunks(40).for_each(|x|{
        x.iter().for_each(|x|{
            let file_path=input_path.join(x);
            copy(x, file_path).expect("error in copying the data");
        });
        breakpoint();
    });
    // --- Xgboost Model
    // xgb::Xgb::runner(encoded_data);

    // ---- Deep learning Models
    // let s=encoded_data.clone().collect().unwrap();
    // let y_columns=s.return_y_columns();
    // let modelling_data=encoded_data.standard_scalar(y_columns.clone()).return_time_sequenced().collect().unwrap();
    // let control=Controller::new(modelling_data);
    // control.lstm_simulation();
}

