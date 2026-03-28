use std::{env, ffi::OsString, fs::{self, File, copy, create_dir, read_dir, remove_dir_all}, path::{Path, PathBuf}};

use decomposer_engine::{Actions, EagerActions, data_engine::*, dl::controller::Controller, xgb};

fn main(){
    let dir=read_dir("../../datasets/").unwrap();
    
    let files=dir.map(|x| x.unwrap().path()
        ).collect::<Vec<PathBuf>>();


    let artifact_dir=Path::new("../lstm_artifact/");
    if artifact_dir.exists(){
        remove_dir_all(artifact_dir).expect("can't find the artifact dir");
    }
    // Join the file names first
    // then copy the content of the files there

    files.chunks(40).for_each(|x|{

        let input_path=Path::new("input");
        if !input_path.exists(){
            let input_lib=create_dir(input_path).unwrap();
        }
        x.iter().for_each(|x|{
            let path=Path::new(x.file_name().unwrap().to_str().unwrap());
            let file_path=input_path.join(path);
            File::create_new(&file_path).expect("unable to create a file");
            copy(x, file_path).expect("error in copying the data");
        });
        // ---- Deep learning Models
        let data_source=Nrel::init();
        let data=data_source.data;
        let mut encoded_data=data.clone().encode_categoricals();
        let s=encoded_data.clone().collect().unwrap();
        let y_columns=s.return_y_columns();
        let modelling_data=encoded_data.standard_scalar(y_columns.clone()).return_time_sequenced().collect().unwrap();
        let control=Controller::new(modelling_data);
        control.lstm_simulation();
        remove_dir_all("input").expect("can't find the input dir");
    });
    // --- Xgboost Model
    // xgb::Xgb::runner(encoded_data);

}

