//TODO: Define the logic for training
//TODO: Define the logic for testing and validation
//TODO: Must make a function to split the dat into train, test and split
//TODO: steps:
//1. One struct that contains the data that will be red through the data engine
//2. define the default implementation for the controller, the data backend, and many more
//3. method for training and predicting
//4. catch the method for the metrics
//5. function to recieve the building input and process or forward the output

use std::{fmt::format, fs::{File,copy, create_dir, read_dir, remove_dir_all}, path::{Path, PathBuf}};

use burn::{backend::{Autodiff, Wgpu, wgpu::WgpuDevice}, optim::AdamWConfig, tensor::backend::AutodiffBackend};
use ndarray::Data;
use polars::frame::DataFrame;
use crate::{Actions, EagerActions, data_engine::Nrel, dl::{inference::Inference, models::{bi_lstm::NucBiLstmConfig, hybrid_models::Seq2SeqConfig, lstm::NucLstmConfig, stacked_bi_lstm::StackedBiLstmConfig, stacked_lstm::StackedLstmConfig}, training::NrelConfig}};

pub struct Controller{
    pub train_data: DataFrame,
    pub test_data: DataFrame,
    pub val_data: DataFrame,
    pub train_files: Vec<PathBuf>,
    pub test_files: Vec<PathBuf>,
}

impl Default for Controller{
    fn default() -> Self {
        Self::new()
    }
}

impl Controller{

    pub fn new() -> Self{
        let data=Self::data_preparation();
        let (train_data, val_data, test_data)=data.train_val_test_spli();
        let (train_files, test_files)=Self::organize_files();
        Self{
            train_data,
            test_data,
            val_data,
            train_files,
            test_files
        }
    }

    pub fn data_preparation() -> DataFrame{
        let data_source=Nrel::init();
        let data=data_source.data;
        let mut encoded_data=data.clone().encode_categoricals();
        let s=encoded_data.clone().collect().unwrap();
        let y_columns=s.return_y_columns();
        encoded_data.standard_scalar(y_columns.clone()).return_time_sequenced().collect().unwrap()
    }

    pub fn organize_files() -> (Vec<PathBuf>, Vec<PathBuf>){
        let dir=read_dir("../../datasets/").unwrap();
        let files=dir.map(|x| x.unwrap().path()
            ).collect::<Vec<PathBuf>>();
        let split_inx= (files.len() as f32 * 0.1).round() as usize;
        let (a, b)=files.split_at(split_inx);
        (b.to_vec(), a.to_vec())
    }

    pub fn run_inference(&self){
        let artifact_dir=Path::new("lstm_artifact/");
        if artifact_dir.exists(){
            remove_dir_all("input").expect("can't find the input dir");
            remove_dir_all(artifact_dir).expect("can't find the artifact dir");
        }
        self.chunks_iteration(self.test_files.clone());
    }

    pub fn run_training(&self){
        let artifact_dir=Path::new("lstm_artifact/");
        if artifact_dir.exists(){
            remove_dir_all("input").expect("can't find the input dir");
            remove_dir_all(artifact_dir).expect("can't find the artifact dir");
        }
        self.chunks_iteration(self.train_files.clone());
    }

    // This is the main function to send the data for the dashboard 
    pub fn infer_one_building(&self, building: &str){

        let input_path=Path::new("input");
        if !input_path.exists(){
            let input_lib=create_dir(input_path).unwrap();
        }
        let bldg_file=format!("{building}-28.parquet").as_str().to_owned();
        // find the original file from the main data
        let dataset_path=self.test_files.iter().filter(
            |x| x.to_str().unwrap().contains(&bldg_file)
            ).collect::<PathBuf>();

        let path=Path::new(&bldg_file);
        let file_path=input_path.join(path);
        File::create_new(&file_path).expect("unable to create a file");
        copy(&dataset_path, file_path).expect("error in copying the data");
        // ---- Deep learning Models
        type Mybackend= Autodiff<Wgpu>;
        let device=WgpuDevice::DiscreteGpu(0);
        self.infer_lstm::<Mybackend>(device);

        remove_dir_all("input").expect("can't find the input dir");
    }

    pub fn chunks_iteration(&self,files: Vec<PathBuf>){

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
            self.lstm_simulation();
            remove_dir_all("input").expect("can't find the input dir");
        });
    }

    pub fn lstm_simulation(&self){
        type Mybackend= Autodiff<Wgpu>;
        let device=WgpuDevice::DiscreteGpu(0);
        // self.train_lstm::<Mybackend>(device.clone());
        self.infer_lstm::<Mybackend>(device);
    }

    pub fn train_lstm<B: AutodiffBackend>(&self,device: B::Device){
        let model=Seq2SeqConfig::default();
        let model_config=NrelConfig::new(model,AdamWConfig::new().with_weight_decay(1e-3));
        model_config.train::<B>(self.train_data.clone(), self.val_data.clone(), "lstm_artifact", device);
    }

    pub fn infer_lstm<B: AutodiffBackend>(&self,device: B::Device){
        Inference::inference::<B>("lstm_artifact", self.test_data.clone(), device);
    }

}


