//TODO: Define the logic for training
//TODO: Define the logic for testing and validation
//TODO: Must make a function to split the dat into train, test and split
//TODO: steps:
//1. One struct that contains the data that will be red through the data engine
//2. define the default implementation for the controller, the data backend, and many more
//3. method for training and predicting
//4. catch the method for the metrics
//5. function to recieve the building input and process or forward the output

use std::{
    fs::{File, copy, create_dir, read_dir, remove_dir_all},
    path::{Path, PathBuf},
};

use crate::{
    Actions, EagerActions,
    data_engine::Nrel,
    dl::{
        inference::Inference,
        models::{
            bi_lstm::NucBiLstmConfig, hybrid_models::Seq2SeqConfig, lstm::NucLstmConfig,
            stacked_bi_lstm::StackedBiLstmConfig, stacked_lstm::StackedLstmConfig,
        },
        training::NrelConfig,
    },
};
use burn::{
    backend::{Autodiff, Wgpu, wgpu::WgpuDevice}, optim::AdamWConfig, prelude::Backend, tensor::backend::AutodiffBackend
};
use polars::{
    frame::DataFrame,
    prelude::{Column, PlRefPath, all, any_horizontal, *},
};

pub struct Controller {
    pub train_data: DataFrame,
    pub val_data: DataFrame,
    pub production_data: DataFrame,
    pub train_files: Vec<PathBuf>,
    pub test_files: Vec<PathBuf>,
    pub timestamp: Column,
}

impl Default for Controller {
    fn default() -> Self {
        Self::new()
    }
}

impl Controller {
    pub fn new() -> Self {
        let (train_files, test_files) = Self::organize_files();
        Self {
            train_data: DataFrame::default(),
            val_data: DataFrame::default(),
            production_data: DataFrame::default(),
            train_files,
            test_files,
            timestamp: Column::default(),
        }
    }

    pub fn data_preparation(&mut self, input: PlRefPath) {
        let data_source = Nrel::init(input);
        let data = data_source.data;
        let meta_data=data_source.meta_data;
        let mut encoded_data = data.clone().encode_categoricals();
        let s = encoded_data.clone().collect().unwrap();
        let y_columns = s.return_y_columns();
        self.timestamp = s
            .column("timestamp")
            .expect("unable to find the column")
            .clone();
        let data = encoded_data
            .standard_scalar(y_columns.clone())
            .return_time_sequenced()
            .collect()
            .unwrap();
        (self.train_data, self.val_data, self.production_data) = data.train_test_split();
    }

    pub fn organize_files() -> (Vec<PathBuf>, Vec<PathBuf>) {
        let dir = read_dir("../../../datasets").unwrap();
        let files = dir.map(|x| x.unwrap().path()).collect::<Vec<PathBuf>>();
        let split_inx = (files.len() as f32 * 0.1).round() as usize;
        let (a, b) = files.split_at(split_inx);
        (b.to_vec(), a.to_vec())
    }

    //UNUSED: a simulation method for the inference
    pub fn run_inference(&mut self) {
        let artifact_dir = Path::new("lstm_artifact/");
        if artifact_dir.exists() {
            remove_dir_all("input").expect("can't find the input dir");
            remove_dir_all(artifact_dir).expect("can't find the artifact dir");
        }
        self.chunks_iteration(self.test_files.clone());
    }

    // a method to simulate the training for the models
    pub fn run_training(&mut self) {
        let artifact_dir = Path::new("lstm_artifact/");
        if artifact_dir.exists() {
            remove_dir_all("input").expect("can't find the input dir");
            remove_dir_all(artifact_dir).expect("can't find the artifact dir");
        }
        self.chunks_iteration(self.train_files.clone());
    }

    // This is the main function to send the data for the dashboard
    pub fn infer_one_building(&mut self, building: &str) -> DataFrame {
        let input_path = Path::new("production_set");
        if !input_path.exists() {
            create_dir(input_path).unwrap();
        }
        let bldg_file = format!("{building}-28.parquet").as_str().to_owned();
        // find the original file from the main data
        let dataset_path = self
            .test_files
            .iter()
            .filter(|x| {
                x.file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .contains(&bldg_file)
            })
            .collect::<PathBuf>();

        let path = Path::new(&bldg_file);
        let file_path = input_path.join(path);
        if !file_path.exists() {
            File::create_new(&file_path).expect("unable to create a file");
        }
        copy(&dataset_path, file_path.clone()).expect("error in copying the data");
        // TODO: Import the data set for the inference
        self.data_preparation(file_path.to_str().unwrap().into());
        // ---- Deep learning Models
        type Mybackend = Wgpu;
        let device = WgpuDevice::default();
        self.infer_lstm::<Mybackend>(device)
    }

    pub fn chunks_iteration(&mut self, files: Vec<PathBuf>) {
        // Join the file names first
        // then copy the content of the files there
        files.chunks(40).for_each(|x| {
            let input_path = Path::new("input");
            if !input_path.exists() {
                create_dir(input_path).unwrap();
            }
            x.iter().for_each(|x| {
                let path = Path::new(x.file_name().unwrap().to_str().unwrap());
                let file_path = input_path.join(path);
                File::create_new(&file_path).expect("unable to create a file");
                copy(x, file_path).expect("error in copying the data");
            });
            // ---- Deep learning Models
            self.data_preparation(("input/*.parquet").into());
            self.lstm_simulation();
            remove_dir_all("input").expect("can't find the input dir");
        });
    }

    pub fn one_trail_training(&mut self){
        let artifact_dir = Path::new("lstm_artifact/");
        if artifact_dir.exists() {
            remove_dir_all("input").expect("can't find the input dir");
            remove_dir_all(artifact_dir).expect("can't find the artifact dir");
        }

        let files=self.train_files.clone().into_iter().take(40).collect::<Vec<PathBuf>>();

        files.into_iter().for_each(|x| {
            let input_path = Path::new("input");
            if !input_path.exists() {
                create_dir(input_path).unwrap();
            }
            let path = Path::new(x.file_name().unwrap().to_str().unwrap());
            let file_path = input_path.join(path);
            File::create_new(&file_path).expect("unable to create a file");
            copy(x, file_path).expect("error in copying the data");
            // ---- Deep learning Models
            self.data_preparation(("input/*.parquet").into());
            self.lstm_simulation();
            remove_dir_all("input").expect("can't find the input dir");
        });
    }
    // return all the buildings available in the data
    pub fn return_nrel_buildings(&self) -> Vec<String> {
        self.test_files
            .iter()
            .map(|x| {
                x.file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .strip_suffix("-28.parquet")
                    .unwrap()
                    .to_string()
            })
            .collect::<Vec<String>>()
    }

    pub fn lstm_simulation(&self) {
        type Mybackend = Autodiff<Wgpu>;
        let device = WgpuDevice::DiscreteGpu(0);
        self.train_lstm::<Mybackend>(device.clone());
    }

    pub fn train_lstm<B: AutodiffBackend>(&self, device: B::Device) {
        let model = Seq2SeqConfig::default();
        let model_config = NrelConfig::new(model, AdamWConfig::new().with_weight_decay(1e-3));
        model_config.train::<B>(
            self.train_data.clone(),
            self.val_data.clone(),
            "lstm_artifact",
            device,
        );
    }

    pub fn infer_lstm<B: Backend>(&self, device: B::Device) -> DataFrame {
        Inference::inference::<B>(
            "lstm_artifact",
            self.production_data.clone(),
            device,
            self.timestamp.clone(),
        )
    }
}
