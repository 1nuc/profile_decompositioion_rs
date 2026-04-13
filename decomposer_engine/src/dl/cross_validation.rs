use std::{fs::{File, copy, create_dir, create_dir_all}, path::{Path, PathBuf}, sync::Arc};
use burn::prelude::Module;
use burn::{config::Config, data::dataloader::{DataLoader, DataLoaderBuilder}, optim::AdamWConfig, prelude::Backend, record::CompactRecorder, tensor::backend::AutodiffBackend, train::{Learner, SupervisedTraining, metric::LossMetric}};
use polars::{frame::DataFrame, prelude::{IntoLazy, LazyFrame, col, lit}};
use rand::seq::SliceRandom;

use crate::dl::{controller::Controller, dataset::{NrelBatch, NrelBatcher, NrelDataset}, models::{bi_lstm::{NucBiLstmConfig}, hybrid_models::{Seq2SeqConfig}, lstm::{NucLstmConfig}, stacked_bi_lstm::{StackedBiLstmConfig}, stacked_lstm::{StackedLstmConfig}}};

pub struct CrossValidate{
    pub training_sets: Vec<DataFrame>,
    pub testing_sets: Vec<DataFrame>,
    pub k_fold: usize,
}
impl Default for CrossValidate{
    fn default() -> Self {
        // set the default k to 8 folds
        Self::new(8)
    }
}
impl CrossValidate{

    pub fn new(k_fold: usize)-> Self{
        let (training_sets, testing_sets)=Self::split_temporal_data(k_fold);
        Self{
            training_sets,
            testing_sets,
            k_fold,
        }
    }

    #[allow(unused_must_use)]
    pub fn split_temporal_data(k: usize)-> (Vec<DataFrame>, Vec<DataFrame>){

        let mut controller=Controller::default();
        let files=controller.train_files.clone().into_iter().take(10).collect::<Vec<PathBuf>>();
        files.into_iter().for_each(|x|{
            let input_path=Path::new("input");
            if !input_path.exists(){
                create_dir(input_path);
            }
            let file_name= Path::new(x.file_name().unwrap().to_str().unwrap()); 
            let new_file_path=input_path.join(file_name);
            if !new_file_path.exists(){
                File::create_new(new_file_path.clone());
            }
            copy(x, new_file_path).expect("unable to copy the files");
        });
        let data=controller.data_preparation(("input/*").into(), true).unwrap();
        // months format from 0 to 12
        Self::cross_valid(data.lazy(), k)
    }

    pub fn cross_valid(data: LazyFrame, k:usize) -> (Vec<DataFrame>, Vec<DataFrame>){
        // generating random number for sampling the time series across the month
        // defining the range
        let mut rnd=rand::rng();

        //creating a vector that contains the list of all available months
        let mut month_vec: Vec<i8>=(1..=12).collect();
        //shuffle the vector based on that rng
        month_vec.shuffle(&mut rnd);
        // extract the months with the size of k 
        let rand_months=month_vec.into_iter().take(k + 1).collect::<Vec<i8>>();
        let mut train_sets=Vec::new();
        let mut test_sets=Vec::new();
        rand_months.into_iter().for_each(|x|{
            let test_data=data.clone().lazy().filter(col("month of the year").eq(lit(x))).collect().unwrap();
            let train_data=data.clone().lazy().filter(col("month of the year").neq(lit(x))).collect().unwrap();
            train_sets.push(train_data);
            test_sets.push(test_data);
        });
        (train_sets, test_sets)
    }
    
}
#[derive(Debug, Config)]
pub struct CrossModels{
    pub lstm: NucLstmConfig,
    pub bi_lstm: NucBiLstmConfig,
    pub stacked_lstm: StackedLstmConfig,
    pub stacked_bi_lstm:StackedBiLstmConfig,
    pub seq2seq: Seq2SeqConfig,
    pub num_epoch: usize,
    pub workers: usize,
    pub seed: u64,
    pub opt: AdamWConfig,
    pub batch_size: usize,
}
impl Default for CrossModels{
    fn default() -> Self {
        Self{
            lstm: NucLstmConfig::default(),
            bi_lstm: NucBiLstmConfig::default(),
            stacked_lstm: StackedLstmConfig::default(),
            stacked_bi_lstm: StackedBiLstmConfig::default(),
            seq2seq: Seq2SeqConfig::default(),
            num_epoch: 30,
            workers: 4,
            seed: 42,
            opt:AdamWConfig::new().with_weight_decay(1e-3),
            batch_size: 360,
        }
    }
}
impl CrossModels{

    #[allow(unused_must_use)]
    fn create_artifact_dir<B: AutodiffBackend>(
        &self,
        artifact_dir: &str,
    ) {
        let artifact_path = Path::new(artifact_dir);
        create_dir_all(artifact_path);
    }

    pub fn train<B: AutodiffBackend>(
        &self,
        train_data: DataFrame,
        test_data: DataFrame,
        artifact_dir: &str,
        device: B::Device,
    ) {
        self.create_artifact_dir::<B>(
            artifact_dir,
        );
        // define the models for training
        let lstm_model = self.lstm.init::<B>(device.clone());
        let bi_lstm_model = self.bi_lstm.init::<B>(device.clone());
        let stackedlstm_model = self.stacked_lstm.init::<B>(device.clone());
        let stackedbilstm_model = self.stacked_bi_lstm.init::<B>(device.clone());
        let seq2seq_model = self.seq2seq.init::<B>(device.clone());

        let (train_loader, test_loader) = self.prepare_training::<B>(train_data, test_data, &device);
        // Initiate the training
        // train lstm
        let result_lstm=||{
             SupervisedTraining::new(artifact_dir, train_loader.clone(), test_loader.clone())
                .metric_train_numeric(LossMetric::new())
                .metric_valid_numeric(LossMetric::new())
                .with_file_checkpointer(CompactRecorder::new())
                .num_epochs(self.num_epoch)
                .summary()
                .launch(Learner::new(lstm_model, self.opt.init(), 1e-2))
         };

        //train_bilstm
        let result_bilstm=||{
             SupervisedTraining::new(artifact_dir, train_loader.clone(), test_loader.clone())
                .metric_train_numeric(LossMetric::new())
                .metric_valid_numeric(LossMetric::new())
                .with_file_checkpointer(CompactRecorder::new())
                .num_epochs(self.num_epoch)
                .summary()
                .launch(Learner::new(bi_lstm_model, self.opt.init(), 1e-2))
         };
        // stacked lstm
        let result_stackedlstm=||{
             SupervisedTraining::new(artifact_dir, train_loader.clone(), test_loader.clone())
                .metric_train_numeric(LossMetric::new())
                .metric_valid_numeric(LossMetric::new())
                .with_file_checkpointer(CompactRecorder::new())
                .num_epochs(self.num_epoch)
                .summary()
                .launch(Learner::new(stackedlstm_model, self.opt.init(), 1e-2))
         };
        // stacked bilstm
        let result_stackedbilstm=||{
             SupervisedTraining::new(artifact_dir, train_loader.clone(), test_loader.clone())
                .metric_train_numeric(LossMetric::new())
                .metric_valid_numeric(LossMetric::new())
                .with_file_checkpointer(CompactRecorder::new())
                .num_epochs(self.num_epoch)
                .summary()
                .launch(Learner::new(stackedbilstm_model, self.opt.init(), 1e-2))
         };
        // seq2seq model
        let result_seq2seq=||{
             SupervisedTraining::new(artifact_dir, train_loader.clone(), test_loader.clone())
                .metric_train_numeric(LossMetric::new())
                .metric_valid_numeric(LossMetric::new())
                .with_file_checkpointer(CompactRecorder::new())
                .num_epochs(self.num_epoch)
                .summary()
                .launch(Learner::new(seq2seq_model, self.opt.init(), 1e-2))
         };
        //TODO: save the configurations
        self.save(format!("{artifact_dir}/config.json").as_str())
            .unwrap();
        //TODO: Save the model results
        result_lstm().model
            .save_file(format!("{artifact_dir}/lstm_model"), &CompactRecorder::new())
            .expect("Error in saving the trained model");
        result_bilstm().model
            .save_file(format!("{artifact_dir}/bilstm_model"), &CompactRecorder::new())
            .expect("Error in saving the trained model");
        result_stackedlstm().model
            .save_file(format!("{artifact_dir}/stacked_lstm_model"), &CompactRecorder::new())
            .expect("Error in saving the trained model");
        result_stackedbilstm().model
            .save_file(format!("{artifact_dir}/stacked_bi_lstm_model"), &CompactRecorder::new())
            .expect("Error in saving the trained model");
        result_seq2seq().model
            .save_file(format!("{artifact_dir}/seq2seq_model"), &CompactRecorder::new())
            .expect("Error in saving the trained model");
    }

    fn prepare_training<B: AutodiffBackend>(
        &self,
        train_data: DataFrame,
        test_data: DataFrame,
        device: &B::Device,
    ) -> (
        Arc<dyn DataLoader<B, NrelBatch<B>>>,
        Arc<dyn DataLoader<B::InnerBackend, NrelBatch<B::InnerBackend>>>,
    ) {
        //TODO: split the data into train and validate
        let train_data = NrelDataset::new(train_data);
        let test_data = NrelDataset::new(test_data);
        let batcher = NrelBatcher::new(device.clone());
        let test_batcher = NrelBatcher::<B::InnerBackend>::new(device.clone());
        // Train Data
        let train_loader = DataLoaderBuilder::new(batcher)
            .batch_size(self.batch_size)
            .num_workers(self.workers)
            .shuffle(self.seed)
            .build(train_data);
        //Test Data
        let test_loader = DataLoaderBuilder::new(test_batcher)
            .batch_size(self.batch_size)
            .num_workers(self.workers)
            .shuffle(self.seed)
            .build(test_data);
        (train_loader, test_loader)
    }
}
