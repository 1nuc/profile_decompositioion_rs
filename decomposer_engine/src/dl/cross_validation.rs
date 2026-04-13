use std::{fs::{File, copy, create_dir, create_dir_all}, path::{Path, PathBuf}, sync::Arc};
use burn::{Tensor, module::Module, nn::{BiLstmRecord, LstmRecord}};
use burn::{config::Config, data::dataloader::{DataLoader, DataLoaderBuilder}, optim::AdamWConfig, prelude::Backend, record::CompactRecorder, tensor::backend::AutodiffBackend, train::{Learner, SupervisedTraining, metric::LossMetric}};
use polars::{frame::DataFrame, prelude::{IntoLazy, LazyFrame, col, lit}};
use rand::seq::SliceRandom;

use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    nn::loss::MseLoss,
    record::Recorder
};
use crate::dl::{controller::Controller, dataset::{NrelBatch, NrelBatcher, NrelDataset, NrelDatasetItem}, models::{bi_lstm::{NucBiLstmConfig, NucBiLstmRecord}, hybrid_models::{Seq2SeqConfig, Seq2SeqRecord}, lstm::{NucLstmConfig, NucLstmRecord}, stacked_bi_lstm::{StackedBiLstmConfig, StackedBilstmRecord}, stacked_lstm::{StackedLstmConfig, StackedlstmRecord}}};

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

    #[allow(unused_must_use)]
    pub fn inference<B: Backend>(
        &self,
        artifact_dir: &str,
        test_data: DataFrame,
        device: B::Device,
    ){
        //Load the configurations of the model
        let config = Self::load(format!("{artifact_dir}/config.json"))
            .expect("unable to find the file");

        // using compact recorder, load the last saved state of the model
        let lstm_record: NucLstmRecord<B>=CompactRecorder::new()
            .load(format!("{artifact_dir}/lstm_model").into(), &device)
            .expect("training model should exist first");
        let bilstm_record: NucBiLstmRecord<B>=CompactRecorder::new()
            .load(format!("{artifact_dir}/bilstm_model").into(), &device)
            .expect("training model should exist first");
        let stackedlstm_record: StackedlstmRecord<B> = CompactRecorder::new()
            .load(format!("{artifact_dir}/stacked_lstm_model").into(), &device)
            .expect("training model should exist first");
        let stacked_bilstm_record: StackedBilstmRecord<B>=CompactRecorder::new()
            .load(format!("{artifact_dir}/stacked_bi_lstm_model").into(), &device)
            .expect("training model should exist first");
        let seq2seq_record: Seq2SeqRecord<B>=CompactRecorder::new()
            .load(format!("{artifact_dir}/seq2seq_model").into(), &device)
            .expect("training model should exist first");
        // load and initialize all models for test
        let lstm = config.lstm.init::<B>(device.clone()).load_record(lstm_record);
        let bilstm = config.bi_lstm.init::<B>(device.clone()).load_record(bilstm_record);
        let stacked_lstm = config.stacked_lstm.init::<B>(device.clone()).load_record(stackedlstm_record);
        let stacked_bi_lstm = config.stacked_bi_lstm.init::<B>(device.clone()).load_record(stacked_bilstm_record);
        let seq2seq = config.seq2seq.init::<B>(device.clone()).load_record(seq2seq_record);

        //load the test data and the batcher and initialize the data items
        let test_data_cloned = test_data.clone();
        // manipulation later on
        let test_data = NrelDataset::new(test_data);
        let batcher: NrelBatcher<B> = NrelBatcher::new(device.clone());

        let batched_data: Vec<NrelDatasetItem>=test_data.iter().collect();

        // convert the vec data into batches and start taking the inference

        let batch = batcher.batch(batched_data, &device);

        // get the predicted and target values
        //lstm
        let lstm_predicted = lstm.forward(batch.sequence);
        //bilstm
        let bilstm_predicted = bilstm.forward(batch.sequence);
        //stacked lstm
        let stackedlstm_predicted = stacked_lstm.forward(batch.sequence);
        // stacked_bi_lstm
        let stacked_bilstm_predicted = stacked_bi_lstm.forward(batch.sequence);
        // seq2seq
        let seq2seq_predicted = seq2seq.forward(batch.sequence);
    
        // main test values
        let targets = batch.target;

        // print some statisitc
        // squeeze both predicted and targets to 1d tensor
        let predicted = predicted
            .flatten::<2>(1, 2)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let targets = targets
            .flatten::<2>(1, 2)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        // display the difference between targets and predicted values
        let r2_score = Self::r2_score(predicted.clone(), targets.clone());
        println!("mse: {:?}", mse_loss_3d.to_data().to_vec::<f32>());
        println!("r2: {:?}", r2_score);
    }
    
    pub fn mse<B: Backend>(data: Tensor<B, 3>, targets: Tensor<B,3>) -> Tensor<B, 1>{
        let loss = MseLoss::new();
        let mse_loss = loss.forward(
            data.clone(),
            targets.clone(),
            burn::nn::loss::Reduction::Mean,
        );
        mse_loss
    }

    pub fn r2_score(preds: Vec<f32>, y_true: Vec<f32>) -> f32 {
        //1- Total sum of residuals / total sum of squares
        let mean = y_true.iter().sum::<f32>() / y_true.len() as f32;
        let total_sum_residuals = y_true
            .iter()
            .zip(preds.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>();
        let total_sum_squares = y_true.iter().map(|x| (x - mean).powi(2)).sum::<f32>();

        1_f32 - (total_sum_residuals / total_sum_squares)
    }
}
