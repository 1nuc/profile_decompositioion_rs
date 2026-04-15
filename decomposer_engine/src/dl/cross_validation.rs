use std::{fs::{File, copy, create_dir, create_dir_all, remove_file}, io::BufWriter, path::{Path, PathBuf}, sync::Arc};
use burn::{Tensor, backend::{Autodiff}, module::Module };
use burn::{config::Config, data::dataloader::{DataLoader, DataLoaderBuilder}, optim::AdamWConfig, prelude::Backend, record::CompactRecorder, tensor::backend::AutodiffBackend, train::{Learner, SupervisedTraining, metric::LossMetric}};
use burn_ndarray::{NdArray, NdArrayDevice};
use polars::{df, error::PolarsResult, frame::DataFrame, prelude::{IntoLazy, JsonFormat, JsonWriter, LazyFrame, UnionArgs, col, concat, lit, *}};
use rand::seq::SliceRandom;
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    nn::loss::MseLoss,
    record::Recorder
};
use crate::{Actions, EagerActions, dl::{controller::Controller, dataset::{NrelBatch, NrelBatcher, NrelDataset, NrelDatasetItem}, models::{bi_lstm::{NucBiLstmConfig, NucBiLstmRecord}, hybrid_models::{Seq2SeqConfig, Seq2SeqRecord}, lstm::{NucLstmConfig, NucLstmRecord}, stacked_bi_lstm::{StackedBiLstmConfig, StackedBilstmRecord}, stacked_lstm::{StackedLstmConfig, StackedlstmRecord}}}};

pub struct CrossValidate{
    pub training_sets: Vec<DataFrame>,
    pub testing_sets: Vec<DataFrame>,
    pub k_fold: usize,
}
impl Default for CrossValidate{
    fn default() -> Self {
        // set the default k to 8 folds
        Self::new(5)
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
        let data_cloned=data.clone().collect().unwrap();
        let columns=data_cloned.return_y_columns();
        rand_months.into_iter().for_each(|x|{
            let test_data=data.clone()
                .lazy()
                .filter(col("timestamp").dt().month().eq(lit(x)))
                .standard_scalar(columns.clone())
                .return_time_sequenced()
                .filter(col("count").eq(lit(96u32)))
                .collect()
                .unwrap();

            let train_data=data.clone()
                .lazy()
                .filter(col("timestamp").dt().month().neq(lit(x)))
                .standard_scalar(columns.clone())
                .return_time_sequenced()
                .filter(col("count").eq(lit(96u32)))
                .collect()
                .unwrap();
            train_sets.push(train_data);
            test_sets.push(test_data);
        });
        (train_sets, test_sets)
    }

    #[allow(unused_must_use, unused_assignments)]
    pub fn run(&self){
        let mut results= Vec::new();
        // edit the setup to make the page exclusive
        self.training_sets.clone()
            .into_iter().zip(self.testing_sets.clone()).for_each(|(train, test)|{
                let mut i=1;
                // define the training data
                let (train_data, val_data,_)=train.clone().train_test_split();
                // define the testing data 
                let test_data=test.clone();
                // type Mybackend= Autodiff<LibTorch>;
                // type InferBackend=LibTorch;
                // let device=LibTorchDevice::Cuda(0);
                type Mybackend=Autodiff<NdArray>;
                type InferBackend=NdArray;
                let device=NdArrayDevice::Cpu;
                self.train::<Mybackend>(train_data, val_data, device);
                let mut result=self.test::<InferBackend>(test_data, device);
                let iteration=Series::new("iteration".into(), [i]);
                let data=result.with_column(iteration.into()).unwrap();
                results.push(data.clone().lazy());
                i+=1;
            });
        //concat the vector of dataframes to one dataframe
        let concated_data=concat(results, UnionArgs::default()).unwrap().collect().unwrap();
        println!("{:?}", concated_data.clone());
        self.write_to_json(concated_data);
    }

    #[allow(unused_must_use)]
    pub fn write_to_json(&self,mut df: DataFrame) -> PolarsResult<()> {
        let output_path = Path::new("cross_validation.json");
        if output_path.exists(){
            remove_file(output_path);
        }
        let file = File::create(output_path).expect("unable to write to the file");
        let writer = BufWriter::new(file);
        JsonWriter::new(writer)
            .with_json_format(JsonFormat::Json)
            .finish(&mut df)
    }

    pub fn train<B: AutodiffBackend>(&self,train: DataFrame,val: DataFrame, device: B::Device){
        let cross_models=CrossModels::default();
        cross_models.train::<B>(train, val, "cross_validation", device);
    }
 
    pub fn test<B: Backend>(&self,data: DataFrame, device: B::Device)-> DataFrame{
        let cross_models=CrossModels::default();
        cross_models.inference::<B>("cross_validation", data, device)
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
    fn create_artifact_dir(
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
        self.create_artifact_dir(
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

    #[allow(clippy::complexity)]
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
    )-> DataFrame{
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
        let test_data = NrelDataset::new(test_data);
        let batcher: NrelBatcher<B> = NrelBatcher::new(device.clone());

        let batched_data: Vec<NrelDatasetItem>=test_data.iter().collect();

        // convert the vec data into batches and start taking the inference

        let batch = batcher.batch(batched_data, &device);

        // get the predicted and target values
        //lstm
        let lstm_predicted = lstm.forward(batch.sequence.clone());
        //bilstm
        let bilstm_predicted = bilstm.forward(batch.sequence.clone());
        //stacked lstm
        let stackedlstm_predicted = stacked_lstm.forward(batch.sequence.clone());
        // stacked_bi_lstm
        let stacked_bilstm_predicted = stacked_bi_lstm.forward(batch.sequence.clone());
        // seq2seq
        let seq2seq_predicted = seq2seq.forward(batch.sequence);
    
        // main test values
        let targets = batch.target;

        // measurements
        // lstm mse
        let lstm_mse=self.mse(lstm_predicted.clone(), targets.clone()).to_data().to_vec::<f32>().unwrap().pop().unwrap();
        // bilstm mse
        let bilstm_mse=self.mse(bilstm_predicted.clone(), targets.clone()).to_data().to_vec::<f32>().unwrap().pop().unwrap();
        // stackedlstm mse
        let stacked_lstm_mse=self.mse(stackedlstm_predicted.clone(), targets.clone()).to_data().to_vec::<f32>().unwrap().pop().unwrap();
        // stackedbilstm mse
        let stacked_bi_lstm_mse=self.mse(stacked_bilstm_predicted.clone(), targets.clone()).to_data().to_vec::<f32>().unwrap().pop().unwrap();
        // seq2seq mse
        let seq2seq_mse=self.mse(seq2seq_predicted.clone(), targets.clone()).to_data().to_vec::<f32>().unwrap().pop().unwrap();
        //----------------------
        // lstm
        let lstm_r2 = self.r2_score(lstm_predicted, targets.clone());
        // bilstm
        let bilstm_r2 = self.r2_score(bilstm_predicted, targets.clone());
        // stackedlstm
        let stacked_lstm_r2 = self.r2_score(stackedlstm_predicted, targets.clone());
        // stackedbilstm
        let stacked_bilstm_r2 = self.r2_score(stacked_bilstm_predicted, targets.clone());
        // sequence to sequence
        let seq2seq_r2 = self.r2_score(seq2seq_predicted, targets.clone());
         
        df!(
            "LSTM Model"=> [lstm_mse, lstm_r2],
            "BI Lstm Model"=> [bilstm_mse, bilstm_r2],
            "Stacked Lstm Model" => [stacked_lstm_mse, stacked_lstm_r2],
            "Stacked BiLstm Model" => [stacked_bi_lstm_mse, stacked_bilstm_r2],
            " Sequence To Sequence Model"=> [seq2seq_mse, seq2seq_r2],
            " Metrics"=> ["MSE", "R2"],
        ).unwrap()
    }
    
    pub fn mse<B: Backend>(&self,data: Tensor<B, 3>, targets: Tensor<B,3>) -> Tensor<B, 1>{
        let loss = MseLoss::new();
        loss.forward(
            data.clone(),
            targets.clone(),
            burn::nn::loss::Reduction::Mean,
        )
    }

    pub fn r2_score<B: Backend>(&self,predicted: Tensor<B, 3>, targets: Tensor<B,3>) -> f32 {
        let preds = predicted
            .flatten::<2>(1, 2)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let y_true = targets
            .flatten::<2>(1, 2)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        // display the difference between targets and predicted values
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
