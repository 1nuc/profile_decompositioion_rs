use std::{fs::{File, copy, create_dir, create_dir_all}, path::{Path, PathBuf}, sync::Arc};

use burn::{data::dataloader::DataLoader, nn::{BiLstm, Lstm}, prelude::Backend, record::CompactRecorder, tensor::backend::AutodiffBackend, train::{Learner, SupervisedTraining, metric::LossMetric}};
use polars::{frame::DataFrame, prelude::{IntoLazy, LazyFrame, col, lit}, *};
use rand::seq::SliceRandom;

use crate::dl::{controller::Controller, dataset::{NrelBatch, NrelBatcher, NrelDataset}, models::{bi_lstm::NucBiLstm, hybrid_models::Seq2Seq, lstm::NucLstm, stacked_bi_lstm::StackedBilstm, stacked_lstm::Stackedlstm}};

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
pub struct CrossModels <B: Backend>{
    pub lsmt: NucLstm<B>,
    pub bi_lst: NucBiLstm<B>,
    pub stacked_lstm: Stackedlstm<B>,
    pub stacked_bi_lstm: StackedBilstm<B>,
    pub seq2seq: Seq2Seq<B>,

}
impl CrossModels<B: Backend>{

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
        let model = self.model.init::<B>(device.clone());
        let (train_loader, test_loader) = self.prepare_training::<B>(train_data, test_data, &device);
        // Initiate the training
        let train = SupervisedTraining::new(artifact_dir, train_loader, test_loader)
            .metric_train_numeric(LossMetric::new())
            .metric_valid_numeric(LossMetric::new())
            .with_file_checkpointer(CompactRecorder::new())
            .num_epochs(self.num_epoch)
            .summary();
        let result = train.launch(Learner::new(model, self.opt.init(), 1e-3));
        //TODO: save the configurations
        self.save(format!("{artifact_dir}/config.json").as_str())
            .unwrap();
        //TODO: Save the model results
        result
            .model
            .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
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
        let train_loader = DataLoadeer::new(batcher)
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
