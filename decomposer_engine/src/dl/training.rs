use burn::{config::Config, data::dataloader::DataLoaderBuilder, module::{Module}, optim::AdamWConfig, record::{CompactRecorder}, tensor::backend::AutodiffBackend, train::{Learner, SupervisedTraining,metric::{LossMetric}}};
use polars::{frame::DataFrame};
use std::{fmt::{Debug}, fs::*};

use crate::dl::{dataset::{NrelBatcher, NrelDataset}, models::{lstm::NucLstmConfig, stacked_lstm::StackedLstmConfig}};


#[derive(Debug, Config)]
pub struct NrelConfig{
        // pub model: NucLstmConfig,
        pub model: StackedLstmConfig,
        #[config(default=30)]
        pub num_epoch: usize,
        #[config(default=4)]
        pub workers: usize,
        #[config(default=42)]
        pub seed: u64,
        pub opt: AdamWConfig,
        #[config(default=360)]
        pub batch_size: usize,
}
impl NrelConfig{

    #[allow(unused_must_use)]
    fn create_artifact_dir(&self,artifact_dir: &str){
        remove_dir_all(artifact_dir);
        create_dir_all(artifact_dir);
    }
    pub fn train<B: AutodiffBackend>(&self, train_data: DataFrame, test_data: DataFrame, artifact_dir: &str, device: B::Device)
    {
        self.create_artifact_dir(artifact_dir);
        //TODO: split the data into train and validate
        let train_data=NrelDataset::new(train_data);
        let test_data=NrelDataset::new(test_data);
        let batcher=NrelBatcher::<B>::new(device.clone());
        let test_batcher=NrelBatcher::<B::InnerBackend>::new(device.clone());
        // Train Data
        let train_loader=DataLoaderBuilder::new(batcher)
            .batch_size(self.batch_size)
            .num_workers(self.workers)
            .shuffle(self.seed)
            .build(train_data);
        //Test Data
        let test_loader=DataLoaderBuilder::new(test_batcher)
            .batch_size(self.batch_size)
            .num_workers(self.workers)
            .shuffle(self.seed)
            .build(test_data);
        //TODO: build the model
        let model=self.model.init::<B>(device);
        let train=SupervisedTraining::new(artifact_dir, train_loader, test_loader)
            .metric_train_numeric(LossMetric::new())
            .metric_valid_numeric(LossMetric::new())
            .with_file_checkpointer(CompactRecorder::new())
            .num_epochs(self.num_epoch)
            .summary();
        let result=train.launch(Learner::new(model, self.opt.init(), 1e-3));
        //TODO: save the configurations
        self.save(format!("{artifact_dir}/config.json").as_str()).unwrap();
        //TODO: Save the model results
        result.model.save_file(format!("{artifact_dir}/model"), &CompactRecorder::new()).expect("Error in saving the trained model");
    }
}
