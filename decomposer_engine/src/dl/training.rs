use burn::{backend::{Autodiff, LibTorch, libtorch::LibTorchDevice}, config::Config, data::dataloader::DataLoaderBuilder, module::{AutodiffModule, Module}, optim::AdamWConfig, record::{CompactRecorder, NoStdInferenceRecorder}, tensor::backend::AutodiffBackend, train::{InferenceStep, ItemLazy, Learner, SupervisedTraining, TrainStep, metric::{Adaptor, LossInput, LossMetric}}};
use polars::{frame::DataFrame, prelude::ChunkCompareIneq};
use std::{fmt::{Debug, Display}, fs::*};

use crate::dl::dataset::{NrelBatch, NrelBatcher, NrelDataset};



#[derive(Debug, Config)]
pub struct NrelConfig{
        #[config(default=15)]
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
    fn create_artifact_dir(&self,artifact_dir: &str){
        remove_dir_all(artifact_dir);
        create_dir_all(artifact_dir);
    }
    pub fn train<T>(&self, model: T,train_data: DataFrame, test_data: DataFrame, artifact_dir: &str)
        where T: TrainStep<Input = NrelBatch<Autodiff<LibTorch>>> + InferenceStep + ItemLazy + Debug + Display + Send + AutodiffModule<Autodiff<LibTorch>> + 'static,
              <T as AutodiffModule<Autodiff<LibTorch>>>::InnerModule: InferenceStep< Input = NrelBatch<LibTorch>> + Module<LibTorch>,
              <T as TrainStep>::Output: ItemLazy,
              <<T as TrainStep>::Output as ItemLazy>::ItemSync: Adaptor<LossInput<Autodiff<LibTorch>>>,
              <<T as AutodiffModule<Autodiff<LibTorch>>>::InnerModule as InferenceStep>::Output: ItemLazy,
              <<<T as AutodiffModule<Autodiff<LibTorch>>>::InnerModule as InferenceStep>::Output as ItemLazy>::ItemSync: Adaptor<LossInput<LibTorch>>,
    {
        self.create_artifact_dir(artifact_dir);
        //TODO: split the data into train and validate
        let train_data=NrelDataset::new(train_data);
        let test_data=NrelDataset::new(test_data);
        //TODO: Set up the backend
        type Mybackend=Autodiff<LibTorch>;
        type Testbackend=LibTorch;
        let device=LibTorchDevice::Cuda(0);
        //TODO: prepare the data loader with the batcher
        let batcher=NrelBatcher::<Mybackend>::new(device);
        let test_batcher=NrelBatcher::<Testbackend>::new(device);
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
