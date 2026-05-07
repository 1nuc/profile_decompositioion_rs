use burn::{
    config::Config,
    data::dataloader::{DataLoader, DataLoaderBuilder},
    module::{Module, ParamId},
    optim::{AdamW, AdamWConfig, Optimizer, record::AdaptorRecord},
    record::{BinFileRecorder, CompactRecorder, FullPrecisionSettings, Recorder},
    tensor::backend::AutodiffBackend,
    train::{Learner, SupervisedTraining, metric::LossMetric},
};
use polars::frame::DataFrame;
use std::{fmt::Debug, fs::*, path::Path, sync::Arc};
use hashbrown::HashMap;
use crate::dl::models::hybrid_models::Seq2Seq;
#[allow(unused_imports)]
use crate::dl::{
    dataset::{NrelBatch, NrelBatcher, NrelDataset},
    models::{
        bi_lstm::NucBiLstmConfig,
        hybrid_models::{Seq2SeqConfig, Seq2SeqRecord},
        lstm::NucLstmConfig,
        stacked_bi_lstm::StackedBiLstmConfig,
        stacked_lstm::StackedLstmConfig,
    },
};

#[derive(Debug, Config)]
pub struct NrelConfig {
    // pub model: NucLstmConfig,
    // pub model: NucBiLstmConfig,
    // pub model: StackedLstmConfig,
    // pub model: StackedBiLstmConfig,
    pub model: Seq2SeqConfig,
    #[config(default = 30)]
    pub num_epoch: usize,
    #[config(default = 4)]
    pub workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    pub opt: AdamWConfig,
    #[config(default = 250)]
    pub batch_size: usize,
}
impl NrelConfig {

    #[allow(unused_must_use)]
    pub fn trainer<B: AutodiffBackend>(
        &self,
        train_data: DataFrame,
        test_data: DataFrame,
        artifact_dir: &str,
        device: B::Device,
    ) {
        let artifact_path = Path::new(artifact_dir);
        if artifact_path.exists() {
            self.inference_learning::<B>(artifact_dir, train_data, test_data, device);
        } else {
            create_dir_all(artifact_dir);
            self.train::<B>(train_data, test_data, artifact_dir, device);
        }
    }
    pub fn train<B: AutodiffBackend>(
        &self,
        train_data: DataFrame,
        test_data: DataFrame,
        artifact_dir: &str,
        device: B::Device,
    ) {
        let model = self.model.init::<B>(device.clone());
        let (train_loader, test_loader) =
            self.prepare_training::<B>(train_data, test_data, &device);
        // Initiate the training
        println!("First Iteration is Starting");
        let train = SupervisedTraining::new(artifact_dir, train_loader, test_loader)
            .metric_train_numeric(LossMetric::new())
            .metric_valid_numeric(LossMetric::new())
            .with_file_checkpointer(CompactRecorder::new())
            .num_epochs(self.num_epoch)
            .summary();
        let opt=self.opt.init();
        let result = train.launch(Learner::new(model,  opt.clone(),1e-3));
        //TODO: save the configurations
        self.save(format!("{artifact_dir}/config.json").as_str())
            .unwrap();
        //TODO: Save the model results
        result
            .model
            .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
            .expect("Error in saving the trained model");

        BinFileRecorder::<FullPrecisionSettings>::default().record(opt.to_record(),
            format!("{artifact_dir}/optim").into())
            .expect("Unable to record the optimizer");

    }

    pub fn inference_learning<B: AutodiffBackend>(
        &self,
        artifact_dir: &str,
        train_data: DataFrame,
        test_data: DataFrame,
        device: B::Device,
    ) {
        println!("Inference Learning Iteration Begins");
        //Load the configurations of the model
        let config = NrelConfig::load(format!("{artifact_dir}/config.json"))
            .expect("unable to find the file");

        //load the optimizer to continue the training
        let adaptor: HashMap<ParamId,AdaptorRecord<AdamW,B>>=BinFileRecorder::<FullPrecisionSettings>::new()
            .load(format!("{artifact_dir}/optim").into(), &device).expect("Error loading the optimizer");

        let opt=config.opt.init::<B, Seq2Seq<B>>().load_record(adaptor);
        // load and initialize the model to continue the training
        let record: Seq2SeqRecord<B> = CompactRecorder::new()
            .load(format!("{artifact_dir}/model").into(), &device)
            .expect("First batch is not trained yet");

        let model = config.model.init::<B>(device.clone()).load_record(record);
        
        let (train_loader, test_loader) =
            self.prepare_training::<B>(train_data, test_data, &device);

        //TODO: build the model
        let train = SupervisedTraining::new(artifact_dir, train_loader, test_loader)
            .metric_train_numeric(LossMetric::new())
            .metric_valid_numeric(LossMetric::new())
            .with_file_checkpointer(CompactRecorder::new())
            .num_epochs(self.num_epoch)
            .summary();

        let result = train.launch(Learner::new(model, opt.clone(), 1e-3));
        //TODO: save the configurations
        self.save(format!("{artifact_dir}/config.json").as_str())
            .unwrap();
        //TODO: Save the model results
        result
            .model
            .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
            .expect("Error in saving the trained model");

        BinFileRecorder::<FullPrecisionSettings>::default().record(opt.to_record(),
            format!("{artifact_dir}/optim").into())
            .expect("Unable to record the optimizer");

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
        let batcher = NrelBatcher::<B>::new(device.clone());
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
