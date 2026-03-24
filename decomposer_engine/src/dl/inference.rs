use burn::{config::Config, data::{dataloader::batcher::Batcher, dataset::Dataset}, module::Module, nn::BatchNormConfig, prelude::Backend, record::{CompactRecorder, Recorder}};
use polars::{df, frame::DataFrame};

use crate::dl::{dataset::{NrelBatcher, NrelDataset, NrelDatasetItem}, models::lstm::NucLstmRecord, training::NrelConfig};

pub struct Inference{}

impl Inference{
    pub fn inference<B: Backend>(artifact_dir: &str, validation_data: DataFrame, device: B::Device){
        //Load the configurations of the model
        let config=NrelConfig::load(
            format!("{artifact_dir}/config.json")).expect("unable to find the file");

        // using compact recorder, load the last saved state of the model
        let record: NucLstmRecord<B>= CompactRecorder::new().load(
            format!("{artifact_dir}/model").into(), &device).expect("training model should exist first");

        // load and initialize the model for validation 
        let model=config.model.init::<B>(device.clone()).load_record(record);

        //load the validation data and the batcher and initialize the data items
        let validation_data=NrelDataset::new(validation_data);
        let batcher: NrelBatcher<B>=NrelBatcher::new(device.clone());

        let batched_data: Vec<NrelDatasetItem>=validation_data.iter().collect();
        
        // convert the vec data into batches and start taking the inference

        let batch=batcher.batch(batched_data, &device);

        // get the predicted and target values
        let predicted= model.forward(batch.sequence);
        let targets=batch.target;
        
        println!("{:?}", predicted);
        println!("{:?}", targets);
        // squeeze both predicted and targets to 1d tensor
        // let predicted=predicted.squeeze_dims::<1>(&[0,1]).into_data();
        // let targets=targets.squeeze_dims::<1>(&[0,1]).into_data();
        // // display the difference between targets and predicted values
        // let result=df!(
        //     "predicted"=> predicted.to_vec::<f32>().unwrap(),
        //     "actual"=> targets.to_vec::<f32>().unwrap(),
        // );
        // println!("{:?}", result);
    }
}
