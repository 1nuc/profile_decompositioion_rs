use burn::{config::Config, data::{dataloader::batcher::Batcher, dataset::Dataset}, module::Module, nn::{BatchNormConfig, loss::MseLoss}, prelude::Backend, record::{CompactRecorder, Recorder}};
use polars::{df, frame::DataFrame};

use crate::dl::{dataset::{NrelBatcher, NrelDataset, NrelDatasetItem}, models::{bi_lstm::NucBiLstmRecord, hybrid_models::Seq2SeqRecord, lstm::NucLstmRecord, stacked_bi_lstm::StackedBilstmRecord, stacked_lstm::StackedlstmRecord}, training::NrelConfig};

pub struct Inference{}

impl Inference{
    pub fn inference<B: Backend>(artifact_dir: &str, test_data: DataFrame, device: B::Device){
        //Load the configurations of the model
        let config=NrelConfig::load(
            format!("{artifact_dir}/config.json")).expect("unable to find the file");

        // using compact recorder, load the last saved state of the model
        let record: Seq2SeqRecord<B>= CompactRecorder::new().load(
            format!("{artifact_dir}/model").into(), &device).expect("training model should exist first");

        // load and initialize the model for test 
        let model=config.model.init::<B>(device.clone()).load_record(record);

        //load the test data and the batcher and initialize the data items
        let test_data=NrelDataset::new(test_data);
        let batcher: NrelBatcher<B>=NrelBatcher::new(device.clone());

        let batched_data: Vec<NrelDatasetItem>=test_data.iter().collect();
        
        // convert the vec data into batches and start taking the inference

        let batch=batcher.batch(batched_data, &device);

        // get the predicted and target values
        let predicted= model.forward(batch.sequence);
        let targets=batch.target;
        
        let loss=MseLoss::new();
        let mse_loss_3d=loss.forward(predicted.clone(), targets.clone(), burn::nn::loss::Reduction::Mean);
        // squeeze both predicted and targets to 1d tensor
        let predicted=predicted.flatten::<2>(1,2).into_data().to_vec::<f32>().unwrap();
        let targets=targets.flatten::<2>(1,2).into_data().to_vec::<f32>().unwrap();
        // display the difference between targets and predicted values
        let r2_score=Self::r2_score(predicted.clone(), targets.clone());
        println!("mse: {:?}", mse_loss_3d.to_data().to_vec::<f32>());
        println!("r2: {:?}", r2_score);
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
