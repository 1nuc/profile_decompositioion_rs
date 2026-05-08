use std::{
    fs::{File, remove_file},
    io::BufWriter,
    path::Path,
};

use crate::{
    EagerActions,
    dl::{
        dataset::{NrelBatcher, NrelDataset, NrelDatasetItem},
        models::hybrid_models::Seq2SeqRecord,
        training::NrelConfig,
    },
};
use burn::{
    Tensor,
    config::Config,
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    module::Module,
    nn::loss::MseLoss,
    prelude::{Backend, ToElement},
    record::{CompactRecorder, Recorder}, tensor::ElementConversion,
};
use tracing::{info, warn};
use polars::{frame::DataFrame, prelude::*};

//TODO: a function to forward the data based on the number or the id of the building
pub struct Inference {}

impl Inference {
    #[allow(unused_must_use)]
    pub fn inference<B: Backend>(
        artifact_dir: &str,
        test_data: DataFrame,
        device: B::Device,
        timestamp: Column,
    ) -> DataFrame {
        //Load the configurations of the model
        let config = NrelConfig::load(format!("{artifact_dir}/config.json"))
            .expect("unable to find the file");

        // using compact recorder, load the last saved state of the model
        let record: Seq2SeqRecord<B> = CompactRecorder::new()
            .load(format!("{artifact_dir}/model").into(), &device)
            .expect("training model should exist first");

        // load and initialize the model for test
        let model = config.model.init::<B>(device.clone()).load_record(record);

        //load the test data and the batcher and initialize the data items
        let test_data_cloned = test_data.clone();
        let cols = test_data_cloned.return_y_columns(); // getting the columns for prediction
        // manipulation later on
        let test_data = NrelDataset::new(test_data);
        let batcher: NrelBatcher<B> = NrelBatcher::new(device.clone());

        let batched_data: Vec<NrelDatasetItem> = test_data.iter().collect();

        // convert the vec data into batches and start taking the inference

        let batch = batcher.batch(batched_data, &device);

        // get the predicted and target values
        let predicted = model.forward(batch.sequence);
        let targets = batch.target;

        let length = test_data_cloned.height();
        let df = Self::process_data::<B>(predicted.clone(), length, cols, timestamp.clone());
        Self::write_to_json(df.clone().transform_col_names(), "data.json");
        match Self::statisitcs(predicted, targets){
            Ok(_)=> info!("Predictions Submitted"),
            Err(_)=> warn!("Error sending predictions"),
        }
        df
    }

    pub fn process_data<B: Backend>(
        tensor_data: Tensor<B, 3>,
        length: usize,
        cols: Vec<&str>,
        timestamp_col: Column,
    ) -> DataFrame {
        let columns = tensor_data
            .clone()
            .iter_dim(2)
            .zip(cols)
            .map(|(tensor, col)| {
                let values = tensor
                    .flatten::<2>(1, 2)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                Column::new(col.into(), values)
            })
            .collect::<Vec<Column>>();
        DataFrame::new(length * 96, columns)
            .unwrap()
            .lazy().with_columns([polars::lazy::dsl::sum_horizontal([col("*")], true).unwrap().alias("Total")])
            .collect().unwrap()
            .hstack_mut(&[timestamp_col])
            .expect("error stacking the timestamp column").clone()
    }
    #[allow(unused_must_use)]
    pub fn write_to_json(mut df: DataFrame, file_path: &str) -> PolarsResult<()> {
        let output_path=Path::new(file_path);
        if output_path.exists() {
            remove_file(output_path);
        }
        let file = File::create(output_path).expect("unable to write to the file");
        let writer = BufWriter::new(file);
        JsonWriter::new(writer)
            .with_json_format(JsonFormat::Json)
            .finish(&mut df)
    }
    pub fn statisitcs<B: Backend, const D: usize>(predicted: Tensor<B, D>, targets: Tensor<B, D>)-> PolarsResult<()>{
        let loss = MseLoss::new();
        let mse= loss.forward(
            predicted.clone(),
            targets.clone(),
            burn::nn::loss::Reduction::Mean,
        ).to_data().to_vec::<f32>().unwrap();
        // print some statisitc
        // display the difference between targets and predicted values
        let r2_score = Self::r2_score(predicted.clone(), targets.clone()).to_data().to_vec::<f32>().unwrap();
        let mae=Self::mae(predicted.clone(), targets.clone()).to_data().to_vec::<f32>().unwrap();
        let rmse=Self::rmse(predicted.clone(), targets.clone()).to_data().to_vec::<f32>().unwrap();
        let mappe=Self::mappe(predicted, targets);
        let metrics=df!(
            "MSE"=>mse,
            "R2_score"=>r2_score,
            "MAE" =>mae,
            "RMSE" => rmse,
            "MAPPE" =>[mappe] 
        ).unwrap();
        Self::write_to_json(metrics, "metrics.json")
    }

    pub fn mae<B: Backend, const D: usize>(predictions: Tensor<B, D>, targets: Tensor<B, D>) -> Tensor<B, 1>{
        (predictions - targets).abs().mean() 
    } 
    pub fn rmse<B: Backend, const D: usize>(predictions: Tensor<B, D>, targets: Tensor<B, D>) -> Tensor<B, 1>{
        predictions.sub(targets).powf_scalar(2.0).mean().sqrt()
    } 

    pub fn mappe<B: Backend, const D: usize>(predictions: Tensor<B, D>, targets: Tensor<B, D>) ->  f32{
        let e=1e-8;
        let error=(targets.clone() - predictions.clone()).abs() / (targets.abs()+e);
        error.mean().into_scalar().elem::<f32>() * 100.00
    } 
    pub fn r2_score<B: Backend, const D: usize>(preds: Tensor<B, D>, y_true: Tensor<B, D>) -> Tensor<B,1> {
        //1- Total sum of residuals / total sum of squares
        // squeeze both predicted and targets to 1d tensor
        let mean=y_true.clone().mean();
        let total_sum_residuals=(y_true.clone() - preds.clone()).powf_scalar(2.0).sum();
        let total_sum_squares=(y_true.clone() - mean.unsqueeze()).powf_scalar(2.0).sum();
        1_f32 - (total_sum_residuals / total_sum_squares)
    }
}
