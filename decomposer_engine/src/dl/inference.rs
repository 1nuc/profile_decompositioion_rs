use std::{fs::File, io::BufWriter, path::Path};

use burn::{
    Tensor, config::Config, data::{dataloader::batcher::Batcher, dataset::Dataset}, module::Module, nn::{BatchNormConfig, loss::MseLoss}, prelude::Backend, record::{CompactRecorder, Recorder}
};
use polars::{frame::DataFrame, prelude::*};

use crate::{EagerActions, dl::{
    dataset::{NrelBatcher, NrelDataset, NrelDatasetItem},
    models::{
        bi_lstm::NucBiLstmRecord, hybrid_models::Seq2SeqRecord, lstm::NucLstmRecord,
        stacked_bi_lstm::StackedBilstmRecord, stacked_lstm::StackedlstmRecord,
    },
    training::NrelConfig,
}};

//TODO: a function to forward the data based on the number or the id of the building
pub struct Inference {}

impl Inference {
    #[allow(unused_must_use)]
    pub fn inference<B: Backend>(artifact_dir: &str, test_data: DataFrame, device: B::Device, timestamp: Column) {
        //Load the configurations of the model
        let config = NrelConfig::load(format!("../{artifact_dir}/config.json"))
            .expect("unable to find the file");

        // using compact recorder, load the last saved state of the model
        let record: Seq2SeqRecord<B> = CompactRecorder::new()
            .load(format!("../{artifact_dir}/model").into(), &device)
            .expect("training model should exist first");

        // load and initialize the model for test
        let model = config.model.init::<B>(device.clone()).load_record(record);

        //load the test data and the batcher and initialize the data items
        let test_data_cloned=test_data.clone();
        let cols= test_data_cloned.return_y_columns(); // getting the columns for prediction
                                                       // manipulation later on
        let test_data = NrelDataset::new(test_data);
        let batcher: NrelBatcher<B> = NrelBatcher::new(device.clone());

        let batched_data: Vec<NrelDatasetItem> = test_data.iter().collect();

        // convert the vec data into batches and start taking the inference

        let batch = batcher.batch(batched_data, &device);

        // get the predicted and target values
        let predicted = model.forward(batch.sequence);
        println!("{:?}", predicted.dims());
        let targets = batch.target;

        let length=test_data_cloned.height();
        let  df=Self::process_data::<B>(predicted.clone(),length, cols, timestamp.clone());
        println!("{:?}", df);
        Self::write_to_json(df);

        let loss = MseLoss::new();
        let mse_loss_3d = loss.forward(
            predicted.clone(),
            targets.clone(),
            burn::nn::loss::Reduction::Mean,
        );
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

    pub fn process_data<B: Backend>(tensor_data: Tensor<B, 3>, length: usize, cols: Vec<&str>, timestamp_col: Column)-> DataFrame{

        let columns=tensor_data.clone().iter_dim(2).zip(cols).map(|(tensor, col)|{
            let values=tensor.flatten::<2>(1, 2).into_data().to_vec::<f32>().unwrap();
            Column::new(col.into(), values)
        }).collect::<Vec<Column>>();
       DataFrame::new(length * 96, columns)
           .unwrap()
           .hstack_mut(&[timestamp_col])
           .expect("error stacking the timestamp column")
           .clone()
           .lazy()
           .select([col("*").implode()])
           .collect()
           .unwrap()
    }
     pub fn write_to_json(mut df: DataFrame) -> PolarsResult<()> {
        let output_path=Path::new("data.json");

        let file = if !output_path.exists(){
            File::create_new(output_path).unwrap()
            }
            else{
                File::open(output_path).unwrap()
        };
        let writer=BufWriter::new(file);
        JsonWriter::new(writer).with_json_format(JsonFormat::Json).finish(&mut df)
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
