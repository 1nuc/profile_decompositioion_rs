use std::default;

use burn::{backend::{Autodiff, Wgpu, wgpu::{self, WgpuDevice}}, config::Config, data::{dataloader::{DataLoaderBuilder, batcher::{self, Batcher}}, dataset::Dataset}, module::Module, nn::{Linear, LinearConfig, Lstm, LstmConfig, loss::MseLoss}, optim::AdamWConfig, prelude::Backend, tensor::{TensorData, backend::AutodiffBackend}, train::{InferenceStep, ItemLazy, Learner, RegressionOutput, SupervisedTraining, TrainOutput, TrainStep, metric::{Adaptor, LossInput}}, *};
use ndarray::{Array2, Array3};
use polars::prelude::*;
use crate::{Actions, EagerActions};
use ndarray::s;

//TODO: Create Nrel Dataset struct and perform the get and len operation (implement Dataset trait)
//TODO: Create the item struct that will be the output of the get method and the input for the batch
//TODO: Create the batcher struct which will be the x and y 
//TODO: the batcher will be a vector of the item struct 
//TODO: Use config values for the preprocessor to separate x and y in this code
//TODO: Make separate methods to split the data to first traint and test then here in this code
//TODO: split them manually to x and y by calling the functions

const Y_COLS: usize=24;
const X_COLS: usize=24;
// const X_COLS=
// The items srtuct for which is the batcher is building
#[derive(Clone, Debug)]
pub struct NrelDatasetItem{
    pub sequence_item: Array2<f32>,
    pub target_item: Array2<f32>,
}

// The main dataset used
pub struct NrelDataset{
    pub sequence: Array3<f32>,
    pub target: Array3<f32>,
}

//Initialize the dataset and set up sequence and target
impl NrelDataset{
    pub fn new(dataset: LazyFrame) -> Self{
        let data=dataset.return_time_sequenced().collect().unwrap();
        let x_cols=data.return_x_columns();
        let y_cols=data.return_y_columns();
        let batches=data.height();
        Self{
            sequence: data 
                .clone()
                .select_sequence(x_cols.clone(), batches),
            target: data 
                .clone()
                .select_sequence(y_cols.clone(), batches),
        }
    }
}

//Specify the get method needed for batch to catch the elements
impl Dataset<NrelDatasetItem> for NrelDataset{
    fn get(&self, index: usize) -> Option<NrelDatasetItem> {
        Some(NrelDatasetItem{
           sequence_item: self.sequence
               .slice(s![index,..,..]).to_owned(),
           target_item: self.target
               .slice(s![index,..,..]).to_owned(),
        })
    }

    fn len(&self) -> usize {
        self.sequence.len()
    }

}

// Prepare the batcher
#[derive(Debug, Clone)]
pub struct NrelBatcher<B: Backend>{
    pub device: B::Device,
}
impl <B: Backend> NrelBatcher<B>{
    pub fn new(device: B::Device)-> Self{
        Self{
            device
        }
    }
}

// The output of elements batching
#[derive(Debug, Clone)]
pub struct NrelBatch<B: Backend>{
    pub sequence: Tensor<B, 3>,
    pub target: Tensor<B, 3>,
}

//Batching elements
#[allow(unused_variables)]
impl <B: Backend> Batcher<B, NrelDatasetItem, NrelBatch<B>> for NrelBatcher<B>{
    fn batch(&self, items: Vec<NrelDatasetItem>, device: &<B as Backend>::Device) -> NrelBatch<B> {
        let mut sequences=Vec::new();
        let mut targets=Vec::new();
        let batch_len=items.len();
        items.iter().clone().map(|x|{
            let tensor_sequence=Tensor::<B,2>::from_data(
                TensorData::new(
                    x.sequence_item.clone().into_raw_vec_and_offset().0, 
                    [96, X_COLS]),
            device);

            let tensor_target=Tensor::<B,2>::from_data(
                TensorData::new(
                    x.target_item.clone().into_raw_vec_and_offset().0, 
                    [96, Y_COLS]),
            device);
            sequences.push(tensor_sequence);
            targets.push(tensor_target);
        });
        let sequence=Tensor::stack(sequences, 0);
        let target=Tensor::stack(targets, 0);
        NrelBatch{
            sequence,
            target,
        }
    }
}

//Prepare the configurations of the model
#[derive(Config, Debug)]
pub struct NucLstmConfig{
    input_size: usize,
    output_size: usize,
    hidden_size: usize,
    // num_layers: usize,
    dropout: f32,
}
//Implementing default for NucLstmConfig
impl Default for NucLstmConfig{
    fn default() -> Self {
        Self{
            input_size: X_COLS,
            output_size: Y_COLS,
            hidden_size: 18,
            dropout: 0.3, //weight decay to prevent overfitting
        }
    }
}
//Initializing the model configurations 
impl NucLstmConfig{
    pub fn init<B: Backend>(&self, device: B::Device) -> NucLstm<B>{
        let lstm=LstmConfig::new(self.input_size, self.hidden_size, true).with_batch_first(true);
        let linear=LinearConfig::new(self.hidden_size, self.output_size);
        NucLstm{
           model: lstm.init(&device),
           output_model: linear.init(&device)
        }
    }
}
//TODO: Prepare the output type to be a sequence
pub struct NrelSequenceOutput<B: Backend>{
    loss: Tensor<B, 1>,
    output: Tensor<B, 3>,
    targets: Tensor<B, 3>,
}

//Apply the adoptor so the loss is calculated accordingly
impl <B: Backend>Adaptor<LossInput<B>>for NrelSequenceOutput<B>{
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}
//implement sync for the implement to be used in the train step.
impl <B: Backend> ItemLazy for NrelSequenceOutput<B>{

    type ItemSync = NrelSequenceOutput<B>; 
    fn sync(self) -> Self::ItemSync{
        Self{
            loss: self.loss.clone(),
            output: self.output.clone(),
            targets: self.targets.clone(),
        }
    }
}

//Model
#[derive(Module, Debug)]
pub struct NucLstm<B :Backend>{
    model: Lstm<B>,
    output_model: Linear<B>,
}

impl <B: Backend>NucLstm<B> {
    //the forward function for which the weights neurons are multiplied
    pub fn forward(&self, input: Tensor<B,3>) -> Tensor<B, 3>{
        let (output,_) =self.model.forward(input, None);
        let [batch_size, seq_length, hidden_size]=output.dims();
        let last_output=output.narrow(2, seq_length-1, 2).reshape([batch_size, seq_length,hidden_size]);
        self.output_model.forward(last_output)
    }
    // Calculating the loss function of the forward step
    pub fn forward_step(&self, items: NrelBatch<B>) ->NrelSequenceOutput<B>{
        let targets: Tensor<B, 3>=items.target;
        let output=self.forward(items.sequence);
        let loss=MseLoss::new().forward(output.clone(), targets.clone(), nn::loss::Reduction::Mean);
        NrelSequenceOutput{
            loss,
            output,
            targets,
        }
    }
}

//Implementing the training step for the model to obtain the gradients (weights after optimization)
impl <B: AutodiffBackend>TrainStep for NucLstm<B>{
    type Output= NrelSequenceOutput<B>;
    type Input=NrelBatch<B>;
    fn step(&self, item: Self::Input) -> TrainOutput<Self::Output> {
        let item=self.forward_step(item);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}
// Prepare the Inference step to redo the process after calculating the gradients
impl <B: Backend> InferenceStep for NucLstm<B>{
    type Input = NrelBatch<B>;
    type Output= NrelSequenceOutput<B>;
    fn step(&self, item: Self::Input) -> Self::Output {
        self.forward_step(item)
    }
}

#[derive(Debug, Config)]
pub struct NrelConfig{
        #[config(default=50)]
        pub num_epoch: usize,
        #[config(default=4)]
        pub workers: usize,
        #[config(default=42)]
        pub seed: u64,
        pub opt: AdamWConfig,
        #[config(default=32)]
        pub batch_size: usize,
}
impl NrelConfig{
    fn train(&self, train_data: LazyFrame, test_data: LazyFrame){
        //TODO: split the data into train and validate
        let train_data=NrelDataset::new(train_data);
        let test_data=NrelDataset::new(test_data);
        //TODO: Set up the backend
        type Mybackend=Autodiff<Wgpu>;
        let device=WgpuDevice::default();
        //TODO: prepare the data loader with the batcher
        let batcher=NrelBatcher::<Mybackend>::new(device.clone());
        // Train Data
        let train_loader=DataLoaderBuilder::new(batcher.clone())
            .batch_size(self.batch_size)
            .num_workers(self.workers)
            .shuffle(self.seed)
            .build(train_data);
        //Test Data
        let test_loader=DataLoaderBuilder::new(batcher.clone())
            .batch_size(self.batch_size)
            .num_workers(self.workers)
            .shuffle(self.seed)
            .build(test_data);
        //TODO: build the model
        let model=NucLstmConfig::default().init::<Mybackend>(device);
        //TODO: save the configurations
        //TODO: Save the model results
    }
}

// fn train(){
//     type Mybackend=Autodiff<Wgpu<f32, i32>>;
//     let data_source=Nrel::init();
//     let mut data=data_source.data;
//     let encoded_data=data.encode_categoricals();
//     let preprocessor=Preprocessor::new(encoded_data.clone(), 42, 0.3);
//     let (mut x_train, mut x_test, mut y_train, y_test)=preprocessor.split_x_y();
//     let device=Default::default();
//     let tensor_train=Tensor::<Mybackend,2>::from_data(x_train.to_ndarry().as_slice().unwrap(), &device);
//     let tensor_test=Tensor::<Mybackend,2>::from_data(x_test.to_ndarry().as_slice().unwrap(), &device);
//     let model=NucLstmConfig::new(preprocessor.x_labels_size, preprocessor.y_labels_size, 20, 2, 0.0).init::<Mybackend>(device.clone());
//     let batcher=Mybatcher::<Mybackend>::new(device.clone());
//     let train_batcher=DataLoaderBuilder::new(batcher).batch_size(32).num_workers(4).build(tensor_train);
//     let test_batcher=DataLoaderBuilder::new(batcher).batch_size(32).num_workers(4).build(tensor_test);
//     let config=SupervisedTraining::new("artifact_dir/", train_batcher, test_batcher).num_epochs(50).summary();
//     let result=config.launch(Learner::new(model, AdamWConfig::new().init(), 1e-3));
// }
