use burn::{backend::{Autodiff, Wgpu}, config::Config, data::{dataloader::{DataLoaderBuilder, batcher::{self, Batcher}}, dataset::Dataset}, module::Module, nn::{Linear, LinearConfig, Lstm, LstmConfig}, optim::AdamWConfig, prelude::Backend, tensor::backend::AutodiffBackend, train::{Learner, SupervisedTraining}, *};
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

pub struct NrelDatasetItem{
    pub sequence_item: Array2<f32>,
    pub target_item: Array2<f32>,
}
pub struct NrelDataset{
    pub sequence: Array3<f32>,
    pub target: Array3<f32>,
}
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

pub struct NrelBatch<B: Backend>{
    pub sequence: Tensor<B, 3>,
    pub target: Tensor<B, 3>,
}
impl <B: Backend> Batcher<B, NrelDatasetItem, NrelBatch<B>> for NrelBatcher<B>{
    fn batch(&self, items: Vec<NrelDatasetItem>, device: &<B as Backend>::Device) -> NrelBatch<B> {
       todo!() 
    }
}

#[derive(Config, Debug)]
pub struct NucLstmConfig{
    input_size: usize,
    output_size: usize,
    hidden_size: usize,
    num_layers: usize,
    dropout: f32,
}
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

#[derive(Module, Debug)]
pub struct NucLstm<B :Backend>{
    model: Lstm<B>,
    output_model: Linear<B>,
}

impl <B: Backend>NucLstm<B> {
    pub fn forward(&self, input: Tensor<B,3>) -> Tensor<B, 2>{
        let (output,_) =self.model.forward(input, None);
        let [batch_size, seq_length, hidden_size]=output.dims();
        let last_output=output.narrow(1, seq_length-1, 1).reshape([batch_size, hidden_size]);
        self.output_model.forward(last_output)
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
