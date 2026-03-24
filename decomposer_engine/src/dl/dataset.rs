use burn::{Tensor, data::{dataloader::batcher::Batcher, dataset::Dataset}, prelude::Backend, tensor::TensorData};
use ndarray::{Array2, Array3, s};
use polars::frame::DataFrame;

use crate::EagerActions;


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
    pub fn new(data: DataFrame) -> Self{
        let mut x_cols=data.return_x_columns();
        let y_cols=data.return_y_columns();
        let batches=data.height();
        x_cols.retain(|x| !x.eq(&"timestamp") & !x.eq(&"bldg_id"));
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
        self.sequence.shape()[0]
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
        for item in items{
            let (seq_cols, seq_rows)=item.sequence_item.dim();
            let (tar_cols, tar_rows)=item.target_item.dim();
            let tensor_sequence=Tensor::<B,2>::from_data(
                TensorData::new(
                    item.sequence_item.clone().into_raw_vec_and_offset().0, 
                    [seq_cols, seq_rows]),
            device);

            let tensor_target=Tensor::<B,2>::from_data(
                TensorData::new(
                    item.target_item.clone().into_raw_vec_and_offset().0, 
                    [tar_cols, tar_rows]),
            device);
            sequences.push(tensor_sequence);
            targets.push(tensor_target);
        }
        let sequence=Tensor::stack(sequences, 0);
        let target=Tensor::stack(targets, 0);
        NrelBatch{
            sequence,
            target,
        }
    }
}
