#![recursion_limit = "256"]

use ndarray::{Array2, Array3};
use polars::prelude::*;
use xgboost::DMatrix;

pub mod data_engine;
pub mod dl;
pub mod preprocessor_engine;
pub mod xgb;

pub trait EagerActions{

    fn select_sequence(&self, cols: Vec<&str>, batches: usize)-> Array3<f32>;

    fn return_x_columns(&self)->Vec<&str>;

    fn return_y_columns(&self)->Vec<&str>;

    fn train_val_test_spli(&self)->(DataFrame,DataFrame,DataFrame);

    fn to_1d_vec(&self) -> Vec<f32>;
}

pub trait Actions {
    fn rename_cols(&self) -> Self;

    fn create_temporal_features(&self) -> Self;

    fn process_meta_data_variants(&self) -> Self;

    fn feature_selection(&self) -> Self;

    fn return_cols(&self) -> Vec<String>;

    fn categorical_cols(&mut self) -> Vec<Expr>;

    fn encode_categoricals(&mut self) -> Self;

    fn standard_scalar(&mut self, cols: Vec<&str>) -> Self;

    fn min_max_scalar(&mut self) -> Self;

    fn to_ndarry(&self) -> Array2<f32>;

    fn to_2d_vec(&self) -> Vec<Vec<f32>>;

    fn to_1d_vec(&self) -> Vec<f32>;

    fn to_matrix(&mut self, cols: Option<Vec<&str>>) -> DMatrix;

    fn return_time_sequenced(&self) -> Self;
}

trait ExpressionActions {
    fn cast_to_categorical(&self) -> Self;
}
