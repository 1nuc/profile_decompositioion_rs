use ndarray::Array2;
use polars::prelude::*;


pub mod data_engine;
pub mod preprocessor_engine;

pub trait Actions{

    fn rename_cols(&self) -> Self;

    fn create_temporal_features(&self)-> Self;

    fn process_meta_data_variants(&self)-> Self;

    fn feature_selection(&self)-> Self;

    fn categorical_cols(&mut self)-> Vec<Expr>;
    
    fn encode_categoricals(&mut self)-> Self;
    
    fn standard_scalar(&mut self)-> Self;

    fn min_max_scalar(&mut self)-> Self;

    fn to_ndarry(&self) -> Array2<f32>;

    fn to_vec(&self) -> Vec<Vec<f32>>;
}

trait ExpressionActions{

    fn cast_to_categorical(&self) -> Self;
}

