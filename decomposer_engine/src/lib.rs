use polars::prelude::{Expr, LazyFrame};


pub mod data_engine;
pub mod preprocessor_engine;

trait Actions{

    fn rename_cols(&self) -> LazyFrame;

    fn create_temporal_features(&self) -> LazyFrame;

    fn process_meta_data_variants(&self) -> LazyFrame;

    fn feature_selection(&self) -> LazyFrame;

    fn categorical_cols(&mut self)-> Vec<Expr>;
    
    fn encode_categoricals(&mut self) -> LazyFrame;
}

trait ExpressionActions{

    fn cast_to_categorical(&self) -> Expr;
}
