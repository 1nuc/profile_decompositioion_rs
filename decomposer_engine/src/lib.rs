use polars::prelude::{Expr, LazyFrame};


pub mod data_engine;
pub mod preprocessor_engine;

trait Actions{

    fn rename_cols(&self) -> LazyFrame;

    fn create_temporal_features(&self) -> LazyFrame;

    fn process_meta_data_variants(&self) -> LazyFrame;

    fn feature_selection(&self) -> LazyFrame;
}

trait ExpressionActions{

    fn cast_to_categorical(&self) -> Expr;
}
