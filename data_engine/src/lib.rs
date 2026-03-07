use polars::prelude::LazyFrame;


mod polars_engine;
mod preprocessor_engine;

trait Actions{

    fn rename(&self) -> LazyFrame;

    fn create_temporal_features(&self) -> LazyFrame;

    fn process_meta_data_variants(&self) -> LazyFrame;

    fn feature_selection(&self) -> LazyFrame;
}
