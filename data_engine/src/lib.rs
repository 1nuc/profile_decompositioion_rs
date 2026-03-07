use polars::prelude::LazyFrame;


mod polars_engine;
mod preprocessor_engine;

trait Actions{

    fn rename(&self, d: LazyFrame);

    fn create_timporal_features(&self,d: LazyFrame);

    fn process_meta_data_variants(&self, d: LazyFrame);

    fn feature_selection(&self, d: LazyFrame);
}
