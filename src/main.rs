use decomposer_engine::data_engine::*; 
use ndarray::prelude::*;
use linfa_preprocessing::linear_scaling::*;
use polars::prelude::*;
use tap::Conv;
fn main() {

    let data_source=Nrel::init();
    let mut data=data_source.data;
    // let data_array=data.collect().unwrap().to_ndarray::<UInt16Type>(IndexOrder::default());
    let schema=data.collect_schema().unwrap();
    // Selecting categorical columns
    let categorical_columns: Vec<Expr>=schema.iter_names_and_dtypes().filter_map(|c| 
        {
            if c.1.is_categorical(){
                Some(col(c.0.as_str()))
            }
            else{
                None
            }
        }).collect();
    let encoded_data=data.clone().with_columns(categorical_columns.iter().map(|x| x.clone().cast(DataType::UInt32)).collect::<Vec<_>>());
    // println!("{:?}", data.select(categorical_columns).collect());
    println!("{:?}", encoded_data.select([col("in.county")]).unique(None, Default::default()).collect().unwrap().to_ndarray::<UInt32Type>(IndexOrder::C));

}
