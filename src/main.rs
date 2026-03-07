use decomposer_engine::data_engine::*; 
use linfa_preprocessing::linear_scaling::*;
use polars::prelude::{DataType, IndexOrder, PlSmallStr, SchemaNamesAndDtypes};
use tap::Conv;
fn main() {

    let data_source=Nrel::init();
    let mut data=data_source.data;
    // let data_array=data.collect().unwrap().to_ndarray::<UInt16Type>(IndexOrder::default());
    let schema=data.collect_schema().unwrap();
    // Selecting categorical columns
    let categorical_columns=schema.iter_names_and_dtypes().filter_map(|c| 
        {
            if c.1.is_string(){
                Some(c.0)
            }
            else{
                None
            }
        }).collect::<Vec<_>>();
    println!("{:?}", categorical_columns);
}
