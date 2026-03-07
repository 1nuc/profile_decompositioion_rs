use decomposer_engine::data_engine::*; 
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
            if c.1.is_string(){
                Some(col(c.0.as_str()))
            }
            else{
                None
            }
        }).collect();
    // println!("{:?}", categorical_columns);
    println!("{:?}", data.select(categorical_columns).collect());

}
