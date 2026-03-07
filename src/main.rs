use std::time;
use tap::Pipe;
use polars::{prelude::*};
fn main() {
    // let mut paths: Vec<_>=glob("../../../datasets").unwrap().filter_map(Result::ok).collect();
    let args= ScanArgsParquet{
        low_memory: true,
        cache: false,
        ..Default::default()
    };
    let t=time::Instant::now();
    let meta_data=LazyFrame::scan_parquet("../../metadata/MetaData.parquet", Default::default()
        ).expect("error reading the file").with_columns(
        [col("bldg_id").cast(DataType::Int32)]).
        select([col("in.occupants").cast(DataType::Int32),
        col("in.state"), col("in.county"), 
        col("in.representative_income"),
        col("in.area_median_income"),
        col("in.income"),
        col("in.income_recs_2020"),
        col("in.income_recs_2015"),
        col("in.building_america_climate_zone"),
        col("in.ashrae_iecc_climate_zone_2004_sub_cz_split").alias("cliamte_zone"),
        col("in.bedrooms").cast(DataType::Int32),
        col("in.tenure"),
        col("bldg_id").cast(DataType::UInt32),
        col("in.household_has_tribal_persons")]);

    let data=LazyFrame::scan_parquet("../../src/input/*.parquet",Default::default() 
        ).expect("error").join(
        meta_data, [col("bldg_id")],
        [col("bldg_id")], Default::default()
        ).drop_nulls(None);
    println!("first time: {:?}", t.elapsed());
}
