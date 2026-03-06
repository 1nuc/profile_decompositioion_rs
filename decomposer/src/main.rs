use std::{default, env, fs, path::{Path, PathBuf}};
use glob::glob;
use polars::prelude::*;
fn main() {
    // let mut paths: Vec<_>=glob("../../../datasets").unwrap().filter_map(Result::ok).collect();
    let args= ScanArgsParquet{
        low_memory: true,
        cache: false,
        ..Default::default()
    };
    let data=LazyFrame::scan_parquet("../../../src/input/*.parquet", args.clone()).expect("error");
    let buildings=data.clone().select([col("bldg_id")]).with_streaming(true).collect().unwrap().column("bldg_id").unwrap().to_owned();

    let meta_data=LazyFrame::scan_parquet("../../../metadata/MetaData.parquet", args.clone()
        ).expect("error reading the file").with_columns(
        [col("bldg_id").cast(DataType::Int32)]).filter(
        col("bldg_id").is_in(lit(buildings))).select([col("in.occupants").cast(DataType::Int32),
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
    let data=data.join(meta_data, [col("bldg_id")], [col("bldg_id")], Default::default()).collect().unwrap();
    println!("{:?}", data);
}
