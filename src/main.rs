use std::time;
use tap::Pipe;
use polars::{prelude::*};
fn rename(d: LazyFrame) -> LazyFrame{
     d.rename(
         ["out.electricity.cooling.energy_consumption..kwh"],
         ["out.electricity.AC.energy_consumption..kwh"], false)
}
fn create_timporal_features(d: LazyFrame) -> LazyFrame{
    d.with_columns([
        col("timestamp").dt().weekday().alias("day of the week").cast(DataType::UInt32),
        col("timestamp").dt().hour().alias("hour of the day"),
        col("timestamp").dt().day().alias("day of the month"),
        col("timestamp").dt().ordinal_day().alias("day of the year"),
        col("timestamp").dt().week().alias("week of the year"),
        col("timestamp").dt().month().alias("month of the year"),
        col("timestamp").dt().quarter().alias("quarter")]).with_columns([
        when(col("day of the week").is_in(
                lit(Series::new("Weekend".into(), &[6u32,7u32])), false)
            ).then(lit("Yes")).otherwise(lit("No")).alias("IsWeekend")
    ])
}

fn feature_selection(d: LazyFrame) -> LazyFrame{
    d.select([col(PlSmallStr::from("^out.electricity.*|^bldg*|^day*|^hour*|^week*|^month*|^time*|^quarter|^IsWeekend|^in.*|^Short|^climate_zone$"))])
}

fn main() {
    let args= ScanArgsParquet{
        low_memory: true,
        cache: false,
        ..Default::default()
    };

    let t=time::Instant::now();
    let meta_data=LazyFrame::scan_parquet("../../metadata/MetaData.parquet".into(), Default::default()
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
        col("in.household_has_tribal_persons")]).unique(None, Default::default());

    // Final Preprocessing step
    let data=LazyFrame::scan_parquet("../../src/input/*.parquet".into(),Default::default() 
        ).expect("error").join(
        meta_data, [col("bldg_id")],
        [col("bldg_id")], Default::default()
        ).pipe(rename).pipe(create_timporal_features).pipe(feature_selection).drop_nulls(None);
    println!("{:?}", data.clone().collect().unwrap().shape());
    // println!("{}", data.collect().unwrap().null_count());

}
