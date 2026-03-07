use polars::prelude::*;

struct Nrel{
    data: LazyFrame,
    meta_data: LazyFrame,
}
impl Nrel{

    fn scan_files(path: PlRefPath) -> LazyFrame{
        LazyFrame::scan_parquet(path, Default::default()).expect("Error reading the file")
    }

    fn init(&self){
        let meta_data=Self::scan_files("../../../metadata/MetaData.parquet")
    }

}
impl Actions for LazyFrame{
    fn rename(&self, d: LazyFrame) -> LazyFrame{
         d.rename(
             ["out.electricity.cooling.energy_consumption..kwh"],
             ["out.electricity.AC.energy_consumption..kwh"], false)
    }
    fn create_timporal_features(&self,d: LazyFrame) -> LazyFrame{
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
    fn process_meta_data_variants(&self, d: LazyFrame) -> LazyFrame{
        d.with_columns(
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
                col("in.household_has_tribal_persons")])
    }

    fn feature_selection(&self, d: LazyFrame) -> LazyFrame{
        d.select([col(PlSmallStr::from("^out.electricity.*|^bldg*|^day*|^hour*|^week*|^month*|^time*|^quarter|^IsWeekend|^in.*|^Short|^climate_zone$"))])
    }
}
