use std::usize;

use polars::prelude::*;
use crate::{Actions, ExpressionActions};

pub struct Nrel{
    pub data: LazyFrame,
    pub meta_data: LazyFrame,
}

impl Nrel{

    fn scan_files(path: PlRefPath) -> LazyFrame{
        LazyFrame::scan_parquet(path, Default::default()).expect("Error reading the file")
    }

    pub fn init() -> Self{
        let meta_data_=Self::scan_files("../metadata/MetaData.parquet".into()
            ).process_meta_data_variants().unique(None, Default::default());
        let data_=Self::scan_files("../src/input/*.parquet".into()
            ).join(meta_data_.clone(), [col("bldg_id")],
            [col("bldg_id")], Default::default()
            ).rename_cols().create_temporal_features().feature_selection().drop_nulls(None);

        Self {
            data: data_,
            meta_data: meta_data_,
        }
    }

}

impl Actions for LazyFrame{
    fn rename_cols(&self) -> LazyFrame{
         self.clone().rename(
             ["out.electricity.cooling.energy_consumption..kwh"],
             ["out.electricity.AC.energy_consumption..kwh"], false)
    }
    fn create_temporal_features(&self) -> LazyFrame{
        self.clone().with_columns([
            col("timestamp").dt().weekday().alias("day of the week").cast(DataType::UInt32),
            col("timestamp").dt().hour().alias("hour of the day"),
            col("timestamp").dt().day().alias("day of the month"),
            col("timestamp").dt().ordinal_day().alias("day of the year"),
            col("timestamp").dt().week().alias("week of the year"),
            col("timestamp").dt().month().alias("month of the year"),
            col("timestamp").dt().quarter().alias("quarter")]).with_columns([
            when(col("day of the week").is_in(
                    lit(Series::new("Weekend".into(), &[6u32,7u32])), false)
                ).then(lit("Yes")).otherwise(lit("No")).alias("IsWeekend").cast_to_categorical()
        ])
    }

    fn process_meta_data_variants(&self) -> LazyFrame{
        self.clone().with_columns(
                [col("bldg_id").cast(DataType::Int32)]).
                select([col("in.occupants").cast(DataType::Int32),
                col("in.state").cast_to_categorical(),
                col("in.county").cast_to_categorical(), 
                col("in.representative_income"),
                col("in.area_median_income").cast_to_categorical(),
                col("in.income").cast_to_categorical(),
                col("in.income_recs_2020").cast_to_categorical(),
                col("in.income_recs_2015").cast_to_categorical(),
                col("in.building_america_climate_zone").cast_to_categorical(),
                col("in.ashrae_iecc_climate_zone_2004_sub_cz_split").alias("cliamte_zone"),
                col("in.bedrooms").cast(DataType::Int32),
                col("in.tenure").cast_to_categorical(),
                col("bldg_id").cast(DataType::UInt32),
                col("in.household_has_tribal_persons").cast_to_categorical()])
    }

    fn feature_selection(&self) -> LazyFrame{
        self.clone().select([
            col(
                PlSmallStr::from("^out.electricity.*|^bldg*|^day*|^hour*|^week*|^month*|^time*|^quarter|^IsWeekend|^in.*|^Short|^climate_zone$"))])
    }
}

impl ExpressionActions for Expr{

    fn cast_to_categorical(&self) -> Expr{
        let hasher= PlSeedableRandomStateQuality::seed_from_u64(42);
        let mapping=CategoricalMapping::with_hasher(usize::MAX, hasher);
        self.clone().cast(DataType::Categorical(Categories::global(), mapping.into()))
    }
}
