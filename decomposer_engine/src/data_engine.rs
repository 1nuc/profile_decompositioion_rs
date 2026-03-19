use crate::{Actions, ExpressionActions};
use ndarray::Array2;
use polars::prelude::*;
use xgboost::DMatrix;

pub struct Nrel {
    pub data: LazyFrame,
    pub meta_data: LazyFrame,
}

impl Nrel {
    fn scan_files(path: PlRefPath) -> LazyFrame {
        LazyFrame::scan_parquet(path, Default::default()).expect("Error reading the file")
    }

    pub fn init() -> Self {
        let meta_data_ = Self::scan_files("../metadata/MetaData.parquet".into())
            .process_meta_data_variants()
            .unique(None, Default::default());
        let data_ = Self::scan_files("../src/input/*.parquet".into())
            .join(
                meta_data_.clone(),
                [col("bldg_id")],
                [col("bldg_id")],
                Default::default(),
            )
            .rename_cols()
            .create_temporal_features()
            .feature_selection()
            .drop_nulls(None);

        Self {
            data: data_,
            meta_data: meta_data_,
        }
    }
}

impl Actions for LazyFrame {
    // Rename the vague fields to a more brief and descriptive names
    fn rename_cols(&self) -> Self {
        self.clone().rename(
            ["out.electricity.cooling.energy_consumption..kwh"],
            ["out.electricity.AC.energy_consumption..kwh"],
            false,
        )
    }
    //Creating the temporal features to boost the model's performance
    fn create_temporal_features(&self) -> Self {
        self.clone()
            .with_columns([
                col("timestamp")
                    .dt()
                    .weekday()
                    .alias("day of the week")
                    .cast(DataType::UInt32),
                col("timestamp").dt().hour().alias("hour of the day"),
                col("timestamp").dt().day().alias("day of the month"),
                col("timestamp").dt().ordinal_day().alias("day of the year"),
                col("timestamp").dt().week().alias("week of the year"),
                col("timestamp").dt().month().alias("month of the year"),
                col("timestamp").dt().quarter().alias("quarter"),
            ])
            .with_columns([
                when(
                    col("day of the week")
                        .is_in(lit(Series::new("Weekend".into(), &[6u32, 7u32])), false),
                )
                .then(lit("Yes"))
                .otherwise(lit("No"))
                .alias("IsWeekend")
                .cast_to_categorical(),
                col("timestamp").cast(DataType::Datetime(
                    TimeUnit::Milliseconds,
                    Some(TimeZone::UTC),
                )),
            ])
    }

    //process the meta data columns and prepare them for further preprocessing
    fn process_meta_data_variants(&self) -> Self {
        self.clone()
            .with_columns([col("bldg_id").cast(DataType::UInt32)])
            .select([
                col("in.occupants").cast(DataType::UInt32),
                col("in.state").cast_to_categorical(),
                col("in.county").cast_to_categorical(),
                col("in.representative_income"),
                col("in.area_median_income").cast_to_categorical(),
                col("in.income").cast_to_categorical(),
                col("in.income_recs_2020").cast_to_categorical(),
                col("in.income_recs_2015").cast_to_categorical(),
                col("in.building_america_climate_zone").cast_to_categorical(),
                col("in.ashrae_iecc_climate_zone_2004_sub_cz_split")
                    .alias("climate_zone")
                    .cast_to_categorical(),
                col("in.bedrooms").cast(DataType::UInt32),
                col("in.tenure").cast_to_categorical(),
                col("bldg_id").cast(DataType::UInt32),
                col("in.household_has_tribal_persons").cast_to_categorical(),
            ])
    }

    // Specify the selection of the inclusive variables for modelling
    fn feature_selection(&self) -> Self {
        self.clone().select([
            col(
                PlSmallStr::from("^out.electricity.*|^bldg*|^day*|^hour*|^week*|^month*|^time*|^quarter|^IsWeekend|^in.*|^Short|^climate_zone$"))])
    }

    fn return_cols(&self) -> Vec<String> {
        self.clone()
            .collect_schema()
            .unwrap()
            .iter_names()
            .map(|x| x.as_str().to_string())
            .collect::<Vec<String>>()
    }
    // Extracting the categorical columns
    fn categorical_cols(&mut self) -> Vec<Expr> {
        self.collect_schema()
            .unwrap()
            .iter_names_and_dtypes()
            .filter_map(|c| {
                if c.1.is_categorical() {
                    Some(col(c.0.as_str()))
                } else {
                    None
                }
            })
            .collect::<Vec<Expr>>()
    }

    fn return_time_sequenced(&self) -> Self {
        let options = DynamicGroupOptions {
            index_column: PlSmallStr::from_str("timestamp"),
            every: Duration::parse("1d"),
            period: Duration::parse("1d"),
            offset: Duration::parse("1d"),
            ..Default::default()
        };
        self.clone()
            .sort(vec![PlSmallStr::from_str("timestamp")], Default::default())
            .group_by_dynamic(col("timestamp"), [], options)
            .agg([col("*")]).with_columns([col("timestamp").dt().timestamp(TimeUnit::Milliseconds)])
    }

    // Encode categorical columns in the data to UInt32 Type
    fn encode_categoricals(&mut self) -> Self {
        let cols = self.categorical_cols();
        self.clone().with_columns(
            cols.iter()
                .map(|x| x.clone().cast(DataType::UInt32))
                .collect::<Vec<_>>(),
        )
    }

    fn standard_scalar(&mut self) -> Self {
        self.clone()
            .with_columns([(col("*") - col("*").mean()) / col("*").std(1)])
            .fill_nan(0)
    }

    fn min_max_scalar(&mut self) -> Self {
        self.clone()
            .with_columns([(col("*") - col("*").min()) / (col("*").max() - col("*").min())])
            .fill_nan(0)
    }

    fn to_ndarry(&self) -> Array2<f32> {
        self.clone()
            .collect()
            .unwrap()
            .to_ndarray::<Float32Type>(IndexOrder::C)
            .expect("unable to to return an array")
    }

    fn to_2d_vec(&self) -> Vec<Vec<f32>> {
        let n_cols = self.clone().collect().unwrap().height();
        let array_d = Self::to_ndarry(self);
        let to_1d_vec = array_d.into_raw_vec_and_offset().0;
        to_1d_vec
            .chunks(n_cols)
            .map(|x| x.to_vec())
            .collect::<Vec<Vec<f32>>>()
    }

    fn to_1d_vec(&self) -> Vec<f32> {
        let array_d = Self::to_ndarry(self);
        array_d.into_raw_vec_and_offset().0
    }
    fn to_matrix(&mut self, with_scalar: bool) -> DMatrix {
        let data = if with_scalar {
            self.standard_scalar().to_1d_vec()
        } else {
            self.to_1d_vec()
        };
        let num_rows = self.clone().collect().unwrap().height();
        DMatrix::from_dense(&data, num_rows).expect("Unable to create DM matrix")
    }
}

impl ExpressionActions for Expr {
    // Fixing current polars limitation with categoricals conversion
    fn cast_to_categorical(&self) -> Self {
        let hasher = PlSeedableRandomStateQuality::fixed();
        let mapping = CategoricalMapping::with_hasher(usize::MAX, hasher);
        self.clone()
            .cast(DataType::Categorical(Categories::global(), mapping.into()))
    }
}

#[cfg(test)]
mod test {

    use super::*;
    #[test]
    fn trial() {
        let data = df![
            "in.state"=> ["CA", "TX", "NY", "CA", "TX"]
        ]
        .unwrap()
        .lazy();
        let encoded_c = data.with_columns([col("in.state").cast_to_categorical()]);
        let encoded = encoded_c.with_columns([col("in.state").cast(DataType::UInt32)]);
        let values = encoded
            .collect()
            .unwrap()
            .column("in.state")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .collect::<Vec<_>>();
        println!(
            "CA={:?}, {:?}, TX={:?}, {:?}, NY={:?}",
            values[0], values[3], values[1], values[4], values[2]
        );
        assert_eq!(values[0], values[3]);
        assert_eq!(values[1], values[4]);
    }
    #[test]
    fn trial2() {
        let data = df![
            "in.state"=> ["CA", "TX", "NY", "CA", "TX"]
        ]
        .unwrap()
        .lazy();
        let encoded_c = data.with_columns([col("in.state").cast_to_categorical()]);
        let encoded = encoded_c.with_columns([col("in.state").cast(DataType::UInt32)]);
        let values = encoded
            .collect()
            .unwrap()
            .column("in.state")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .collect::<Vec<_>>();
        println!(
            "CA={:?}, {:?}, TX={:?}, {:?}, NY={:?}",
            values[0], values[3], values[1], values[4], values[2]
        );
        assert_eq!(values[0], values[3]);
        assert_eq!(values[1], values[4]);
    }
}
