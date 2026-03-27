use ndarray::Data;
use polars::prelude::*;
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};

pub struct Preprocessor<'a>{
    pub x: DataFrame,
    pub y: DataFrame,
    pub n: usize,
    pub x_n: usize,
    pub y_n: usize,
    pub test_size: f32,
    pub rnd_state: u64,
    pub labels: Vec<&'a str>,
    pub x_labels: Vec<&'a str>,
    pub y_labels: Vec<&'a str>,
    pub x_labels_size: usize,
    pub y_labels_size: usize,
}

impl <'a> Preprocessor<'a>{
    pub fn new(d: LazyFrame, rnd_state_: u64, test_size_: f32) -> Self {
        let (x_, y_) = Self::extract_x_nd_y(d.clone());
        let d_ = d.clone().collect().unwrap();
        Self {
            x: x_.clone(),
            y: y_.clone(),
            n: d_.height(),
            x_n: x_.height(),
            y_n: y_.height(),
            test_size: test_size_,
            rnd_state: rnd_state_,
            labels: Self::extract_labels(d_),
            x_labels: Self::extract_labels(x_.clone()),
            y_labels: Self::extract_labels(y_.clone()),
            x_labels_size: x_.shape().1,
            y_labels_size: y_.shape().1,
        }
    }

    fn extract_x_nd_y(d: LazyFrame) -> (DataFrame, DataFrame) {
        let x=d.clone().select([
            col(
                PlSmallStr::from("^day*|^hour*|^week*|^month*|^time*|^quarter|^IsWeekend|^in.*|^Short|^climate_zone$")
                ),col("out.electricity.total.energy_consumption..kwh").alias("total_usage")
        ]).drop(
        Selector::ByName {
            names: Arc::new([PlSmallStr::from_str("in.sqft")]),
            strict: true
        }).collect().unwrap();
        let y = d
            .clone()
            .select([col(PlSmallStr::from("^out.electricity.*..kwh$"))])
            .drop(Selector::ByName {
                names: Arc::new([
                    PlSmallStr::from_str("out.electricity.total.energy_consumption..kwh"),
                    PlSmallStr::from_str("out.electricity.net.energy_consumption..kwh"),
                    PlSmallStr::from_str("out.electricity.pv.energy_consumption..kwh"),
                    PlSmallStr::from_str("out.electricity.pool_heater.energy_consumption..kwh"),
                    PlSmallStr::from_str(
                        "out.electricity.hot_water_solar_th.energy_consumption..kwh",
                    ),
                    PlSmallStr::from_str("out.electricity.ev_charging.energy_consumption..kwh"),
                ]),
                strict: true,
            }).collect().unwrap();
        (x, y)
    }

    fn extract_labels(d: DataFrame) -> Vec<&'a str> {
        d.get_column_names().iter().map(|x| x.to_owned().as_str()).collect::<Vec<&str>>()
    }

    pub fn split_x_y(&self) -> (DataFrame, DataFrame, DataFrame, DataFrame) {
        let x_test_n = self.x_n as f32 * self.test_size;
        let x_train_n = self.x_n as f32 - x_test_n;

        let y_test_n = self.y_n as f32 * self.test_size;
        let y_train_n = self.y_n as f32 - y_test_n;

        let (x_train, x_test) = self.splitting(self.x.clone(), self.x_n, x_train_n);
        let (y_train, y_test) = self.splitting(self.y.clone(), self.y_n, y_train_n);

        (x_train, x_test, y_train, y_test)
    }

    fn splitting(&self, d: DataFrame, n: usize, x_n: f32) -> (DataFrame, DataFrame) {
        let mut arr: Vec<u32> = (0..n as u32).collect();
        let seed_rng = &mut <SmallRng as SeedableRng>::seed_from_u64(self.rnd_state);
        arr.shuffle(seed_rng);
        let train_arr = &arr[..x_n as usize];
        let test_arr = &arr[x_n as usize..];
        let t = ChunkedArray::from_slice("new".into(), test_arr);
        let r = ChunkedArray::from_slice("new".into(), train_arr);
        let test_t = d
            .take(&t)
            .expect("error fetching the testing data");
        let train_t = d
            .take(&r)
            .expect("error fetching the training data");
        (train_t, test_t)
    }

    // fn labelling(&self);
}
