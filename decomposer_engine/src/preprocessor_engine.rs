use polars::prelude::*;

struct Preprocessor {
    x: LazyFrame,
    y: LazyFrame,
    n: usize,
    test_size: f32,
    rnd_state: u64,
    labels: Vec<String>,
    x_labels: Vec<String>,
    y_labels: Vec<String>,
    x_labels_size: usize,
    y_labels_size: usize,
}

impl Preprocessor{
    fn new(&self, d: LazyFrame, rnd_state_: u64, test_size_: f32) -> Self{
        let (x_, y_)=Self::extract_x_nd_y(d.clone());
        Self{
            x: x_.clone(),
            y: y_.clone(),
            n: d.clone().collect().unwrap().height(),
            test_size: test_size_,
            rnd_state: rnd_state_,
            labels: Self::extract_labels(d),
            x_labels: Self::extract_labels(x_.clone()),
            y_labels: Self::extract_labels(y_.clone()),
            x_labels_size: x_.collect().unwrap().shape().1,
            y_labels_size: y_.collect().unwrap().shape().1,
        }
    }

    fn extract_x_nd_y(d: LazyFrame) -> (LazyFrame, LazyFrame){
        let x=d.clone().select([
            col(
                PlSmallStr::from("^day*|^hour*|^week*|^month*|^time*|^quarter|^IsWeekend|^in.*|^Short|^climatezone$")
                ),col("out.electricity.total.energy_consumption").alias("total_usage")
        ]);
        let y=d.clone().select([
            col(
                PlSmallStr::from("^out.electricity.*..kwh$")
            )]);
        (x,y)
    }

    fn extract_labels(d: LazyFrame) -> Vec<String>{
          d.clone().collect_schema(
          ).unwrap().iter_names().map(
                |x| x.as_str().to_string()
            ).collect::<Vec<String>>()
    }

    // fn split_x_y(&self);
    // fn splitting(&self);
    // fn labelling(&self);
}
