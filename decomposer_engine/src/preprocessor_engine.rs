use polars::prelude::*;

struct Preprocessor <'a>{
    x: LazyFrame,
    y: LazyFrame,
    n: usize,
    test_size: f32,
    rnd_state: u64,
    labels: Vec<&'a str>,
    x_labels_size: u32,
    y_labels_size: u32,
    x_labels: Vec<&'a str>,
    y_labels: Vec<&'a str>,
}
impl <'a> Preprocessor<'a>{
    fn new(&self);

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
    fn split_x_y(&self);
    fn splitting(&self);
    fn labelling(&self);
}
