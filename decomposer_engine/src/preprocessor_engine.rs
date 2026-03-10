use polars::prelude::LazyFrame;

struct Preprocessor <'a>{
    x: LazyFrame,
    y: LazyFrame,
    n: usize,
    test_size: f32,
    rnd_state: u64,
    labels: Vec<&'a str>,
}
impl <'a> Preprocessor<'a>{
    fn new();
    fn extract_x_nd_y();
    fn split_x_y();
    fn splitting();
    fn labelling();
}
