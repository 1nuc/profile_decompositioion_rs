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
    let data=LazyFrame::scan_parquet("../../../datasets/*.parquet", args).expect("error");
    let length=data.select([len()]).with_streaming(true).collect();
}
