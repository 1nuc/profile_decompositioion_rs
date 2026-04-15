use decomposer_engine::dl::{controller::Controller, cross_validation::{CrossValidate}};
use polars::prelude::{IntoLazy, LazyFrame, col, lit};
use rand::seq::SliceRandom;
use std::{fs::{File, copy, create_dir}, path::{Path, PathBuf}};
fn main() {
    // ----Cross Validation
    // let cross_valid=CrossValidate::default();
    // cross_valid.run();
    //------One Trail Training
    Controller::default().run_training();
}

