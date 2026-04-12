use decomposer_engine::dl::controller::Controller;
use ndarray::Array;
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use polars::prelude::{IntoLazy, LazyFrame, col, lit};
use std::{fs::{File, copy, create_dir}, path::{Path, PathBuf}};
fn main() {
    let mut controller=Controller::default();
    let files=controller.train_files.clone().into_iter().take(10).collect::<Vec<PathBuf>>();
    files.into_iter().for_each(|x|{
        let input_path=Path::new("input");
        if !input_path.exists(){
            create_dir(input_path);
        }
        let file_name= Path::new(x.file_name().unwrap().to_str().unwrap()); 
        let new_file_path=input_path.join(file_name);
        if !new_file_path.exists(){
            File::create_new(new_file_path.clone());
        }
        copy(x, new_file_path).expect("unable to copy the files");
    });
    let data=controller.data_preparation(("input/*").into(), true).unwrap();
    // months format from 0 to 12
    let months=data.column("month of the year").unwrap().unique().unwrap();

    // let test_data=data.clone().lazy().filter(col("month of the year").eq(lit(2))).collect().unwrap();
    // let train_data=data.lazy().filter(col("month of the year").neq(lit(2))).collect().unwrap();
    cross_valid(data.lazy(), 7);
}

fn cross_valid(data: LazyFrame, k: i32){
    let rand_months=Array::random((1, k as usize), Uniform::new(1., 12.).unwrap()).into_raw_vec_and_offset().0;
    let mut train_sets=Vec::new();
    let mut test_sets=Vec::new();
    rand_months.into_iter().for_each(|x|{
        let test_data=data.clone().lazy().filter(col("month of the year").eq(lit(x as i8))).collect().unwrap();
        let train_data=data.clone().lazy().filter(col("month of the year").neq(lit(x as i8))).collect().unwrap();
        train_sets.push(train_data);
        test_sets.push(test_data);
    });

    println!("{:?}", train_sets.pop().unwrap());
    println!("{:?}", test_sets.pop().unwrap());
}
