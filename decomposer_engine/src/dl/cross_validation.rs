use std::{fs::{File, copy, create_dir}, path::{Path, PathBuf}};

use polars::{frame::DataFrame, prelude::{IntoLazy, LazyFrame, col, lit}, *};
use rand::seq::SliceRandom;

use crate::dl::controller::Controller;

pub struct CrossValidate{
    pub training_sets: Vec<DataFrame>,
    pub testing_sets: Vec<DataFrame>,
    pub k_fold: usize,
}
impl Default for CrossValidate{
    fn default() -> Self {
        // set the default k to 8 folds
        Self::new(8)
    }
}
impl CrossValidate{

    pub fn new(k_fold: usize)-> Self{
        let (training_sets, testing_sets)=Self::split_temporal_data(k_fold);
        Self{
            training_sets,
            testing_sets,
            k_fold,
        }
    }

    #[allow(unused_must_use)]
    pub fn split_temporal_data(k: usize)-> (Vec<DataFrame>, Vec<DataFrame>){

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
        Self::cross_valid(data.lazy(), k)
    }

    pub fn cross_valid(data: LazyFrame, k:usize) -> (Vec<DataFrame>, Vec<DataFrame>){
        // generating random number for sampling the time series across the month
        // defining the range
        let mut rnd=rand::rng();

        //creating a vector that contains the list of all available months
        let mut month_vec: Vec<i8>=(1..=12).collect();
        //shuffle the vector based on that rng
        month_vec.shuffle(&mut rnd);
        // extract the months with the size of k 
        let rand_months=month_vec.into_iter().take(k + 1).collect::<Vec<i8>>();
        let mut train_sets=Vec::new();
        let mut test_sets=Vec::new();
        rand_months.into_iter().for_each(|x|{
            let test_data=data.clone().lazy().filter(col("month of the year").eq(lit(x))).collect().unwrap();
            let train_data=data.clone().lazy().filter(col("month of the year").neq(lit(x))).collect().unwrap();
            train_sets.push(train_data);
            test_sets.push(test_data);
        });
        (train_sets, test_sets)
    }
}
