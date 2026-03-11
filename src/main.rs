use decomposer_engine::{Actions, data_engine::*, preprocessor_engine::Preprocessor}; 
use polars::prelude::*;
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};
use smartcore::linalg::basic::matrix::DenseMatrix;
use tap::Conv;
fn main() {

    //TODO: Finish the train test and split function in polars Done
    //TODO: train the xgboost
    //TODO: Extract x and y features Done
    let data_source=Nrel::init();
    let mut data=data_source.data;
    let encoded_data=data.encode_categoricals();
    let preprocessor=Preprocessor::new(encoded_data.clone(), 42, 0.3);
    let (x_train, x_test, y_train, y_test)=preprocessor.split_x_y();
    let y_train=y_train.select([col("out.electricity.AC.energy_consumption..kwh")]);
    let y_test=y_test.select([col("out.electricity.AC.energy_consumption..kwh")]);

    let x_train=DenseMatrix::from_2d_vec(&x_train.to_vec());


}

    // let dense_file=DenseMatrix::from_2d_vec(&vec_t).unwrap();
    // let transformer =StandardScaler::fit(&dense_file, Default::default()).unwrap();
    // let transformed_data=transformer.transform(&dense_file);
