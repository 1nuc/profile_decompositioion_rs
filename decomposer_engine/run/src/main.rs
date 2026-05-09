#[allow(unused_imports)]
use decomposer_engine::{Actions, data_engine::Nrel, dl::controller::Controller, xgb::Xgb};

fn main(){
    // -- Deep learning training
    // Controller::default().run_training_multiple_processes();

    // --Xgboost training
    // let data_source = Nrel::init("../../../../input/*".into());
    // let data = data_source.data;
    // let encoded_data = data.clone().encode_categoricals();
    // Xgb::runner(encoded_data);
}
