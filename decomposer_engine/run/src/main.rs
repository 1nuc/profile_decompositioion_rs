#[allow(unused_imports)]
use decomposer_engine::dl::{controller::Controller, cross_validation::{CrossValidate}};
fn main() {
    // ----Cross Validation
    // let cross_valid=CrossValidate::default();
    // cross_valid.run();
    //------One Trail Training
    Controller::default().run_training_multiple_processes();
}

