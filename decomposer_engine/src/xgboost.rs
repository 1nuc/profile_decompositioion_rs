use xgboost::{parameters::{BoosterParameters, TrainingParameters, learning::LearningTaskParameters, tree::{TreeBoosterParameters, TreeBoosterParametersBuilder}}, *};
struct Xgb<'a>{
    tree_param: TreeBoosterParameters,
    learning_param: LearningTaskParameters,
    booster_param: BoosterParameters,
    parameters: TrainingParameters<'a>,
    d_train: DMatrix,
    d_test: DMatrix,
}

impl <'a> Xgb<'a>{

    fn new() -> Self{
        Self{
            ..Default::default(),
        }
    }

    fn set_data(d: Vec<f32>, num_rows: usize) -> DMatrix{
        DMatrix::from_dense(&d, num_rows).expect("Unable to create DM matrix")
    }
    
    fn set_y_train(&mut self,d: Vec<f32>) -> (){
        self.d_train.set_labels(&d).expect("unable to set labels for DM matrix")
    }

    fn set_x_train(&mut self,d: Vec<f32>) -> (){
        self.d_test.set_labels(&d).expect("unable to set labels for DM matrix")
    }

    fn set_tree_param() -> TreeBoosterParameters{
        TreeBoosterParametersBuilder::default().
            eta(0.1).subsample(0.7).build().unwrap()
    }
}
