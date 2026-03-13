use xgboost::{parameters::{BoosterParameters, BoosterParametersBuilder, TrainingParameters, TrainingParametersBuilder, learning::{self, EvaluationMetric, LearningTaskParameters, LearningTaskParametersBuilder}, tree::{TreeBoosterParameters, TreeBoosterParametersBuilder}}, *};
struct Xgb{
    d_train: DMatrix,
    d_test: DMatrix,
}

impl Xgb{

    fn new(x_train: Vec<f32>, x_test: Vec<f32>) -> Self{
        Self { 
            d_train: Self::set_data(x_train, num_rows),
            d_test: Self::set_data(x_test, num_rows),
        }
    }

    fn set_data(d: Vec<f32>, num_rows: usize) -> DMatrix{
        DMatrix::from_dense(&d, num_rows).expect("Unable to create DM matrix")
    }
    
    fn set_y_train(&mut self,d: Vec<f32>){
        self.d_train.set_labels(&d).expect("unable to set labels for DM matrix")
    }

    fn set_x_train(&mut self,d: Vec<f32>){
        self.d_test.set_labels(&d).expect("unable to set labels for DM matrix")
    }

    fn set_tree_param() -> TreeBoosterParameters{
        TreeBoosterParametersBuilder::default().
            eta(0.1).subsample(0.7).build().unwrap()
    }

    fn set_learning_param() -> LearningTaskParameters{
        LearningTaskParametersBuilder::default().
            eval_metrics(learning::Metrics::Custom(vec![EvaluationMetric::MAE])).
            objective(learning::Objective::RegLinear).
            build().unwrap()
    }

    fn set_booster_param() -> BoosterParameters{
        let learning_param=Self::set_learning_param();
        let tree_param=Self::set_tree_param();
        BoosterParametersBuilder::default().
            booster_type(parameters::BoosterType::Tree(tree_param)).
            threads(None).learning_params(learning_param).
            build().unwrap()
    }

    fn set_training_param(&mut self) -> TrainingParameters{
        let booster_param=Self::set_booster_param();
        TrainingParametersBuilder::default().
            dtrain(&self.d_train).booster_params(booster_param).
            boost_rounds(100).build().unwrap()

    }
}
