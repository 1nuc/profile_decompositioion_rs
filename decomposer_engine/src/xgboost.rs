use polars::prelude::LazyFrame;

use xgboost::{
    parameters::{
        BoosterParameters, BoosterParametersBuilder, TrainingParameters, TrainingParametersBuilder,
        learning::{self, EvaluationMetric, LearningTaskParameters, LearningTaskParametersBuilder},
        tree::{TreeBoosterParameters, TreeBoosterParametersBuilder},
    },
    *,
};
pub struct Xgb <'a>{
    pub parametrs: TrainingParameters<'a>,
    pub d_train: DMatrix,
    pub d_test: DMatrix,
    pub booster: Booster,
    pub preds: Vec<f32>,
}

impl <'a> Xgb<'a>{

    pub fn new(d_train: DMatrix,d_test: DMatrix)-> Self{
        Self{
            d_train,
            d_test,
            parametrs: TrainingParametersBuilder::default().build().unwrap(),
            booster: Booster::new(&Self::set_booster_param()).unwrap(),
            preds: Vec::new(),
        }
    }
    pub fn set_y_train(&mut self, d: Vec<f32>){
        self.d_train
            .set_labels(&d)
            .expect("unable to set labels for DM matrix")
    }

    pub fn set_y_test(&mut self, d: Vec<f32>) {
        self.d_test
            .set_labels(&d)
            .expect("unable to set labels for DM matrix")
    }

    fn set_tree_param() -> TreeBoosterParameters {
        TreeBoosterParametersBuilder::default()
            .eta(0.1)
            .subsample(0.7)
            .build()
            .unwrap()
    }

    fn set_learning_param() -> LearningTaskParameters {
        LearningTaskParametersBuilder::default()
            .eval_metrics(learning::Metrics::Custom(vec![EvaluationMetric::MAE]))
            .objective(learning::Objective::RegLinear)
            .build()
            .unwrap()
    }

    fn set_booster_param() -> BoosterParameters {
        let learning_param = Self::set_learning_param();
        let tree_param = Self::set_tree_param();
        BoosterParametersBuilder::default()
            .booster_type(parameters::BoosterType::Tree(tree_param))
            .threads(None)
            .learning_params(learning_param)
            .build()
            .unwrap()
    }

    pub fn set_training_param(&'a mut self){
        let booster_param = Self::set_booster_param();
        self.parametrs=TrainingParametersBuilder::default()
            .dtrain(&self.d_train)
            .booster_params(booster_param)
            .boost_rounds(100)
            .build()
            .unwrap()
    }

    pub fn train(&mut self){
        self.booster=Booster::train(&self.parametrs).unwrap();
    }

    pub fn predict(&mut self) -> Vec<f32>{
        self.preds=self.booster.predict(&self.d_test).unwrap();
        self.preds.clone()
    }
    pub fn r2_score(){

    }
}
