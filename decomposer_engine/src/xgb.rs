use crate::Actions;
use polars::prelude::*;

use polars::prelude::{LazyFrame, PlSmallStr};

use xgboost::{
    parameters::{
        BoosterParameters, BoosterParametersBuilder, TrainingParameters, TrainingParametersBuilder,
        learning::{self, EvaluationMetric, LearningTaskParameters, LearningTaskParametersBuilder},
        tree::{TreeBoosterParameters, TreeBoosterParametersBuilder, TreeMethod},
    },
    *,
};

pub struct Xgb {
    pub d_train: DMatrix,
    pub d_test: DMatrix,
    pub booster: Vec<Booster>,
    pub preds: Vec<Vec<f32>>,
}

impl Xgb {
    pub fn new(d_train: DMatrix, d_test: DMatrix) -> Self {
        Self {
            d_train,
            d_test,
            booster: vec![Booster::new(&Self::set_booster_param()).unwrap()],
            preds: Vec::new(),
        }
    }
    pub fn set_y_train(&mut self, d: Vec<f32>) -> &mut Self {
        self.d_train
            .set_labels(&d)
            .expect("unable to set labels for DM matrix");
        self
    }

    pub fn set_y_test(&mut self, d: Vec<f32>) -> &mut Self {
        self.d_test
            .set_labels(&d)
            .expect("unable to set labels for DM matrix");
        self
    }

    fn set_tree_param() -> TreeBoosterParameters {
        TreeBoosterParametersBuilder::default()
            .eta(0.1)
            .subsample(0.7)
            .tree_method(TreeMethod::Hist)
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

    pub fn set_training_param<'a>(&'a self) -> TrainingParameters<'a> {
        let booster_param = Self::set_booster_param();
        TrainingParametersBuilder::default()
            .dtrain(&self.d_train)
            .booster_params(booster_param)
            .boost_rounds(100)
            .build()
            .unwrap()
    }
    pub fn modelling(&mut self) -> &mut Self {
        let param = self.set_training_param();
        let boost=Arc::new(Booster::train(&param).unwrap());
        let boost_clone=Arc::clone(&boost);
        self.booster.push(Arc::into_inner(boost).unwrap());
        self.preds.push(boost_clone.predict(&self.d_test).unwrap());
        self
    }

    pub fn evaluation(&self, booster: Booster) {
        let metric = booster.evaluate(&self.d_test).unwrap();
        println!("{:?}", metric);
    }

    pub fn r2_score(&self,preds: Vec<f32>) -> f32 {
        //1- Total sum of residuals / total sum of squares
        let y_true = self.d_test.get_labels().unwrap();
        let mean = y_true.iter().sum::<f32>() / y_true.len() as f32;
        let total_sum_residuals = y_true
            .iter()
            .zip(preds.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>();
        let total_sum_squares = y_true.iter().map(|x| (x - mean).powi(2)).sum::<f32>();

        1_f32 - (total_sum_residuals / total_sum_squares)
    }

    pub fn train(&mut self, y_train: LazyFrame, y_test: LazyFrame){
        // loop through the columns
        // if the index is 0 train the first column
        // if the index is not zero containue updating the model
        let cols = y_train.return_cols();
        cols.iter().for_each(|x|{
            let y_train = y_train
                .clone()
                .select([col(PlSmallStr::from_string(x.clone()))]);
            let y_test = y_test
                .clone()
                .select([col(PlSmallStr::from_string(x.clone()))]);
            self.set_y_train(y_train.to_1d_vec())
                .set_y_test(y_test.to_1d_vec());
            self.modelling();
        });
    }
}
