use burn::{config::Config, prelude::Backend, record::{CompactRecorder, Recorder}};

use crate::dl::training::NrelConfig;

struct Inference{}
impl Inference{
    pub fn inference<B: Backend>(artifact_dir: &str, device: B::Device){
        let config=NrelConfig::load(
            format!("{artifact_dir}/config.json")).expect("unable to find the file");

        // let record=CompactRecorder::new().load(
        //     format!("{artifact_dir}/model").into(), &device).expect("training model should exist first");
    }
}
