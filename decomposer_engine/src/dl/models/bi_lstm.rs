use burn::{
    config::Config,
    module::Module,
    nn::{BiLstm, BiLstmConfig, Linear, LinearConfig, Relu, loss::MseLoss},
    prelude::Backend,
    tensor::backend::AutodiffBackend,
    train::{
        InferenceStep, ItemLazy, TrainOutput, TrainStep,
        metric::{Adaptor, LossInput},
    },
    *,
};

use crate::dl::dataset::NrelBatch;

//Prepare the configurations of the model
#[derive(Config, Debug)]
pub struct NucBiLstmConfig {
    input_size: usize,
    output_size: usize,
    hidden_size: usize,
    dropout: f32,
}
//Implementing default for NucLstmConfig
impl Default for NucBiLstmConfig {
    fn default() -> Self {
        Self {
            input_size: 21,
            output_size: 24,
            hidden_size: 128,
            dropout: 0.5, //weight decay to prevent overfitting
        }
    }
}
//Initializing the model configurations
impl NucBiLstmConfig {
    pub fn init<B: Backend>(&self, device: B::Device) -> NucBiLstm<B> {
        let model = BiLstmConfig::new(self.input_size, self.hidden_size, true)
            .with_batch_first(true)
            .init(&device);
        let output_model = LinearConfig::new(self.hidden_size * 2, self.output_size).init(&device);
        NucBiLstm {
            model,
            output_model,
        }
    }
}
//TODO: Prepare the output type to be a sequence
pub struct NrelSequenceOutput<B: Backend> {
    loss: Tensor<B, 1>,
    output: Tensor<B, 3>,
    targets: Tensor<B, 3>,
}

//Apply the adoptor so the loss is calculated accordingly
impl<B: Backend> Adaptor<LossInput<B>> for NrelSequenceOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}
//implement sync for the implement to be used in the train step.
impl<B: Backend> ItemLazy for NrelSequenceOutput<B> {
    type ItemSync = NrelSequenceOutput<B>;
    fn sync(self) -> Self::ItemSync {
        Self {
            loss: self.loss,
            output: self.output,
            targets: self.targets,
        }
    }
}

//Model
#[derive(Module, Debug)]
pub struct NucBiLstm<B: Backend> {
    model: BiLstm<B>,
    output_model: Linear<B>,
}

impl<B: Backend> NucBiLstm<B> {
    //the forward function for which the weights neurons are multiplied
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let (lstm_output, _) = self.model.forward(input, None);
        Relu::new().forward(self.output_model.forward(lstm_output))
    }
    // Calculating the loss function of the forward step
    pub fn forward_step(&self, items: NrelBatch<B>) -> NrelSequenceOutput<B> {
        let targets: Tensor<B, 3> = items.target;
        let output = self.forward(items.sequence);
        let loss =
            MseLoss::new().forward(output.clone(), targets.clone(), nn::loss::Reduction::Mean);
        //returning the output and target with their losses
        NrelSequenceOutput {
            loss,
            output,
            targets,
        }
    }
}

//Implementing the training step for the model to obtain the gradients (weights after optimization)
impl<B: AutodiffBackend> TrainStep for NucBiLstm<B> {
    type Output = NrelSequenceOutput<B>;
    type Input = NrelBatch<B>;
    fn step(&self, item: Self::Input) -> TrainOutput<Self::Output> {
        // Here where the model actually trains and starts calculating the gradients
        let item = self.forward_step(item);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}
// Prepare the Inference step to redo the process after calculating the gradients
impl<B: Backend> InferenceStep for NucBiLstm<B> {
    // The inference step of the model
    type Input = NrelBatch<B>;
    type Output = NrelSequenceOutput<B>;
    fn step(&self, item: Self::Input) -> Self::Output {
        self.forward_step(item)
    }
}
