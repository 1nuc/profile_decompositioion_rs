use burn::{
    config::Config, module::Module, nn::{
        BiLstm, BiLstmConfig, Linear, LinearConfig, Relu, conv::{Conv1d, Conv1dConfig}, loss::MseLoss
    }, prelude::Backend, tensor::{TensorCreationOptions, backend::AutodiffBackend}, train::{
        InferenceStep, ItemLazy,TrainOutput, TrainStep, metric::{
            Adaptor, LossInput}}, *};

use crate::dl::dataset::NrelBatch;


//Prepare the configurations of the model
#[derive(Config, Debug)]
pub struct Seq2SeqConfig{
    input_size: usize,
    output_size: usize,
    hidden_size: usize,
    // num_layers: usize,
    dropout: f32,
}
//Implementing default for NucLstmConfig
impl Default for Seq2SeqConfig{
    fn default() -> Self {
        Self{
            input_size: 22,
            output_size: 24,
            hidden_size: 128,
            dropout: 0.5, //weight decay to prevent overfitting
        }
    }
}
//Initializing the model configurations 
impl Seq2SeqConfig{
    pub fn init<B: Backend>(&self, device: B::Device) -> Seq2Seq<B>{
        let encoder_1=Conv1dConfig::new(self.input_size, self.hidden_size, 7)
            .with_padding(nn::PaddingConfig1d::Same).init(&device);
        let encoder_2=Conv1dConfig::new(self.input_size, self.hidden_size, 11)
            .with_padding(nn::PaddingConfig1d::Same).init(&device);
        let encoder_3=Conv1dConfig::new(self.input_size, self.hidden_size, 15)
            .with_padding(nn::PaddingConfig1d::Same).init(&device);
        let decoder=BiLstmConfig::new(self.hidden_size * 3, self.hidden_size, true).with_batch_first(true).init(&device);
        let output_model=LinearConfig::new(self.hidden_size *2, self.output_size).init(&device);
        Seq2Seq{
           encoder_1,
           encoder_2,
           encoder_3,
           decoder,
           output_model,
        }
    }
}
//TODO: Prepare the output type to be a sequence
pub struct NrelSequenceOutput<B: Backend>{
    loss: Tensor<B, 1>,
    output: Tensor<B, 3>,
    targets: Tensor<B, 3>,
}

//Apply the adoptor so the loss is calculated accordingly
impl <B: Backend>Adaptor<LossInput<B>>for NrelSequenceOutput<B>{
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}
//implement sync for the implement to be used in the train step.
impl <B: Backend> ItemLazy for NrelSequenceOutput<B>{

    type ItemSync = NrelSequenceOutput<B>; 
    fn sync(self) -> Self::ItemSync{
        Self{
            loss: self.loss,
            output: self.output,
            targets: self.targets,
        }
    }
}

//Model
#[derive(Module, Debug)]
pub struct Seq2Seq<B :Backend>{
    encoder_1: Conv1d<B>,
    encoder_2: Conv1d<B>,
    encoder_3: Conv1d<B>,
    decoder: BiLstm<B>,
    output_model: Linear<B>,
}

impl <B: Backend>Seq2Seq<B> {
    //the forward function for which the weights neurons are multiplied
    pub fn forward(&self, input: Tensor<B,3>) -> Tensor<B, 3>{
        let data=input.permute([0,2,1]);
        // get the output from the first encoder
        let encoder_output_1=self.encoder_1.forward(data.clone());
        //get the output from the second encoder
        let encoder_output_2=self.encoder_1.forward(data.clone());
        // get the output from the third encoder
        let encoder_output_3=self.encoder_1.forward(data);

        //concatinate all the results to the column dimension so it will be 128 * 3
        let cnn_output=Tensor::cat(vec![encoder_output_1, encoder_output_2, encoder_output_3], 1);
        //set the order back to what it was
        let output=cnn_output.permute([0,2,1]);
        let (lstm_output, _) =self.decoder.forward(output, None);
        Relu::new().forward(self.output_model.forward(lstm_output))
        
    }
    // Calculating the loss function of the forward step
    pub fn forward_step(&self, items: NrelBatch<B>) ->NrelSequenceOutput<B>{
        let targets: Tensor<B, 3>=items.target;
        let output=self.forward(items.sequence);
        let loss=MseLoss::new().forward(output.clone(), targets.clone(), nn::loss::Reduction::Mean);
        NrelSequenceOutput{
            loss,
            output,
            targets,
        }
    }
}

//Implementing the training step for the model to obtain the gradients (weights after optimization)
impl <B: AutodiffBackend>TrainStep for Seq2Seq<B>{
    type Output= NrelSequenceOutput<B>;
    type Input=NrelBatch<B>;
    fn step(&self, item: Self::Input) -> TrainOutput<Self::Output> {
        let item=self.forward_step(item);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}
// Prepare the Inference step to redo the process after calculating the gradients
impl <B: Backend> InferenceStep for Seq2Seq<B>{
    type Input = NrelBatch<B>;
    type Output= NrelSequenceOutput<B>;
    fn step(&self, item: Self::Input) -> Self::Output {
        self.forward_step(item)
    }
}
