use burn::{nn::Lstm, prelude::Backend, *};

struct NucLstmackend<B> where  B: Backend{
    model: Lstm<B>,
}

impl <B: Backend> NucLstmackend<B>{
    fn new(){

    }
}
