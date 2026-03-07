use decomposer_engine::data_engine::*; 

fn main() {
    let data_source=Nrel::init();
    let data=data_source.data;
    println!("{:?}", data.collect_with_engine(polars::prelude::Engine::Gpu));
}
