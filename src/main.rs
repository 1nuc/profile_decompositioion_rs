use std::{fs::read_dir, path::PathBuf};

use axum::{Router, http::Response, response::IntoResponse, routing::get, serve};
use decomposer_engine::{Actions, EagerActions, data_engine::*, dl::controller::{self, Controller}, xgb};

#[tokio::main]
async fn main(){
    let app=Router::new().route("/", get(send_data));
    let listner=tokio::net::TcpListener::bind("localhost:8000").await.unwrap();
    serve(listner, app).await.unwrap();
    // controller::run_train();
    // controller::process_chunks();
    // let data_source=Nrel::init();
    // let data=data_source.data;
    // let mut encoded_data=data.clone().encode_categoricals();
    //
    // // Deep Learning Model
    // let s=encoded_data.clone().collect().unwrap();
    // let y_columns=s.return_y_columns();
    // let modelling_data=encoded_data.standard_scalar(y_columns.clone()).return_time_sequenced().collect().unwrap();
    // let control=Controller::new(modelling_data);
    // control.lstm_simulation();
    // --- Xgboost Model
    // xgb::Xgb::runner(encoded_data);

}

async fn send_data()-> impl IntoResponse{
    let msg="Decomposer says hi".to_string();
    Response::new(msg)
}
