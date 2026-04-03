use std::sync::{Arc, Mutex};

use axum::{Json, Router, extract::{Path, State}, http::Response, response::IntoResponse, routing::get};
use decomposer_engine::{Actions, EagerActions, data_engine::*, dl::controller::{self, Controller}, xgb};
use polars::frame::DataFrame;

#[tokio::main]
async fn main(){
    let shared_state = Arc::new(Mutex::new(Controller::default()));
    serve(shared_state).await;
}

async fn serve(shared_state: Arc<Mutex<Controller>>){
    let app=Router::new()
        .route("/", get(welcome))
        .route("/buildings", get(send_bldg))
        .route("/predictions/{bldg_id}", get(send_data)).with_state(shared_state);
    let listner=tokio::net::TcpListener::bind("localhost:8000").await.unwrap();
    axum::serve(listner, app).await.unwrap();
}

// Return the available buildings in the data
async fn send_bldg(State(state): State<Arc<Mutex<Controller>>>)-> Json<Vec<String>>{
    let lock=state.lock().unwrap();
    let buildings=lock.return_nrel_buildings();
    Json(buildings)
}

async fn send_data(State(state): State<Arc<Mutex<Controller>>>, Path(bldg_id): Path<String>)-> Json<DataFrame>{
    let mut lock=state.lock().unwrap();
    let data=lock.infer_one_building(&bldg_id);
    Json(data)
}

// async fn send_prediction(State(state): State<Arc<Controller>>)
async fn welcome()-> impl IntoResponse{
    let msg="Decomposer says hi".to_string();
    Response::new(msg)
}

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
