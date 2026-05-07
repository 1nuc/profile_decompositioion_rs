use std::{fs, sync::{Arc, Mutex}};
use std::fs::remove_dir_all;
use serde::Serialize;
use axum::{Json, Router, extract::{Path, State}, http::Response, response::IntoResponse, routing::get};
use decomposer_engine::{dl::controller::{Controller} };

#[derive(Serialize)]
struct Metrics{
    eval_metrics: String,
    cross_val_metrics: String,
}
#[tokio::main]
async fn main(){
    tracing_subscriber::fmt::init();
    let shared_state = Arc::new(Mutex::new(Controller::default()));
    serve(shared_state).await;
}

async fn serve(shared_state: Arc<Mutex<Controller>>){
    let app=Router::new()
        .route("/", get(welcome))
        .route("/buildings", get(send_bldg))
        .route("/train_one_trail", get(trail_train))
        .route("/predictions/{bldg_id}", get(send_data)).with_state(shared_state.clone())
        .route("/metrics", get(send_metrics)).with_state(shared_state);
    let listner=tokio::net::TcpListener::bind("localhost:8000").await.unwrap();
    axum::serve(listner, app).await.unwrap();
}

// Return the available buildings in the data
async fn send_bldg(State(state): State<Arc<Mutex<Controller>>>)-> Json<Vec<String>>{
    let lock=state.lock().expect("Error while fetching the buildings");
    let buildings=lock.return_nrel_buildings();
    drop(lock);
    Json(buildings)
}

async fn trail_train(State(state): State<Arc<Mutex<Controller>>>)-> impl IntoResponse{
    let mut lock=state.lock().expect("Error while fetching the buildings");
    lock.one_trail_training();
    drop(lock);
    Json("Training finished".to_string())
}

#[allow(unused_must_use)]
async fn send_data(State(state): State<Arc<Mutex<Controller>>>, Path(bldg_id): Path<String>)-> Json<serde_json::Value>{

    let mut lock=state.lock().expect("Error while fetching the data");
    lock.infer_one_building(&bldg_id);
    remove_dir_all("production_set");
    // take the predictions from the json file made and send them
    let data_file=fs::read_to_string("data.json").unwrap();
    drop(lock);
    Json(serde_json::from_str(&data_file).unwrap())
}
async fn send_metrics(State(state): State<Arc<Mutex<Controller>>>)-> Json<Metrics>{

    let lock=state.lock().expect("Error while fetching the data");
    // take the predictions from the json file made and send them
    let metrics=fs::read_to_string("metrics.json").unwrap();
    let cross_validation=fs::read_to_string("../../cross_validation/cross_validation.json").unwrap();
    // initiliaze the metrics object and prepare it to be sent
    let metrics_object=Metrics{eval_metrics: metrics, cross_val_metrics: cross_validation};
    drop(lock);
    Json(metrics_object)
}

async fn welcome()-> impl IntoResponse{
    let msg="Decomposer says hi".to_string();
    Response::new(msg)
}
