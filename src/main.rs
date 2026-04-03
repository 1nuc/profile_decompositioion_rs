use std::sync::Arc;

use axum::{Json, Router, extract::State, http::Response, response::IntoResponse, routing::get};
use decomposer_engine::{Actions, EagerActions, data_engine::*, dl::controller::{self, Controller}, xgb};

#[tokio::main]
async fn main(){
    let shared_state = Arc::new(Controller::default());
    serve(shared_state).await;
    // controller::run_training();
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

async fn serve(shared_state: Arc<Controller>){
    let app=Router::new()
        .route("/", get(send_data))
        .route("/buildings", get(send_bldg)).with_state(shared_state);
    let listner=tokio::net::TcpListener::bind("localhost:8000").await.unwrap();
    axum::serve(listner, app).await.unwrap();
}

// Return the available buildings in the data
async fn send_bldg(State(state): State<Arc<Controller>>)-> Json<Vec<String>>{
    let buildings=state.return_nrel_buildings();
    Json(buildings)
}

async fn send_data()-> impl IntoResponse{
    let msg="Decomposer says hi".to_string();
    Response::new(msg)
}
