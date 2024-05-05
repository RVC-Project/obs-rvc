use std::path::{self, Path};

use ort::{
    CPUExecutionProvider, CUDAExecutionProvider, GraphOptimizationLevel, Session,
    TensorRTExecutionProvider,
};

pub struct RvcInfer {
    session: Option<Session>,
    contentvec_session: Option<Session>,
}

fn load_model_from_file<P>(model_path: P) -> Result<Session, ort::Error>   
where
    P: AsRef<Path> {
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .with_execution_providers([
            TensorRTExecutionProvider::default()
                .with_timing_cache(true)
                .with_engine_cache(true)
                .with_fp16(true)
                .build(),
            CUDAExecutionProvider::default().build(),
            CPUExecutionProvider::default().build(),
        ])?
        .commit_from_file(model_path)
}

impl RvcInfer {
    pub fn new() -> Self {
        RvcInfer { 
            session: None,
            contentvec_session: None,
        }
    }

    pub fn load_model<P>(&mut self, model_path: P) -> Result<(), ort::Error>
    where
        P: AsRef<Path>,
    {
        self.session = Some(load_model_from_file(model_path)?);
        Ok(())
    }

    pub fn unload_model(&mut self) {
        self.session = None;
    }

    pub fn infer(&self, input: ndarray::ArrayView1<f32>) -> Result<ndarray::Array1<f32>, ort::Error> {
        let feats = input.into_shape((1, input.len()));
        
        Ok(input.to_owned())
    }
}
