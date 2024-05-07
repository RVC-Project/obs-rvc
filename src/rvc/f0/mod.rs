use super::errors::RvcInferError;

pub mod rmvpe;
 
pub trait F0Algorithm {
    fn new(session: ort::Session) -> Self;

    fn pitch(&self, input: ndarray::ArrayView1<f32>,
        pitch_shift: i32,
        sample_frame_16k_size: usize) -> Result<ndarray::Array1<f32>, RvcInferError>;
}
