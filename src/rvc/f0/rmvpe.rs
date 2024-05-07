use ndarray::s;

use crate::rvc::errors::RvcInferError;

use super::F0Algorithm;


pub struct Rmvpe {
    session: ort::Session,

}

impl F0Algorithm for Rmvpe {
    fn new(session: ort::Session) -> Self {
        Rmvpe {
            session: session,
        }
    }

    fn pitch(&self, input: ndarray::ArrayView1<f32>,
        pitch_shift: i32,
        sample_frame_16k_size: usize) -> Result<ndarray::Array1<f32>, RvcInferError> {
        
        let f0_extractor_frame = 5120 * ((sample_frame_16k_size + 800 - 1) / 5120 + 1) - 160;
        let input = input.slice(s![input.len() - f0_extractor_frame..]);
        
        let (f0_coarse, f0) = {
            let rmvpe = self.session;

            
        };


        let input_len = input.len();
        let feats = input.into_shape((1, 1, input_len))?;

        let f0_output = f0.run(ort::inputs!["source" => feats]?)?;
        let pitch = f0_output["pitch"]
            .try_extract_tensor::<f32>()?
            .into_dimensionality::<ndarray::Ix1>()?;
        Ok(pitch.to_owned())
    }
}

