use std::path::{Path, PathBuf};

use ndarray::{s, Axis};
use ort::Session;

use super::{
    enums::{PitchAlgorithm, RvcModelVersion},
    errors::RvcInferError,
    f0::{rmvpe::Rmvpe, F0Algorithm},
    models::{load_contentvec_from_file, load_f0_from_file, load_model_from_file},
};

pub struct RvcInfer {
    data_path: PathBuf,
    session: Option<Session>,
    contentvec_session: Option<Session>,
    f0_algorithm: Option<PitchAlgorithm>,
}

impl RvcInfer {
    pub fn new(data_path: PathBuf) -> Self {
        RvcInfer {
            data_path,
            session: None,
            contentvec_session: None,
            f0_algorithm: None,
        }
    }

    pub fn load_contentvec(&mut self, model_version: RvcModelVersion) -> Result<(), ort::Error> {
        self.contentvec_session = Some(load_contentvec_from_file(
            self.data_path.join("contentvec"),
            self.data_path.join("cache"),
            model_version.text_encoder_in_channels(),
            model_version.output_layers(),
        )?);
        Ok(())
    }

    pub fn load_model(&mut self, model_path: PathBuf) -> Result<(), ort::Error> {
        let cache_path = self.data_path.join("cache");
        self.session = Some(load_model_from_file(model_path, cache_path)?);
        Ok(())
    }

    pub fn load_f0(&mut self, pitch_algorithm: PitchAlgorithm) -> Result<(), ort::Error> {
        match pitch_algorithm {
            PitchAlgorithm::Rmvpe(Some(rmvpe)) => {
                self.f0_algorithm = Some(PitchAlgorithm::Rmvpe(Some(rmvpe)));
            }
            PitchAlgorithm::Rmvpe(None) => {
                let f0_session = Some(load_f0_from_file(
                    self.data_path.join("f0"),
                    self.data_path.join("cache"),
                    pitch_algorithm,
                )?);
                self.f0_algorithm =
                    Some(PitchAlgorithm::Rmvpe(Some(Rmvpe::new(f0_session.unwrap()))));
            }
        }
        Ok(())
    }

    pub fn unload_model(&mut self) {
        self.session = None;
    }

    pub fn hubert(
        &self,
        input: ndarray::ArrayView1<f32>,
    ) -> Result<ndarray::Array3<f32>, RvcInferError> {
        let contentvec = self
            .contentvec_session
            .as_ref()
            .ok_or(RvcInferError::ContentvecNotLoaded)?;
        let input_len = input.len();
        let feats = input.into_shape((1, 1, input_len))?;

        let hubert_output = contentvec.run(ort::inputs!["source" => feats]?)?;
        let hubert = hubert_output["embed"]
            .try_extract_tensor::<f32>()?
            .into_dimensionality::<ndarray::Ix3>()?;
        Ok(hubert.permuted_axes([0, 2, 1]).to_owned())
    }

    pub fn pitch(
        &self,
        input: ndarray::ArrayView1<f32>,
        pitch_shift: i32,
        sample_frame_16k_size: usize,
    ) -> Result<(), ()> {
        // return pitch, pitchf

        match self.f0_algorithm {
            Some(PitchAlgorithm::Rmvpe(Some(rmvpe))) => {
                rmvpe.pitch(input, pitch_shift, sample_frame_16k_size)
            }
            _ => Err(()),
        }
    }

    pub fn infer(
        &self,
        input: ndarray::ArrayView1<f32>,
        sample_frame_16k_size: usize,
        pitch_shift: Option<i32>,
    ) -> Result<ndarray::Array1<f32>, RvcInferError> {
        if self.session.is_none() {
            return Err(RvcInferError::ModelNotLoaded);
        }

        let hubert_output = self.hubert(input)?;
        let hubert_length = hubert_output.len_of(Axis(1));

        // TODO: index search

        // if f0
        let pitch_shift = pitch_shift.unwrap_or(0);
        let (pitch, pitchf) = self.pitch(input, pitch_shift, sample_frame_16k_size);

        Ok(ndarray::Array1::zeros(1))
    }
}
