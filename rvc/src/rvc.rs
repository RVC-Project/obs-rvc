use std::path::{Path, PathBuf};

use ndarray::{s, Axis};
use ndarray_rand::{rand_distr::Normal, RandomExt};
use ort::Session;
use crate::{f0::F0Algorithm, ndarray_ext::CopyWithin};

use super::{
    f0::{get_f0_post, rmvpe::Rmvpe},
    models::{load_contentvec_from_file, load_f0_from_file, load_model_from_file},
};

use rvc_common::{
    enums::{PitchAlgorithm, RvcModelVersion},
    errors::RvcInferError,
};

pub struct RvcInfer {
    data_path: PathBuf,
    session: Option<Session>,
    contentvec_session: Option<Session>,
    f0_algorithm: Option<F0Algorithm>,
    f0_mel_min: f32,
    f0_mel_max: f32,

    cache_pitchf: ndarray::Array1<f32>,
}

impl RvcInfer {
    pub fn new(data_path: PathBuf) -> Self {
        const F0_MIN: f32 = 50.0;
        const F0_MAX: f32 = 500.0f32;
        let f0_mel_min = (F0_MIN / 700.0 + 1.).ln() * 1127.;
        let f0_mel_max = (F0_MAX / 700.0 + 1.).ln() * 1127.;
        RvcInfer {
            data_path,
            session: None,
            contentvec_session: None,
            f0_algorithm: None,
            f0_mel_min,
            f0_mel_max,
            cache_pitchf: ndarray::Array1::zeros(1024),
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
            PitchAlgorithm::Rmvpe => {
                let f0_session = load_f0_from_file(
                    self.data_path.join("f0"),
                    self.data_path.join("cache"),
                    pitch_algorithm,
                )?;
                self.f0_algorithm =
                    Some(F0Algorithm::Rmvpe(Rmvpe::new(f0_session)));
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

    pub fn extract_feature(&self, input: ndarray::ArrayView1<f32>) -> Result<ndarray::Array3<f32>, RvcInferError> {
        let raw_hubert = self.hubert(input)?;
        let extended_hubert_shape = {
            let raw_h = raw_hubert.shape();
            [raw_h[0], raw_h[1], raw_h[2] * 2 + 1]
        };
        let max_k = raw_hubert.len_of(Axis(2)) - 1;
        Ok(ndarray::Array3::from_shape_fn(extended_hubert_shape, 
            |(i, j, k)| raw_hubert[[i, j, usize::min(k / 2, max_k)]]
        ).permuted_axes([0, 2, 1]))
    }

    pub fn pitch(
        &mut self,
        input: ndarray::ArrayView1<f32>,
        pitch_shift: i32,
        sample_frame_16k_size: usize,
    ) -> Result<ndarray::Array1<f32>, RvcInferError> {
        // return pitch, pitchf

        let f0 = match &mut self.f0_algorithm {
            Some(F0Algorithm::Rmvpe(rmvpe)) => {
                let uppower = 2.0f32.powi(pitch_shift / 12);
                let f0 = rmvpe.pitch(input, sample_frame_16k_size, 0.03)? * uppower;
                f0
            }
            _ => unreachable!(),
        };

        Ok(f0)

        // Ok(get_f0_post(f0, self.f0_mel_min, self.f0_mel_max))
    }

    pub fn infer(
        &mut self,
        input: ndarray::ArrayView1<f32>,
        sample_frame_16k_size: usize,
        pitch_shift: Option<i32>,
        skip_head: u32,
        return_length: u32,
    ) -> Result<ndarray::Array1<f32>, RvcInferError> {
        if self.session.is_none() {
            return Err(RvcInferError::ModelNotLoaded);
        }

        let start_time = std::time::Instant::now();
        
        let skip_head = skip_head as usize;
        let return_length = return_length as usize;

        // let hubert_output = self.hubert(input)?;
        let hubert_output = self.extract_feature(input)?;

        let hubert_length = usize::min(input.len() / 160, hubert_output.len_of(Axis(1)));
        // let hubert_output = hubert_output.slice(s![.., ..hubert_length, ..]);
        let hubert_output = hubert_output.slice(s![.., skip_head..skip_head + return_length, ..]);

        let hubert_time = start_time.elapsed();

        // TODO: index search
        

        // if f0
        let pitch_shift = pitch_shift.unwrap_or(0);
        let (pitch, pitchf) = {
            let pitchf = self.pitch(input, pitch_shift, sample_frame_16k_size)?;

            let pitch_len = pitchf.len();
            let shift = sample_frame_16k_size / 160;
            
            self.cache_pitchf.copy_within(shift.., 0);

            let cache_pitch_start = self.cache_pitchf.len() + 4 - pitch_len;

            self.cache_pitchf.slice_mut(s![cache_pitch_start..]).assign(&pitchf.slice(s![3..pitch_len - 1]));

            let cached_range_start = self.cache_pitchf.len() - hubert_length + skip_head;
            let cached_range_end = cached_range_start + return_length;

            let result_pitchf = self.cache_pitchf.slice(s![cached_range_start..cached_range_end]).to_owned();
            let (pitch, pitchf) = get_f0_post(result_pitchf, self.f0_mel_min, self.f0_mel_max);
            (pitch.insert_axis(Axis(0)), pitchf.insert_axis(Axis(0)))
        };

        let pitch_time = start_time.elapsed() - hubert_time;

        // let ds = 0;
        // let ds = ndarray::Array1::from_elem(1, ds as i32);
        // let rnd = ndarray::Array3::random((1, 192, hubert_length), Normal::from_mean_cv(0.0f32, 1.0f32).unwrap());

        // let skip_head = ndarray::Array1::from_elem(1, skip_head as i64);
        // let return_length = ndarray::Array1::from_elem(1, return_length as i64);

        let output = {
            let session = self.session.as_ref().unwrap();
            session.run(ort::inputs![
                "phone" => hubert_output, 
                // "phone_lengths" => hubert_length_arr, 
                "pitch" => pitch,
                "pitchf" => pitchf,
                // "ds" => ds,
                // "rnd" => rnd
                // "skip_head" => skip_head,
                // "max_len" => return_length,
            ]?)?
        };

        let output_tensor = output["audio"]
            .try_extract_tensor::<f32>()?
            .into_dimensionality::<ndarray::Ix1>()?;

        let out = output_tensor
            // .remove_axis(Axis(0))
            // .remove_axis(Axis(0))
            .to_owned();
            // .mapv(|x| x * 32767.0f32);

        eprintln!("hubert: {:?}, pitch: {:?}, inference: {:?}", hubert_time, pitch_time, start_time.elapsed() - pitch_time - hubert_time);

        Ok(out)
    }
}
