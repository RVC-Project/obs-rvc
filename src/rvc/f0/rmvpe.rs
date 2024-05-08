use std::{collections::HashMap, mem::MaybeUninit};

use mel_spec::{mel, stft};
use ndarray::{azip, s, Axis, CowArray, Zip};
use ndarray_stats::QuantileExt;
use num_complex::{Complex, Complex32, Complex64};
use rustfft::{num_traits::Zero, FftPlanner};

use crate::rvc::errors::RvcInferError;


pub struct Rmvpe {
    session: ort::Session,
    mel_extractor: MelSpectrogram,
    cents_mapping: ndarray::Array1<f32>,
}

struct MelSpectrogram {
    mel_basis: ndarray::Array2<f32>,
    fft_size: usize,
    win_length: usize,
    hop_length: usize,
    hann_window_cache: HashMap<i32, ndarray::Array1<f32>>,
    clamp: f32,
}

fn get_hann_window(window_length: usize) -> ndarray::Array1<f32> {
    ndarray::Array1::linspace(0.0, window_length as f32 - 1.0, window_length).mapv(|n| {
        0.5 * (1.0 - (2.0 * std::f32::consts::PI * n / (window_length as f32 - 1.0)).cos())
    })
}

// fn stft(
//     input: ndarray::ArrayView1<f32>,
//     fft_size: usize,
//     hop_length: usize,
//     window: ndarray::ArrayView1<f32>,
//     center: bool,
// ) -> ndarray::Array2<Complex32> {
//     let mut planner = FftPlanner::new();
//     let fft = planner.plan_fft_forward(fft_size);

//     let mut output = ndarray::Array2::uninit(
//         ((input.len() - fft_size) / hop_length + 1, fft_size)
//     );

//     let window_sum = window.sum();

//     for (i, mut row) in output.outer_iter_mut().enumerate() {
//         let start = if center {
//             i * hop_length - fft_size / 2
//         } else {
//             i * hop_length
//         };

//         let mut frame = ndarray::Array1::uninit(fft_size);
//         Zip::from(frame.slice_mut(s![..input.len() - start]))
//             .and(input.slice(s![start..]))
//             .and_broadcast(window)
//             .for_each(|frame, &input, &window| {
//                 *frame = MaybeUninit::new(Complex32::new(input * window / window_sum, 0.0f32));
//             });
//         let mut frame = unsafe { frame.assume_init() };
//         fft.process(frame.as_slice_mut().unwrap());
//         row.assign(&frame.mapv(MaybeUninit::new));
//     }

//     unsafe {output.assume_init()}
// }


fn pad_reflect(mut input_data: ndarray::Array1<f32>, pad_amount: usize) -> ndarray::Array1<f32> {
    let shape: ndarray::prelude::Dim<[usize; 1]> = input_data.raw_dim();
    let mut padded = ndarray::Array1::zeros(shape[0] + 2 * pad_amount);
    padded.slice_mut(s![pad_amount..shape[0]+pad_amount]).assign(&input_data);
    for i in 0..pad_amount {
        padded[i] = input_data[pad_amount-i-1];
        padded[shape[0]+pad_amount+i] = input_data[shape[0]-i-1];
    }
    padded
}

fn unfold_permute_forward_transform(input_data: ndarray::ArrayView1<f32>, filter_length: usize, hop_length: usize) -> ndarray::Array2<f32> {
    let shape = input_data.raw_dim();
    let unfolded_dim = (shape[1] - filter_length) / hop_length + 1;
    let mut unfolded = ndarray::Array2::zeros((unfolded_dim, filter_length));
    for i in 0..unfolded_dim {
        unfolded.slice_mut(s![i, ..]).assign(&input_data.slice(s![i*hop_length..i*hop_length+filter_length]));
    }
    unfolded
}

// fn stft(input_data: ndarray::Array1<f32>, fft_size: usize, hop_length: usize) -> (ndarray::Array2<f32>, Array2<f32>) {
//     let shape = input_data.raw_dim();
//     let unfolded_dim = (shape[1] - filter_length) / hop_length + 1;
//     let mut forward_transform = ndarray::Array3::zeros((shape[0], unfolded_dim, filter_length));
//     for i in 0..unfolded_dim {
//         forward_transform.slice_mut(s![.., i, ..]).assign(&input_data.slice(s![.., i*hop_length..i*hop_length+filter_length]));
//     }
//     let mut planner = FftPlanner::new();
//     let fft = planner.plan_fft_forward(filter_length);
//     for i in 0..shape[0] {
//         for j in 0..unfolded_dim {
//             let mut buffer: Vec<Complex<f32>> = forward_transform.slice(s![i, j, ..]).to_vec().into_iter().map(|x| Complex::new(x, 0.0)).collect();
//             fft.process(&mut buffer);
//             forward_transform.slice_mut(s![i, j, ..]).assign(&Array1::from(buffer).mapv(|x| x.norm()));
//         }
//     }
//     let cutoff = (filter_length / 2) + 1;
//     let real_part = forward_transform.slice(s![.., ..cutoff, ..]).to_owned();
//     let imag_part = forward_transform.slice(s![.., cutoff.., ..]).to_owned();
//     (real_part, imag_part)
// }


fn stft(signal: ndarray::ArrayView1<f32>, fft_size: usize, hop_length: usize, window: ndarray::ArrayView1<f32>) -> ndarray::Array2<Complex<f32>> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    let L = signal.len();
    let N = fft_size / 2 + 1;
    let T = if true { 1 + L / hop_length } else { 1 + (L - fft_size) / hop_length };
    let pad_len = fft_size / 2;

    let signal = pad_reflect(signal.to_owned(), pad_len);
    let forward_transform = unfold_permute_forward_transform(signal.view(), fft_size, hop_length);

    let mut output = ndarray::Array2::uninit((N, T));

    let window_sum = window.sum();
    for (t, mut col) in output.outer_iter_mut().enumerate() {
        let mut frame = ndarray::Array1::from_elem(fft_size, Complex32::zero());
        let start = if true { t * hop_length - fft_size / 2 } else { t * hop_length };
        Zip::from(frame.slice_mut(s![..L - start]))
            .and(signal.slice(s![start..]))
            .and_broadcast(window)
            .for_each(|frame, &signal, &window| {
                *frame = Complex32::new(signal * window / window_sum, 0.0f32);
            });
        let mut frame = CowArray::from(frame);
        fft.process(frame.as_slice_mut().unwrap());
        col.assign(&frame.mapv(MaybeUninit::new));
    }

    unsafe { output.assume_init() }
}

fn to_local_average_cents(salience: ndarray::ArrayView2<f32>, cents_mapping: ndarray::ArrayView1<f32>, threshold: f32) -> ndarray::Array1<f32> {
    let mut salience_padded = ndarray::Array2::zeros((salience.nrows(), salience.ncols() + 8));
    salience_padded.slice_mut(s![.., 4..-4]).assign(&salience);

    let starts = salience_padded.map_axis(Axis(1), |row| row.argmax().unwrap() as usize);

    let todo_salience = ndarray::Array2::from_shape_fn((salience.nrows(), 9), |(x, y)| salience[[x, starts[x]+y]]);
    let todo_cents_mapping = ndarray::Array2::from_shape_fn((salience.nrows(), 9), |(x, y)| cents_mapping[[starts[x]+y]]);

    let product_sum = todo_salience.clone() * todo_cents_mapping;
    let weight_sum = todo_salience.sum_axis(Axis(1));
    let mut devided = product_sum.sum_axis(Axis(1)) / weight_sum;
    let maxx = salience.map_axis(Axis(1), |row| *row.max().unwrap());
    azip!((a in &mut devided, b in &maxx) *a = if *b > threshold { *a } else { 0.0 });
    devided
}

impl MelSpectrogram {
    fn new(
        fft_size: usize,
        sample_rate: usize,
        n_mels: usize,
        win_length: usize,
        hop_length: usize,
        clamp: f32,
    ) -> Self {
        let mel_basis =
            mel::mel(sample_rate as f64, fft_size, n_mels, true, true).mapv(|x| x as f32);
        MelSpectrogram {
            mel_basis,
            fft_size,
            win_length,
            hop_length,
            hann_window_cache: HashMap::new(),
            clamp,
        }
    }

    fn mel_extract(
        &mut self,
        input: ndarray::ArrayView1<f32>,
        keyshift: Option<i32>,
        speed: Option<usize>,
        center: Option<bool>,
    ) -> ndarray::Array2<f32> {
        let keyshift = keyshift.unwrap_or(0);
        let speed = speed.unwrap_or(1);
        let center = center.unwrap_or(true);

        let factor = 2.0f64.powf(keyshift as f64 / 12.0);
        let fft_size_new = (self.fft_size as f64 * factor).round() as usize;
        let win_length_new = (self.win_length as f64 * factor).round() as usize;
        let hop_length_new = self.hop_length * speed;
        if !self.hann_window_cache.contains_key(&keyshift) {
            self.hann_window_cache.insert(keyshift, get_hann_window(win_length_new));
        }

        let hann_window = self.hann_window_cache[&keyshift].view();
        let mut magnitude = stft(
            input,
            fft_size_new,
            hop_length_new,
            hann_window,
            center,
        ).mapv(|x| x.norm_sqr());

        println!("{:?}", magnitude);

        if keyshift != 0 {
            let size = self.fft_size / 2 + 1;
            let resize = magnitude.len_of(Axis(1));
            if resize < size {
                let mut padded = ndarray::Array2::zeros((magnitude.len_of(Axis(0)), size));
                padded.slice_mut(s![.., ..resize]).assign(&magnitude);
                magnitude = padded;
            }

            let rhs = magnitude.slice(s![.., size..]).to_owned();
            magnitude.slice_mut(s![.., ..size])
                .scaled_add(self.win_length as f32 / win_length_new as f32, &rhs);
        }

        let mel_output = self.mel_basis.dot(&magnitude.t());
        mel_output.mapv(|x| x.max(self.clamp).ln())
    }

}

impl Rmvpe {
    
    pub fn new(session: ort::Session) -> Self {
        let cents_mapping = {
            let mut field = ndarray::Array1::zeros(360 + 2 * 4);
            field.indexed_iter_mut().for_each(|(i, x)| *x = (i as f32 - 4.) * 20. + 1997.3794084376191);
            field
        };

        Rmvpe {
            session: session,
            mel_extractor: MelSpectrogram::new(1024, 16000, 128, 1024, 160, 1e-5),
            cents_mapping,
        }
    }

    fn mel2hidden(&self, mel: ndarray::Array2<f32>) -> std::result::Result<ndarray::Array3<f32>, RvcInferError> {
        let n_frames = mel.len_of(Axis(1));
        let n_pad = 32 * ((n_frames - 1) / 32 + 1) - n_frames;
        let mut mel = mel;
        if n_pad > 0 {
            let mut padded = ndarray::Array2::zeros((mel.len_of(Axis(0)), n_pad));
            padded.slice_mut(s![.., ..n_frames]).assign(&mel);
            mel = padded.into();
        }

        let output = self.session.run(ort::inputs!["input" => mel]?)?;
        let output_hidden = output["output"]
            .try_extract_tensor::<f32>()?
            .into_dimensionality::<ndarray::Ix3>()?;

        Ok(output_hidden.slice(s![.., ..n_frames, ..]).to_owned())
    }

    fn decode(&self, hidden: ndarray::Array2<f32>, threshold: f32) -> Result<ndarray::Array1<f32>, RvcInferError> {
        let cents_pred = to_local_average_cents(hidden.view(), self.cents_mapping.view(), threshold);
        let mut f0 = cents_pred.mapv(|x| 10.0f32 * (2.0f32.powf(x / 1200.0)));
        f0.mapv_inplace(|x| if x == 10.0f32 { 0.0 } else { x });
        Ok(f0)
    }
    
    pub fn pitch(
        &mut self,
        input: ndarray::ArrayView1<f32>,
        sample_frame_16k_size: usize,
        threshold: f32
    ) -> Result<ndarray::Array1<f32>, RvcInferError> {
        let f0_extractor_frame = 5120 * ((sample_frame_16k_size + 800 - 1) / 5120 + 1) - 160;
        let input = input.slice(s![input.len() - f0_extractor_frame..]);
        let mel = self.mel_extractor.mel_extract(input, None, None, Some(true));
        let hidden = self.mel2hidden(mel)?.remove_axis(Axis(0));
        self.decode(hidden, threshold)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stft() {
        // Test case 1
        let signal1 = ndarray::Array1::linspace(0.0f32, 1.0f32, 500);
        let fft_size1 = 16;
        let hop_length1 = 160;
        let window1 = get_hann_window(16);
        let center1 = true;
        let expected_output1 = ndarray::arr2(&[
            [Complex::new(1.5, 0.0), Complex::new(0.0, 0.0)],
            [Complex::new(3.5, 0.0), Complex::new(0.0, 0.0)],
        ]);
        let output1 = stft(signal1.view(), fft_size1, hop_length1, window1.view(), center1);
        assert_eq!(output1, expected_output1);
    }
}
