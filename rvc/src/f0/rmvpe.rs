use std::{collections::HashMap, mem::MaybeUninit};

use mel_spec::mel;
use ndarray::{azip, s, Axis, CowArray, Zip};
use ndarray_stats::QuantileExt;
use num_complex::{Complex, Complex32, Complex64};
use rustfft::{num_traits::Zero, FftPlanner};

use rvc_common::errors::RvcInferError;


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
    ndarray::Array1::from_shape_fn(window_length, |i| {
        0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (window_length as f64 - 1.0)).cos() as f32)
    })
}

fn get_hann_window_periodic(window_length: usize) -> ndarray::Array1<f32> {
    ndarray::Array1::from_shape_fn(window_length, |i| {
        0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / ((window_length + 1) as f64 - 1.0)).cos() as f32)
    })
}

fn pad_constant<N>(input_data: ndarray::Array1<N>, pad_amount: usize, constant_value: N) -> ndarray::Array1<N>
where N: Copy {
    let shape: ndarray::prelude::Dim<[usize; 1]> = input_data.raw_dim();
    let mut padded = ndarray::Array1::from_elem(shape[0] + 2 * pad_amount, constant_value);
    padded.slice_mut(s![pad_amount..shape[0]+pad_amount]).assign(&input_data);
    padded
}

fn pad_reflect<N>(arr: ndarray::ArrayView1<N>, pad: usize) -> ndarray::Array1<N>
where N: Copy
{
    let len = arr.len();
    let mut padded = ndarray::Array1::<N>::uninit(len + 2 * pad);
    let arr_uninit = arr.mapv(MaybeUninit::new);

    // Copy the original array to the center of the padded array
    padded.slice_mut(s![pad..len + pad]).assign(&arr_uninit);

    // Pad left
    for i in 0..pad {
        padded.slice_mut(s![pad - i - 1]).assign(&arr_uninit.slice(s![i+1]));
    }

    // Pad right
    for i in 0..pad {
        padded.slice_mut(s![len + pad + i]).assign(&arr_uninit.slice(s![len - i - 2]));
    }

    unsafe { padded.assume_init() }
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

fn stft(signal: ndarray::ArrayView1<f32>, fft_size: usize, hop_length: usize, window: ndarray::ArrayView1<f32>, center: bool) -> ndarray::Array2<f32> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    let L = signal.len();
    let N = fft_size / 2 + 1; // onesided = true
    let T = if true { 1 + L / hop_length } else { 1 + (L - fft_size) / hop_length };

    let win_length = window.len();

    let signal = match center {
        true => pad_reflect(signal, fft_size / 2).to_shared(),
        false => signal.to_shared(),
    };

    let window = if win_length < fft_size {
        // this should not happen yet... might have problem
        let left = (fft_size - win_length) / 2;
        window.slice(s![left..left + win_length]).to_shared()
    } else {
        window.to_shared()
    };
    
    let input = ndarray::Array2::from_shape_fn(
        (T, fft_size), 
        |(i, j)| signal[i * hop_length + j]
    ) * window;

    let mut input = input.mapv(|x| Complex::new(x, 0.0));

    for mut row in input.rows_mut() {
        fft.process(row.as_slice_mut().unwrap());
    }

    input.swap_axes(0, 1);
    input.slice(s![..N, ..]).mapv(|x| x.norm_sqr().sqrt())
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
            self.hann_window_cache.insert(keyshift, get_hann_window_periodic(win_length_new));
        }

        let hann_window = self.hann_window_cache[&keyshift].view();

        let mut magnitude = stft(
            input,
            fft_size_new,
            hop_length_new,
            hann_window,
            center,
        );

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

        let mel_output = self.mel_basis.dot(&magnitude);
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
        let mel = mel.insert_axis(Axis(0));
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
        let signal1 = ndarray::Array1::linspace(0.0f32, 1.0f32, 500);
        let fft_size1 = 16;
        let hop_length1 = 160;
        let window1 = get_hann_window_periodic(16);
        let center1 = true;

        // TODO: this expected output is from torch. It has a considerable difference from the output of this stft function
        let expected_output1 = ndarray::arr2(&[
            [3.7801e-02, 2.5651e+00, 5.1303e+00, 7.6954e+00],
            [5.7373e-03, 1.2829e+00, 2.5653e+00, 3.8478e+00],
            [1.4787e-02, 6.7956e-03, 6.7958e-03, 6.7957e-03],
            [3.2463e-03, 1.6874e-03, 1.6874e-03, 1.6875e-03],
            [2.3478e-03, 6.6042e-04, 6.6042e-04, 6.6054e-04],
            [1.4494e-03, 3.1195e-04, 3.1202e-04, 3.1184e-04],
            [1.2455e-03, 1.5500e-04, 1.5485e-04, 1.5491e-04],
            [1.0416e-03, 6.5722e-05, 6.5798e-05, 6.5790e-05],
            [1.0417e-03, 0.0000e+00, 0.0000e+00, 2.3842e-07]
        ]);
        let output1 = stft(signal1.view(), fft_size1, hop_length1, window1.view(), center1);
        // assert_eq!(output1.shape(), expected_output1.shape());
        assert_eq!(output1, expected_output1);
    }

    #[test]
    fn test_pad_reflect() {
        // Test case 1
        let input_data1 = ndarray::arr1(&[1.0, 2.0, 3.0]);
        let pad_amount1 = 2;
        let expected_output1 = ndarray::arr1(&[3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]);
        let output1 = pad_reflect(input_data1.view(), pad_amount1);
        assert_eq!(output1, expected_output1);

        // Test case 2
        let input_data2 = ndarray::arr1(&[4.0, 5.0]);
        let pad_amount2 = 1;
        let expected_output2 = ndarray::arr1(&[5.0, 4.0, 5.0, 4.0]);
        let output2 = pad_reflect(input_data2.view(), pad_amount2);
        assert_eq!(output2, expected_output2);
    }

    #[test]
    fn test_pad_constant() {
        // Test case 1
        let input_data1 = ndarray::arr1(&[1.0, 2.0, 3.0]);
        let pad_amount1 = 2;
        let constant_value1 = 0.0;
        let expected_output1 = ndarray::arr1(&[0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
        let output1 = pad_constant(input_data1, pad_amount1, constant_value1);
        assert_eq!(output1, expected_output1);
        // Test case 2
        let input_data2 = ndarray::arr1(&[4.0, 5.0]);
        let pad_amount2 = 1;
        let constant_value2 = 2.0;
        let expected_output2 = ndarray::arr1(&[2.0, 4.0, 5.0, 2.0]);
        let output2 = pad_constant(input_data2, pad_amount2, constant_value2);
        assert_eq!(output2, expected_output2);
    }
}

