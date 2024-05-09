use ndarray::{s, Array1, ArrayView1, ArrayViewMut1, Axis, Zip};
use ndarray_conv::ConvFFTExt as _;
use obs_wrapper::media::{AudioData, AudioDataContext};

pub fn downmix_to_mono(audio: &mut AudioDataContext, channels: usize) -> std::io::Result<&mut [f32]> {
    let main_channel = audio.get_channel_as_mut_slice(0).ok_or_else(|| std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        "No main channel found.",
    ))?;

    let interpolation_factor = 1.0 / channels as f32;
    main_channel.iter_mut().for_each(|sample| *sample *= interpolation_factor);
    for channel in 1..channels {
        let buffer = audio
            .get_channel_as_mut_slice(channel)
            .ok_or_else(|| std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Channel count said there was a buffer here.",
            ))?;

        for (base_stream, additional_stream) in main_channel.iter_mut().zip(buffer.iter()) {
            *base_stream = *base_stream + *additional_stream * interpolation_factor;
        }
    }

    Ok(main_channel)
}

pub fn upmix_audio_data_context(audio: &mut AudioDataContext, channels: usize) -> std::io::Result<()> {
    let main_channel = audio.get_channel_as_mut_slice(0).ok_or_else(|| std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        "No main channel found.",
    ))?;

    for channel in 1..channels {
        let buffer = audio
            .get_channel_as_mut_slice(channel)
            .ok_or_else(|| std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Channel count said there was a buffer here.",
            ))?;

        buffer.copy_from_slice(&main_channel)
    }

    Ok(())
}

pub fn upmix_audio_data(audio: &mut AudioData) -> std::io::Result<()> {
    let audio_data = &mut audio.data;
    let (first_channel_data, remaining_channels_data) = audio_data.split_at_mut(1);
    let first_channel_data = &mut first_channel_data[0];
    for channel_data in remaining_channels_data {
        channel_data.copy_from_slice(first_channel_data)
    }

    Ok(())
}

pub fn get_sola_offset(input_buffer: ndarray::ArrayView1<f32>, sola_buffer: ndarray::ArrayView1<f32>, 
    buffer_frame_size: usize, search_frame_size: usize) -> Result<usize, Box<dyn std::error::Error>> {
    let conv_input_size = buffer_frame_size + search_frame_size;
    let conv_input = input_buffer
        .slice(s![..conv_input_size])
        .into_shape((1, 1, conv_input_size))?;
    let sola_buffer_view = sola_buffer
        .into_shape((1, 1, buffer_frame_size))?;

    let cor_nom = conv_input.conv_fft(
        &sola_buffer_view, ndarray_conv::ConvMode::Valid, 
        ndarray_conv::PaddingMode::Zeros
    )?;

    let cor_den_filler = ndarray::Array3::<f32>::ones((1, 1, buffer_frame_size));
    let conv_input_squared = conv_input.mapv(|x| x * x);
    let mut cor_den = conv_input_squared.conv_fft(
        &cor_den_filler, ndarray_conv::ConvMode::Valid, 
        ndarray_conv::PaddingMode::Zeros
    )?;
    cor_den.mapv_inplace(|x| (x + 1e-8).sqrt());

    let cor_nom_len = cor_nom.len();
    let cor_den_len = cor_den.len();
    
    let cor_nom_1d = cor_nom.into_shape(cor_nom_len)?;
    let cor_den_1d = cor_den.into_shape(cor_den_len)?;
    let cor = cor_nom_1d / cor_den_1d;
    let (idx_max, _val_max) =
        cor.indexed_iter()
            .fold((0, cor[0]), |(idx_max, val_max), (idx, val)| {
                if &val_max > val {
                    (idx_max, val_max)
                } else {
                    (idx, *val)
                }
            });
    Ok(idx_max)
}


fn rms(y: ArrayView1<f32>, frame_length: usize, hop_length: usize) -> Array1<f32> {
    let padding = frame_length / 2;
    let y_padded = ndarray::concatenate![Axis(0), Array1::zeros(padding), y, Array1::zeros(padding)].mapv(|x| x.powi(2));
    let y_mean = y_padded
        .windows((frame_length,))
        .into_iter()
        .step_by(hop_length)
        .map(|f| f.mean().unwrap().sqrt());
    y_mean.collect()
}

fn linear_interpolate_align_corners(input: ArrayView1<f32>, size: usize) -> Array1<f32> {
    let mut output = Array1::zeros(size);
    let step = (input.len() - 1) as f32 / (size - 1) as f32;

    Zip::indexed(output.view_mut()).for_each(|idx, val| {
        let idx = idx as f32 * step;
        let idx_floor = usize::clamp(idx.floor() as usize, 0, input.len() - 1);
        let idx_ceil = usize::clamp(idx.ceil() as usize, 0, input.len() - 1);
        let idx_frac = idx - idx_floor as f32;
        *val = input[idx_floor] * (1.0 - idx_frac) + input[idx_ceil] * idx_frac;
    });

    output
}

pub fn envelop_mixing(input: ArrayView1<f32>, output: ArrayViewMut1<f32>, sample_rate: usize, mix_rate: f64) {
    let zc = sample_rate / 100;
    let output_len = output.len();
    let rms1 = rms(input.slice(s![..output_len]), 4*zc, zc);
    let rms2 = rms(output.view(), 4*zc, zc);
    let rms1 = linear_interpolate_align_corners(rms1.view(), output_len + 1);
    let rms2 = linear_interpolate_align_corners(rms2.view(), output_len + 1)
        .mapv(|x| f32::max(x, 1e-3));
    let mix_power = 1.0f64 - mix_rate;
    Zip::from(output).and(rms1.slice(s![..output_len])).and(rms2.slice(s![..output_len]))
        .for_each(|out, rms1, rms2| {
            *out = *out * (rms1 / rms2).powf(mix_power as f32);
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms() {
        let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let frame_length = 4;
        let hop_length = 2;
        let expected_rms = Array1::from(vec![1.118034, 2.738613, 4.6368093, 6.595453, 8.573215, 6.726812]);

        let rms_values = rms(y.view(), frame_length, hop_length);

        assert_eq!(rms_values, expected_rms);
    }

    #[test]
    fn test_linear_interpolate_align_corners() {
        let input = Array1::from(vec![0.2353, 0.9068, 0.7870, 0.5878, 0.0097, 0.7160, 0.5812, 0.8901, 0.8822, 0.8547]);
        let expected_output_3 = Array1::from(vec![0.2353, 0.36285, 0.8547]);
        let expected_output_15 = Array1::from(vec![0.2353, 0.66697854, 0.8725714, 0.79555714, 0.6731714, 0.4639215, 0.09228568, 0.36285, 0.6967429, 0.6100857, 0.7135856, 0.8895357, 0.8844571, 0.8723786, 0.8547]);
        let output_3 = linear_interpolate_align_corners(input.view(), 3);
        let output_15 = linear_interpolate_align_corners(input.view(), 15);
        assert_eq!(output_3, expected_output_3);
        assert_eq!(output_15, expected_output_15);
    }
}
