use ndarray::s;
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

