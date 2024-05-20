#[cfg(test)]
mod tests {
    use ndarray::{s, Array1, Zip};
    use ndarray_npy::read_npy;

    use crate::rt_utils::{linear_interpolate_align_corners, rms};

    #[test]
    fn test_envelop_mixing_rms() {
        let input_wav: Array1<f32> = read_npy("D:\\obs-rvc\\obs-rvc\\src\\tests\\envelop_input_wav.npy").unwrap();
        let infer_wav: Array1<f32> = read_npy("D:\\obs-rvc\\obs-rvc\\src\\tests\\envelop_infer_wav.npy").unwrap();
        let zc = 480;
        let mix_rate = 0.8;
        let rms1_expected: Array1<f32> = read_npy("D:\\obs-rvc\\obs-rvc\\src\\tests\\envelop_rms1.npy").unwrap();
        let rms2_expected: Array1<f32> = read_npy("D:\\obs-rvc\\obs-rvc\\src\\tests\\envelop_rms2.npy").unwrap();
        
        let output_len = infer_wav.len();
        let rms1_actual = rms(input_wav.slice(s![..output_len]), 4*zc, zc);
        let rms2_actual = rms(infer_wav.view(), 4*zc, zc);
        let rms1_actual = linear_interpolate_align_corners(rms1_actual.view(), output_len + 1);
        let rms2_actual = linear_interpolate_align_corners(rms2_actual.view(), output_len + 1)
            .mapv(|x| f32::max(x, 1e-3));

        approx::assert_abs_diff_eq!(rms1_actual.slice(s![..output_len]), rms1_expected, epsilon = 1e-6);
        approx::assert_abs_diff_eq!(rms2_actual.slice(s![..output_len]), rms2_expected, epsilon = 1e-6);
        
        let infer_wav2_expected: Array1<f32> = read_npy("D:\\obs-rvc\\obs-rvc\\src\\tests\\envelop_infer_wav2.npy").unwrap();
        
        let mix_power = 1.0f64 - mix_rate;
        let infer_wav2_actual = Zip::from(&infer_wav).and(rms1_actual.slice(s![..output_len])).and(rms2_actual.slice(s![..output_len]))
            .map_collect(|out, rms1, rms2| {
                *out * (*rms1 / *rms2).powf(mix_power as f32)
            });

        approx::assert_abs_diff_eq!(infer_wav2_actual, infer_wav2_expected, epsilon = 1e-6);
    }
}
