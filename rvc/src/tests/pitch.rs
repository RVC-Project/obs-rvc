#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use mel_spec::mel::mel;
    use ndarray::{Array1, Array3};
    use ndarray_npy::read_npy;

    use crate::{tests::{get_rvc, init_ort}, RvcInfer};

    #[test]
    fn test_mel() {
        let mel = mel(16000.0f64, 1024, 128, Some(30.0), Some(8000.0), true, true);

        println!("{:?}", mel);
    }

    #[test]
    fn test_pitch_rmvpe() {
        init_ort();
        let mut rvc = get_rvc();
        
        rvc.load_f0(rvc_common::enums::PitchAlgorithm::Rmvpe).unwrap();


        let input: Array1<f32> = read_npy("D:\\obs-rvc\\rvc\\src\\tests\\input_wav2.npy").unwrap();
        let (pitch, pitchf) = rvc.pitch(input.view(), 13, 4800).unwrap();
        println!("pitch: {:?}", pitch);
        println!("pitchf: {:?}", pitchf);
    }
}
