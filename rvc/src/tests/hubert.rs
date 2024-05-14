#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ndarray::{Array1, Array3};
    use ndarray_npy::read_npy;

    use crate::{tests::{get_rvc, init_ort}, RvcInfer};

    #[test]
    fn test_hubert_v2() {
        init_ort();
        let mut rvc = get_rvc();
        rvc.load_contentvec(rvc_common::enums::RvcModelVersion::V2).unwrap();
        let input: Array1<f32> = read_npy("D:\\obs-rvc\\rvc\\src\\tests\\input_wav.npy").unwrap();
        let feats: Array3<f32> = read_npy("D:\\obs-rvc\\rvc\\src\\tests\\feats.npy").unwrap();
        let output = rvc.extract_feature(input.view()).unwrap();
        approx::assert_abs_diff_eq!(output, feats, epsilon = 2e-3);
    }
}
