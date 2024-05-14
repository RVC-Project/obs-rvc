#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ndarray::{Array1, Array3};
    use ndarray_npy::read_npy;

    use crate::rt_utils::get_sola_offset;

    #[test]
    fn test_sola() {
        let input: Array1<f32> = read_npy("D:\\obs-rvc\\obs-rvc\\src\\tests\\infer_wav.npy").unwrap();
        let sola_buffer: Array1<f32> = read_npy("D:\\obs-rvc\\obs-rvc\\src\\tests\\sola_buffer.npy").unwrap();
        let output = get_sola_offset(input.view(), sola_buffer.view(), 1920, 480).unwrap();
        assert_eq!(output, 321);
    }
}
