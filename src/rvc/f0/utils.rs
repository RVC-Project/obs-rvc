pub fn mel(sample_rate: usize, n_fft: usize, n_mels: uisize, f_min: f64, f_max: f64, htk: bool) {

}

pub struct MelSpectrogram {
    sample_rate: usize,
    n_fft: usize,
    n_mel_channels: usize,
    f_min: f64,
    f_max: f64,
    htk: bool,
    mel_basis: Array2<f64>,
    clamp: f64,
    win_length: usize,
    hop_length: usize,
}
