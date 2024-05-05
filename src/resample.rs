use dasp::{interpolate::sinc::Sinc, ring_buffer, signal, Sample, Signal};

pub fn resample_16k(input: &[f64], input_sample_rate: f64) -> Vec<f64> {
    let signal = signal::from_iter(input.iter().cloned());
    let ring_buffer = ring_buffer::Fixed::from([[0.0]; 100]);
    let sinc = Sinc::new(ring_buffer);
    let input_signal = dasp::signal::from_iter(input.iter().cloned());

    let new_signal = signal.from_hz_to_hz(sinc, input_sample_rate, 16000_f64);
    new_signal.until_exhausted().collect()
}
