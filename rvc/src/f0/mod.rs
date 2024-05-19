use ndarray::Array1;

use self::rmvpe::Rmvpe;

pub mod rmvpe;
 
pub fn get_f0_post(f0: ndarray::Array1<f32>, f0_mel_min: f32, f0_mel_max: f32) -> (Array1<i32>, Array1<f32>) {
    let f0_coarse = f0.mapv(|x| (x / 700.0 + 1.).ln() * 1127.)
        .mapv(|x| if x <= 0. { x } else { (x - f0_mel_min) * 254. / (f0_mel_max - f0_mel_min) + 1. })
        .mapv(|x| x.clamp(1., 255.).round() as i32);
    (f0_coarse, f0)
}

pub enum F0Algorithm {
    Rmvpe(Rmvpe),
}
