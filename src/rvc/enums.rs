#[derive(PartialEq, Clone, Copy, Debug)]
pub enum RvcModelVersion {
    V1,
    V2,
}

impl RvcModelVersion {
    pub fn text_encoder_in_channels(&self) -> usize {
        match self {
            RvcModelVersion::V1 => 256,
            RvcModelVersion::V2 => 768,
        }
    }

    pub fn output_layers(&self) -> usize {
        match self {
            RvcModelVersion::V1 => 9,
            RvcModelVersion::V2 => 12,
        }
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum PitchAlgorithm {
    Rmvpe,
}
