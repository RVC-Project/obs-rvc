use super::f0::rmvpe::Rmvpe;
use std::cmp::PartialEq;

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

pub enum PitchAlgorithm {
    Rmvpe(Option<Rmvpe>),
}

impl PartialEq for PitchAlgorithm {
    fn eq(&self, other: &Self) -> bool {
        // Compare the enum variants only, ignoring the internal content
        match (self, other) {
            (PitchAlgorithm::Rmvpe(_), PitchAlgorithm::Rmvpe(_)) => true,
        }
    }
}

impl PitchAlgorithm {
    pub fn clone_lossy(&self) -> Self {
        match self {
            PitchAlgorithm::Rmvpe(_) => PitchAlgorithm::Rmvpe(None),
        }
    }
}
