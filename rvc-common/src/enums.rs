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

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum PitchAlgorithm {
    Rmvpe,
}



impl From<RvcModelVersion> for i64 {
    fn from(version: RvcModelVersion) -> Self {
        match version {
            RvcModelVersion::V1 => 1,
            RvcModelVersion::V2 => 2,
        }
    }
}

impl From<i64> for RvcModelVersion {
    fn from(val: i64) -> Self {
        match val {
            1 => RvcModelVersion::V1,
            2 => RvcModelVersion::V2,
            _ => RvcModelVersion::V2,
        }
    }
}

impl From<RvcModelVersion> for String {
    fn from(version: RvcModelVersion) -> Self {
        match version {
            RvcModelVersion::V1 => "v1".to_string(),
            RvcModelVersion::V2 => "v2".to_string(),
        }
    }
}

impl From<String> for RvcModelVersion {
    fn from(val: String) -> Self {
        Self::from(val.as_str())
    }
}

impl From<&str> for RvcModelVersion {
    fn from(val: &str) -> Self {
        match val {
            "v1" => RvcModelVersion::V1,
            "v2" => RvcModelVersion::V2,
            _ => RvcModelVersion::V2,
        }
    }
}

impl ToString for RvcModelVersion {
    fn to_string(&self) -> String {
        match self {
            RvcModelVersion::V1 => "v1".to_string(),
            RvcModelVersion::V2 => "v2".to_string(),
        }
    }
}

impl RvcModelVersion {
    pub(crate) fn is_valid(val: i64) -> bool {
        match val {
            1 | 2 => true,
            _ => false,
        }
    }
}


impl From<PitchAlgorithm> for i64 {
    fn from(algorithm: PitchAlgorithm) -> Self {
        match algorithm {
            PitchAlgorithm::Rmvpe => 1,
        }
    }
}

impl From<i64> for PitchAlgorithm {
    fn from(val: i64) -> Self {
        match val {
            1 => PitchAlgorithm::Rmvpe,
            _ => PitchAlgorithm::Rmvpe,
        }
    }
}

impl From<PitchAlgorithm> for String {
    fn from(algorithm: PitchAlgorithm) -> Self {
        match algorithm {
            PitchAlgorithm::Rmvpe => "rmvpe".to_string(),
        }
    }
}

impl From<String> for PitchAlgorithm {
    fn from(val: String) -> Self {
        Self::from(val.as_str())
    }
}

impl From<&str> for PitchAlgorithm {
    fn from(val: &str) -> Self {
        match val {
            "rmvpe" => PitchAlgorithm::Rmvpe,
            _ => PitchAlgorithm::Rmvpe,
        }
    }
}

impl ToString for PitchAlgorithm {
    fn to_string(&self) -> String {
        match self {
            PitchAlgorithm::Rmvpe => "rmvpe".to_string(),
        }
    }
}

impl PitchAlgorithm {
    pub(crate) fn is_valid(val: i64) -> bool {
        match val {
            1 => true,
            _ => false,
        }
    }
}
