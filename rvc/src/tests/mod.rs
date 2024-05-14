use std::path::PathBuf;

use crate::RvcInfer;

mod hubert;
mod pitch;

fn init_ort() {
    std::env::set_var("PATH", format!("C:\\ProgramData\\obs-studio\\plugins\\obsrvc\\bin\\64bit;{}", std::env::var("PATH").unwrap_or_default()));
    ort::init_from("C:\\ProgramData\\obs-studio\\plugins\\obsrvc\\bin\\64bit\\onnxruntime.dll").commit().unwrap();
}

fn get_rvc() -> RvcInfer {
    let data_path = PathBuf::from("C:\\ProgramData\\obs-studio\\plugins\\obsrvc\\data\\rvcinfer");
    RvcInfer::new(data_path)
}
