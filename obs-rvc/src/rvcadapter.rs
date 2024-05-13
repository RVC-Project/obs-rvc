use std::{io::{BufReader, BufWriter}, os::windows::process::CommandExt, path::PathBuf, process::{Child, ChildStdin, ChildStdout}};

use rvc_common::{enums::{PitchAlgorithm, RvcModelVersion}, errors::RvcInferError};
use std::process::{Command, Stdio};
use std::io::{Read, Write};
use ndarray::Array1;

pub struct RvcInfer {
    subprocess: Child,
    input: BufWriter<ChildStdin>,
    output: BufReader<ChildStdout>,
}

#[derive(Debug)]
pub enum RvcAdapterError {
    RvcInferError(RvcInferError),
    IoError(std::io::Error),
}

impl From<RvcInferError> for RvcAdapterError {
    fn from(err: RvcInferError) -> Self {
        RvcAdapterError::RvcInferError(err)
    }
}

impl From<std::io::Error> for RvcAdapterError {
    fn from(err: std::io::Error) -> Self {
        RvcAdapterError::IoError(err)
    }
}


impl RvcInfer {
    pub fn new(binary_path: PathBuf, model_version: RvcModelVersion, pitch_algorithm: PitchAlgorithm, model_path: PathBuf, data_path: PathBuf) -> Self {
        let working_dir = binary_path.parent().unwrap().to_owned();

        let mut subprocess = Command::new(binary_path)
            .arg(model_version.to_string())
            .arg(pitch_algorithm.to_string())
            .arg(model_path)
            .arg(data_path)
            .current_dir(working_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .creation_flags(0x08000000)
            .spawn()
            .expect("Failed to spawn child process");

        let buffered_stdin = std::io::BufWriter::with_capacity(1024 * 1024, subprocess.stdin.take().unwrap());
        let buffered_stdout = std::io::BufReader::with_capacity(1024 * 1024, subprocess.stdout.take().unwrap());

        RvcInfer {
            subprocess,
            input: buffered_stdin,
            output: buffered_stdout,
        }
    }

    pub fn infer(
        &mut self,
        input: ndarray::ArrayView1<f32>,
        sample_frame_16k_size: usize,
        pitch_shift: i32,
        skip_head: u32,
        return_length: u32,
    ) -> Result<ndarray::Array1<f32>, RvcAdapterError> {
        // Convert input array to bytes
        let input_bytes: Vec<u8> = input.iter().flat_map(|&x| x.to_le_bytes().to_vec()).collect();

        let sample_frame_16k_size = sample_frame_16k_size as u32;
        let input_bytes_length = input_bytes.len() as u32;

       { 
            // Write input bytes to the subprocess stdin
            // let stdin = self.subprocess.stdin.as_mut().ok_or(std::io::Error::other("Failed to open stdin"))?;
            let stdin = &mut self.input;
            stdin.write_all(&input_bytes_length.to_le_bytes())?;
            stdin.write_all(&input_bytes)?;

            // Write sample_frame_16k_size to the subprocess stdin
            stdin.write_all(&sample_frame_16k_size.to_le_bytes())?;

            // Write pitch_shift to the subprocess stdin
            stdin.write_all(&pitch_shift.to_le_bytes())?;

            // Write skip_head to the subprocess stdin
            stdin.write_all(&skip_head.to_le_bytes())?;

            // Write return_length to the subprocess stdin
            stdin.write_all(&return_length.to_le_bytes())?;


            // Flush the stdin buffer
            stdin.flush()?;
        }

        // Read the result from the subprocess stdout
        // let stdout = self.subprocess.stdout.as_mut().ok_or(std::io::Error::other("Failed to open stdout"))?;
        let stdout = &mut self.output;

        let mut output_bytes_length = [0u8; 4];
        stdout.read_exact(&mut output_bytes_length)?;
        let output_bytes_length = u32::from_le_bytes(output_bytes_length) as usize;

        println!("output_bytes_length: {}", output_bytes_length);

        let mut output_bytes = vec![0u8; output_bytes_length];
        stdout.read_exact(&mut output_bytes)?;

        println!("read from stdout");

        // Convert output bytes to array
        let output_iter = output_bytes
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|chunk| {
                let mut bytes = [0u8; std::mem::size_of::<f32>()];
                bytes.copy_from_slice(chunk);
                f32::from_le_bytes(bytes)
            });

        println!("converted to f32");

        let output_array = Array1::from_iter(output_iter);

        println!("converted to array");

        Ok(output_array)
    }
}

impl Drop for RvcInfer {
    fn drop(&mut self) {
        self.subprocess.kill().expect("Failed to kill subprocess");
    }
}
