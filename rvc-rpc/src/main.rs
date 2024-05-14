use std::io::Write;
use std::{env, io::Read};
use std::path::PathBuf;
use ndarray::Array1;
use rvc_common::enums::{PitchAlgorithm, RvcModelVersion};
use rvc::RvcInfer;

fn main() {
    #[cfg(debug_assertions)]
    tracing_subscriber::fmt::fmt().with_writer(std::io::stderr).init();

    let args: Vec<String> = env::args().collect();

    if args.len() < 4 {
        eprintln!("Usage: rvc-rpc <version> <f0_algorithm> <model> <data>");
        return;
    }
    
    let model_version = RvcModelVersion::from(args[1].as_str());
    let pitch_algorithm = PitchAlgorithm::from(args[2].as_str());
    let model_path = PathBuf::from(&args[3]);
    let data_path = PathBuf::from(&args[4]);

    let cwd = env::current_dir().unwrap();
    let ort_path = cwd.join("onnxruntime.dll");
    match ort::init_from(ort_path.to_string_lossy()).commit() {
        Ok(_) => (),
        Err(e) => {
            panic!("Error loading onnxruntime: {:?}", e);
        }
    }

    let mut rvc = RvcInfer::new(data_path);

    match rvc.load_contentvec(model_version) {
        Ok(_) => (),
        Err(e) => {
            panic!("Error loading contentvec model: {:?}", e);
        }
    }

    match rvc.load_f0(pitch_algorithm) {
        Ok(_) => (),
        Err(e) => {
            panic!("Error loading f0 model: {:?}", e);
        }
    }

    match rvc.load_model(model_path) {
        Ok(_) => (),
        Err(e) => {
            panic!("Error loading model: {:?}", e);
        }
    }

    let stdin = std::io::stdin().lock();
    let stdout = std::io::stdout().lock();

    let mut buffered_stdin = std::io::BufReader::with_capacity(1024 * 1024, stdin);
    let mut buffered_stdout = std::io::BufWriter::with_capacity(1024 * 1024, stdout);

    eprintln!("Ready to receive input");

    loop {
        let mut input_bytes_length = [0u8; 4];
        buffered_stdin.read_exact(&mut input_bytes_length).unwrap();
        let input_bytes_length = u32::from_le_bytes(input_bytes_length) as usize;

        let mut input_bytes = vec![0u8; input_bytes_length];
        buffered_stdin.read_exact(&mut input_bytes).unwrap();
        let input = Array1::from_shape_fn(input_bytes_length / 4, |i| {
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&input_bytes[i * 4..(i + 1) * 4]);
            f32::from_le_bytes(bytes)
        });

        let mut sample_frame_16k_size = [0u8; 4];
        buffered_stdin.read_exact(&mut sample_frame_16k_size).unwrap();
        let sample_frame_16k_size = u32::from_le_bytes(sample_frame_16k_size) as usize;

        let mut pitch_shift = [0u8; 4];
        buffered_stdin.read_exact(&mut pitch_shift).unwrap();
        let pitch_shift = i32::from_le_bytes(pitch_shift);

        let mut skip_head = [0u8; 4];
        buffered_stdin.read_exact(&mut skip_head).unwrap();
        let skip_head = u32::from_le_bytes(skip_head);

        let mut return_length = [0u8; 4];
        buffered_stdin.read_exact(&mut return_length).unwrap();
        let return_length = u32::from_le_bytes(return_length);

        let output = rvc.infer(input.view(), sample_frame_16k_size, Some(pitch_shift), skip_head, return_length).unwrap();

        let output_bytes: Vec<u8> = output.iter().flat_map(|&x| x.to_le_bytes().to_vec()).collect();
        let output_bytes_length = output_bytes.len();

        buffered_stdout.write_all(&(output_bytes_length as u32).to_le_bytes()).unwrap();
        buffered_stdout.write_all(&output_bytes).unwrap();
        buffered_stdout.flush().unwrap();
    }

}
