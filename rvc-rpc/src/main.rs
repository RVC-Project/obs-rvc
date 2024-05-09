use std::env;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 4 {
        eprintln!("Usage: program_name <arg1> <arg2> <arg3>");
        return;
    }
    
    let contentvec_path = PathBuf::from(&args[1]);
    let f0_path = PathBuf::from(&args[2]);
    let model_path = PathBuf::from(&args[3]);

    // Rest of your code here...

}
