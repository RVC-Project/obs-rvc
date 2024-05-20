mod ndarray_ext;
mod rt_utils;
mod rvcadapter;

#[cfg(test)]
mod tests;

use crossbeam::{queue::ArrayQueue, sync::{Parker, Unparker}};
use ndarray::{s, ArrayView1, Zip};
use parking_lot::{Condvar, FairMutex, Mutex};
use rt_utils::{envelop_mixing, get_sola_offset, upmix_audio_data_context};
use rubato::{FftFixedInOut, Resampler};
use rvc_common::enums::{PitchAlgorithm, RvcModelVersion};
use rvcadapter::RvcInfer;

use obs_wrapper::{
    media::{audio, AudioData},
    obs_register_module, obs_string,
    prelude::*,
    properties::{BoolProp, NumberProp, PathProp, PathType, Properties},
    source::*,
};

use std::{
    borrow::Cow, cell::RefCell, collections::VecDeque, f32::consts::PI, panic, path::PathBuf, sync::{atomic::{AtomicBool, AtomicUsize}, Arc}, thread::{yield_now, JoinHandle}, time::{self, Duration, Instant}
};

use crate::{rt_utils::downmix_to_mono, rvcadapter::RvcAdapterError};

static mut BINARY_PATH: Option<PathBuf> = None;
static mut DATA_PATH: Option<PathBuf> = None;

macro_rules! get_path_from_settings {
    ($settings:ident, $setting:ident) => {
        if let Some(path) = $settings.get::<Cow<str>>($setting) {
            if !path.is_empty() {
                let path = PathBuf::from(path.to_string());
                if path.exists() {
                    Some(path)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    };

    ($field:expr, $settings:ident, $setting:ident) => {
        if let Some(path) = $settings.get::<Cow<str>>($setting) {
            let new_path_str = path.to_string();
            let orig_path = $field.as_ref().map(|p| p.to_str().unwrap_or(""));
            if $field.is_none() || orig_path != Some(new_path_str.as_str()) {
                if !path.is_empty() {
                    let path = PathBuf::from(path.to_string());
                    if path.exists() {
                        $field = Some(path);
                    } else {
                        $field = None;
                    }
                    true
                } else {
                    $field = None;
                    true
                }
            } else {
                false
            }
        } else {
            false
        }
    };
}

const SETTING_MODEL_PATH: ObsString = obs_string!("model_path");
const SETTING_INDEX_PATH: ObsString = obs_string!("index_path");
const SETTING_PITCH_SHIFT: ObsString = obs_string!("pitch_shift");
const SETTING_RESONANCE_SHIFT: ObsString = obs_string!("resonance_shift");
const SETTING_INDEX_RATE: ObsString = obs_string!("index_rate");
const SETTING_LOUDNESS_FACTOR: ObsString = obs_string!("loudness_factor");
const SETTING_PITCH_ALGORITHM: ObsString = obs_string!("pitch_algorithm");
const SETTING_SAMPLE_LENGTH: ObsString = obs_string!("sample_length");
const SETTING_FADE_LENGTH: ObsString = obs_string!("fade_length");
const SETTING_EXTRA_INFERENCE_TIME: ObsString = obs_string!("extra_inference_time");
const SETTING_DEST_SAMPLE_RATE: ObsString = obs_string!("dest_sample_rate");
const SETTING_MODEL_VERSION: ObsString = obs_string!("model_version");
const SETTING_SKIP_INFERENCE: ObsString = obs_string!("skip_inference");

struct Frame {
    data: Vec<f32>,
    timestamp: u64,
}

struct RvcInferenceState {
    model_path: Option<PathBuf>,
    index_path: Option<PathBuf>,
    model_version: RvcModelVersion,
    pitch_algorithm: PitchAlgorithm,
    model_output_sample_rate: usize,
    pitch_shift: i32,
    resonance_shift: f64,
    index_rate: f64,
    rms_mix_rate: f64,
    sample_length: f64,
    crossfade_length: f64,
    extra_inference_time: f64,

    sample_rate: usize,

    sample_frame_size: usize,
    sample_frame_16k_size: usize,
    crossfade_frame_size: usize,
    sola_buffer_frame_size: usize,
    sola_search_frame_size: usize,
    extra_frame_size: usize,
    model_return_length: usize,
    model_return_size: usize,

    input_buffer: Vec<f32>,
    input_buffer_16k: Vec<f32>,
    sola_buffer: ndarray::Array1<f32>,
    output_buffer: Vec<f32>,

    fade_in_window: ndarray::Array1<f32>,
    fade_out_window: ndarray::Array1<f32>,

    skip_inference: bool,

    upsampler: FftFixedInOut<f32>,
    downsampler: FftFixedInOut<f32>,

    engine: Option<RvcInfer>,
}

struct RvcInferenceSharedState {
    state: FairMutex<RvcInferenceState>,
    running: AtomicBool,
    channels: usize,
    input: ArrayQueue<Frame>,
    output: ArrayQueue<Frame>,
    buffer_changed: AtomicBool,
    sample_frame_size: AtomicUsize,
}

struct RvcInferenceFilter {
    thread_handle: Option<JoinHandle<()>>,
    shared_state: Arc<RvcInferenceSharedState>,
    has_input: Option<Unparker>,
    filter_audio_lock: Mutex<()>,
}

struct RvcInferenceModule {
    context: ModuleRef,
}

impl Sourceable for RvcInferenceFilter {
    fn get_id() -> ObsString {
        obs_string!("rvc_inference_filter")
    }
    fn get_type() -> SourceType {
        SourceType::Filter
    }
    fn create(create: &mut CreatableSourceContext<Self>, _source: SourceRef) -> Self {
        let (sample_rate, channels) =
            create.with_audio(|audio| (audio.sample_rate(), audio.channels()));

        let settings = &mut create.settings;

        let model_path = get_path_from_settings!(settings, SETTING_MODEL_PATH);
        let index_path = get_path_from_settings!(settings, SETTING_INDEX_PATH);

        settings.set_default::<i32>(SETTING_DEST_SAMPLE_RATE, 40000);
        settings.set_default::<i32>(SETTING_PITCH_SHIFT, 12);
        settings.set_default::<f32>(SETTING_RESONANCE_SHIFT, 0.07);
        settings.set_default::<f32>(SETTING_INDEX_RATE, 0.0);
        settings.set_default::<f32>(SETTING_LOUDNESS_FACTOR, 0.5);
        settings.set_default::<f32>(SETTING_SAMPLE_LENGTH, 0.30);
        settings.set_default::<f32>(SETTING_FADE_LENGTH, 0.07);
        settings.set_default::<f32>(SETTING_EXTRA_INFERENCE_TIME, 2.00);
        settings.set_default::<RvcModelVersion>(SETTING_MODEL_VERSION, RvcModelVersion::V2);
        settings
            .set_default::<PitchAlgorithm>(SETTING_PITCH_ALGORITHM, PitchAlgorithm::Rmvpe);
        settings.set_default::<bool>(SETTING_SKIP_INFERENCE, false);

        let mut model_output_sample_rate = settings.get(SETTING_DEST_SAMPLE_RATE).unwrap_or(40000);
        let sample_length = settings.get(SETTING_SAMPLE_LENGTH).unwrap_or(0.30);
        let crossfade_length = settings.get(SETTING_FADE_LENGTH).unwrap_or(0.07);
        let extra_inference_time = settings.get(SETTING_EXTRA_INFERENCE_TIME).unwrap_or(2.00);
        let model_version = settings
            .get(SETTING_MODEL_VERSION)
            .unwrap_or(RvcModelVersion::V2);
        let pitch_algorithm = settings
            .get(SETTING_PITCH_ALGORITHM)
            .unwrap_or(PitchAlgorithm::Rmvpe);

        let skip_inference = settings.get(SETTING_SKIP_INFERENCE).unwrap_or(false);

        let zc = sample_rate / 100;

        let sample_frame_time = (sample_length * sample_rate as f64 / zc as f64).round() as usize;
        let sample_frame_size = sample_frame_time * zc;

        let sample_frame_16k = sample_frame_time * 160;
        let crossfade_frame_size =
            (crossfade_length * sample_rate as f64 / zc as f64).round() as usize * zc;
        let sola_buffer_frame_size = usize::min(crossfade_frame_size, 4 * zc);
        let sola_search_frame_size = zc;
        let extra_frame_size =
            (extra_inference_time * sample_rate as f64 / zc as f64).round() as usize * zc;

        let input_buffer_size =
            extra_frame_size + crossfade_frame_size + sola_search_frame_size + sample_frame_size;
        let input_buffer = vec![0_f32; input_buffer_size];

        let input_buffer_16k_size = 160 * input_buffer_size / zc;
        let input_buffer_16k = vec![0_f32; input_buffer_16k_size];

        let model_return_length =
            (sample_frame_size + sola_buffer_frame_size + sola_search_frame_size) / zc;
        let mut model_return_size = model_return_length * (model_output_sample_rate / 100);

        if skip_inference {
            model_output_sample_rate = 16000;
            model_return_size = model_return_length * 160;
        }

        let sola_buffer = ndarray::Array1::zeros(sola_buffer_frame_size);

        let mut fade_in_window = ndarray::Array1::linspace(0.0, 1.0, sola_buffer_frame_size);
        fade_in_window.mapv_inplace(|x| f32::sin(x * 0.5 * PI).powi(2));
        let fade_out_window = fade_in_window.mapv(|x| 1.0 - x);

        // 48k => 16k sample frame size
        let downsampler = FftFixedInOut::new(
            sample_rate, 16000, sample_frame_size + 2 * zc, 1).unwrap();

        // model_sample_size => 48k
        let upsampler =
            FftFixedInOut::new(model_output_sample_rate, sample_rate, model_return_size, 1)
                .unwrap();

        let output_buffer = vec![0_f32; upsampler.output_frames_max()];

        let binary_path = unsafe { BINARY_PATH.as_ref().unwrap().parent().unwrap().join("rvc-rpc.exe") };
        let infer_data_path = unsafe { DATA_PATH.as_ref().unwrap() }.join("rvcinfer");

        let rvc = match model_path.clone() {
            Some(path) => Some(RvcInfer::new(binary_path, model_version, pitch_algorithm, path, infer_data_path)),
            None => None,
        };

        let state = RvcInferenceState {
            sample_rate,

            model_path,
            index_path,
            model_version,
            pitch_algorithm,
            model_output_sample_rate,
            pitch_shift: settings.get(SETTING_PITCH_SHIFT).unwrap_or(12),
            resonance_shift: settings.get(SETTING_RESONANCE_SHIFT).unwrap_or(0.00),
            index_rate: settings.get(SETTING_INDEX_RATE).unwrap_or(0.00),
            rms_mix_rate: settings.get(SETTING_LOUDNESS_FACTOR).unwrap_or(0.00),
            sample_length,
            crossfade_length,
            extra_inference_time,

            sample_frame_size,
            sample_frame_16k_size: sample_frame_16k,
            crossfade_frame_size,
            sola_buffer_frame_size,
            sola_search_frame_size,
            extra_frame_size,
            model_return_length,
            model_return_size,

            input_buffer,
            input_buffer_16k,
            sola_buffer,
            output_buffer,

            fade_in_window,
            fade_out_window,

            skip_inference,

            upsampler,
            downsampler,

            engine: rvc,
        };

        let state = FairMutex::new(state);

        let shared_state = RvcInferenceSharedState {
            state,
            running: AtomicBool::new(true),
            channels,
            input: ArrayQueue::new(200),
            output: ArrayQueue::new(300),
            buffer_changed: AtomicBool::new(false),
            sample_frame_size: AtomicUsize::new(sample_frame_size),
        };

        let shared_state = Arc::new(shared_state);

        Self {
            thread_handle: None,
            shared_state,
            has_input: None,
            filter_audio_lock: Mutex::new(()),
        }
    }
}

impl GetNameSource for RvcInferenceFilter {
    fn get_name() -> ObsString {
        obs_string!("Retrieval Voice Conversion")
    }
}

impl GetPropertiesSource for RvcInferenceFilter {
    fn get_properties(&mut self) -> Properties {
        let mut p = Properties::new();

        p.add(
            SETTING_MODEL_PATH,
            obs_string!("模型路径"),
            PathProp::new(PathType::File).with_filter(obs_string!("ONNX 模型文件 (*.onnx)")),
        );

        p.add(
            SETTING_INDEX_PATH,
            obs_string!("RVC 音高索引文件路径"),
            PathProp::new(PathType::File).with_filter(obs_string!("Index 文件 (*.index)")),
        );

        let mut version_list =
            p.add_list::<RvcModelVersion>(SETTING_MODEL_VERSION, obs_string!("模型版本"), false);

        version_list.push(obs_string!("v1"), RvcModelVersion::V1);
        version_list.push(obs_string!("v2"), RvcModelVersion::V2);

        p.add(
            SETTING_DEST_SAMPLE_RATE,
            obs_string!("模型目标采样率"),
            NumberProp::new_int()
                .with_range(16000..=48000)
                .with_step(4000)
                .with_slider(),
        );

        let mut pitch_algorithm_list =
            p.add_list::<PitchAlgorithm>(SETTING_PITCH_ALGORITHM, obs_string!("音高算法"), false);

        pitch_algorithm_list.push(obs_string!("RMVPE"), PitchAlgorithm::Rmvpe);

        p.add(
            SETTING_PITCH_SHIFT,
            obs_string!("音调设置"),
            NumberProp::new_int()
                .with_range(-24..=24)
                .with_step(1)
                .with_slider(),
        );

        p.add(
            SETTING_RESONANCE_SHIFT,
            obs_string!("共振偏移"),
            NumberProp::new_float(0.07)
                .with_range(-5.0..=5.0)
                .with_slider(),
        );

        p.add(
            SETTING_INDEX_RATE,
            obs_string!("索引率"),
            NumberProp::new_float(0.01)
                .with_range(0.00..=1.00)
                .with_slider(),
        );

        p.add(
            SETTING_LOUDNESS_FACTOR,
            obs_string!("响度因子"),
            NumberProp::new_float(0.01)
                .with_range(0.00..=1.00)
                .with_slider(),
        );

        p.add(
            SETTING_SAMPLE_LENGTH,
            obs_string!("采样长度"),
            NumberProp::new_float(0.01)
                .with_range(0.01..=1.50)
                .with_slider(),
        );

        p.add(
            SETTING_FADE_LENGTH,
            obs_string!("淡入淡出长度"),
            NumberProp::new_float(0.01)
                .with_range(0.01..=0.15)
                .with_slider(),
        );

        p.add(
            SETTING_EXTRA_INFERENCE_TIME,
            obs_string!("额外推理时长"),
            NumberProp::new_float(0.01)
                .with_range(0.00..=5.00)
                .with_slider(),
        );

        p.add(
            SETTING_SKIP_INFERENCE,
            obs_string!("跳过推理"),
            BoolProp
        );

        p
    }
}

impl UpdateSource for RvcInferenceFilter {
    fn update(&mut self, settings: &mut DataObj, context: &mut GlobalContext) {
        let mut state = self.shared_state.state.lock();

        let sample_rate = context.with_audio(|audio| audio.sample_rate());
        state.sample_rate = sample_rate;

        let model_changed = get_path_from_settings!(state.model_path, settings, SETTING_MODEL_PATH);
        let index_changed = get_path_from_settings!(state.index_path, settings, SETTING_INDEX_PATH);

        let mut recalculate_input_buffer = false;
        let mut reload_rvc = model_changed || index_changed;

        if let Some(new_pitch_shift) = settings.get(SETTING_PITCH_SHIFT) {
            if state.pitch_shift != new_pitch_shift {
                state.pitch_shift = new_pitch_shift;
            }
        }

        if let Some(new_resonance_shift) = settings.get(SETTING_RESONANCE_SHIFT) {
            if state.resonance_shift != new_resonance_shift {
                state.resonance_shift = new_resonance_shift;
            }
        }

        if let Some(new_index_rate) = settings.get(SETTING_INDEX_RATE) {
            if state.index_rate != new_index_rate {
                state.index_rate = new_index_rate;
            }
        }

        if let Some(new_rms_mix_rate) = settings.get(SETTING_LOUDNESS_FACTOR) {
            if state.rms_mix_rate != new_rms_mix_rate {
                state.rms_mix_rate = new_rms_mix_rate;
            }
        }

        if let Some(new_sample_length) = settings.get(SETTING_SAMPLE_LENGTH) {
            if state.sample_length != new_sample_length {
                state.sample_length = new_sample_length;
                recalculate_input_buffer = true;
            }
        }

        if let Some(new_fade_length) = settings.get(SETTING_FADE_LENGTH) {
            if state.crossfade_length != new_fade_length {
                state.crossfade_length = new_fade_length;
                recalculate_input_buffer = true;
            }
        }

        if let Some(new_extra_inference_time) = settings.get(SETTING_EXTRA_INFERENCE_TIME) {
            if state.extra_inference_time != new_extra_inference_time {
                state.extra_inference_time = new_extra_inference_time;
                recalculate_input_buffer = true;
            }
        }

        if let Some(new_dest_sample_rate) = settings.get(SETTING_DEST_SAMPLE_RATE) {
            if state.model_output_sample_rate != new_dest_sample_rate {
                state.model_output_sample_rate = new_dest_sample_rate;
                recalculate_input_buffer = true;
            }
        }

        if let Some(new_model_version) = settings.get(SETTING_MODEL_VERSION) {
            if state.model_version != new_model_version {
                state.model_version = new_model_version;
                reload_rvc = true;
            }
        }

        if let Some(new_pitch_algorithm) = settings.get(SETTING_PITCH_ALGORITHM) {
            if state.pitch_algorithm != new_pitch_algorithm {
                state.pitch_algorithm = new_pitch_algorithm;
                reload_rvc = true;
            }
        }

        if let Some(new_skip_inference) = settings.get(SETTING_SKIP_INFERENCE) {
            if state.skip_inference != new_skip_inference {
                state.skip_inference = new_skip_inference;
                recalculate_input_buffer = true;
            }
        }

        if recalculate_input_buffer {
            self.shared_state
                .buffer_changed
                .store(true, std::sync::atomic::Ordering::Relaxed);
            let sample_length = state.sample_length;
            let crossfade_length = state.crossfade_length;
            let extra_inference_time = state.extra_inference_time;
            let mut model_output_sample_rate = state.model_output_sample_rate;

            // zc is sample per 0.1 sec
            let zc = sample_rate / 100;

            let sample_frame_time =
                (sample_length * sample_rate as f64 / zc as f64).round() as usize;
            let sample_frame_size = sample_frame_time * zc;
            let sample_frame_16k = sample_frame_time * 160;
            let crossfade_frame_size =
                (crossfade_length * sample_rate as f64 / zc as f64).round() as usize * zc;
            let sola_buffer_frame_size = usize::min(crossfade_frame_size, 4 * zc);
            let sola_search_frame_size = zc;
            let extra_frame_size =
                (extra_inference_time * sample_rate as f64 / zc as f64).round() as usize * zc;
            let model_return_length =
                (sample_frame_size + sola_buffer_frame_size + sola_search_frame_size) / zc;
            let mut model_return_size = model_return_length * (model_output_sample_rate / 100);

            if state.skip_inference {
                model_output_sample_rate = 16000;
                model_return_size = model_return_length * 160;
            }

            state.sample_frame_size = sample_frame_size;
            state.sample_frame_16k_size = sample_frame_16k;
            state.crossfade_frame_size = crossfade_frame_size;
            state.sola_buffer_frame_size = sola_buffer_frame_size;
            state.sola_search_frame_size = sola_search_frame_size;
            state.extra_frame_size = extra_frame_size;
            state.model_return_length = model_return_length;
            state.model_return_size = model_return_size;
            self.shared_state.sample_frame_size.store(sample_frame_size, std::sync::atomic::Ordering::Relaxed);

            let input_buffer_size = extra_frame_size
                + crossfade_frame_size
                + sola_search_frame_size
                + sample_frame_size;
            state.input_buffer.resize(input_buffer_size, 0_f32);

            let input_buffer_16k_size = 160 * input_buffer_size / zc;
            state.input_buffer_16k.resize(input_buffer_16k_size, 0_f32);

            let mut fade_in_window = ndarray::Array1::linspace(0.0, 1.0, sola_buffer_frame_size);
            fade_in_window.mapv_inplace(|x| f32::sin(x * 0.5 * PI).powi(2));
            let fade_out_window = fade_in_window.mapv(|x| 1.0 - x);

            state.fade_in_window = fade_in_window;
            state.fade_out_window = fade_out_window;

            // model_sample_size => 48k
            state.upsampler =
                FftFixedInOut::new(model_output_sample_rate, sample_rate, model_return_size, 1)
                    .unwrap();
            let output_buffer_size = state.upsampler.output_frames_max();
            state.output_buffer.resize(output_buffer_size, 0_f32);
            // 48k => 16k sample frame size
            state.downsampler =
                FftFixedInOut::new(sample_rate, 16000, sample_frame_size + 2 * zc, 1).unwrap();

            state.input_buffer.fill(0_f32);
            state.input_buffer_16k.fill(0_f32);


        }
    
        if reload_rvc {
            Self::restart_rvc_engine_inner(&mut state);
        }
    }
}

impl FilterAudioSource for RvcInferenceFilter {
    fn filter_audio(&mut self, audio: &mut audio::AudioDataContext) -> FilterAudioResult {
        // self.start_thread();
        let _lock = self.filter_audio_lock.lock();
        let timestamp = audio.timestamp();
        let main_channel = downmix_to_mono(audio, self.shared_state.channels).unwrap();
        
        let frame = Frame {
            data: main_channel.to_vec(),
            timestamp,
        };

        self.shared_state.input.force_push(frame);

        if let Some(has_input) = self.has_input.as_ref() {
            has_input.unpark();
        }

        let output = match self.shared_state.output.pop() {
            Some(frame) => frame,
            None => return FilterAudioResult::Discarded,
        };

        let timestamp = output.timestamp;
        // assuming same length
        if output.data.len() < main_channel.len() {
            let mut output_head = 0;
            main_channel[output_head..output.data.len()].copy_from_slice(&output.data);
            output_head += output.data.len();

            while output_head < main_channel.len() {
                let output = match self.shared_state.output.pop() {
                    Some(frame) => frame,
                    None => break,
                };

                main_channel[output_head..(output_head + output.data.len())].copy_from_slice(&output.data);
                output_head += output.data.len();
            }
            
        } else {
            main_channel.copy_from_slice(&output.data);
        }

        audio.set_timestamp(timestamp);
        upmix_audio_data_context(audio, self.shared_state.channels).unwrap();
        FilterAudioResult::Modified
    }
}

impl ActivateSource for RvcInferenceFilter {
    fn activate(&mut self) {
        self.clear_state();
        self.start_thread();
    }
}

impl DeactivateSource for RvcInferenceFilter {
    fn deactivate(&mut self) {
        self.stop_thread();
        self.clear_state();
    }
}

fn process_one_frame(input_sample: &[f32], state: &mut RvcInferenceState) -> ndarray::Array1<f32> {
    // move and append the last n samples
    {
        let input_buffer_retaining = state.input_buffer.len() - state.sample_frame_size;
        state.input_buffer.copy_within(state.sample_frame_size.., 0);
        state.input_buffer[input_buffer_retaining..].copy_from_slice(input_sample);
    }

    // resample and set to 16k

    state
        .input_buffer_16k
        .copy_within(state.sample_frame_16k_size.., 0);

    let downsample_start = state.input_buffer.len() - state.sample_frame_size - 2 * state.sample_rate / 100;
    let input_sample = &[&state.input_buffer[downsample_start..]];
    match state.downsampler.process(input_sample, None) {
        Ok(result) => {
            let copy_begin = state.input_buffer_16k.len() - (state.sample_frame_size / (state.sample_rate / 100) + 1) * 160;
            state.input_buffer_16k[copy_begin..].copy_from_slice(&result[0][160..]);
        },
        Err(e) => {
            panic!("Error: {:?}", e);
        }
    };

    let input_buffer_view =
        ndarray::ArrayView1::from_shape((state.input_buffer.len(),), &state.input_buffer).unwrap();

    let input_buffer_16k_view =
        ndarray::ArrayView1::from_shape((state.input_buffer_16k.len(),), &state.input_buffer_16k)
            .unwrap();

    // println!("input: {:?}", input_buffer_16k_view);

    let skip_head = (state.extra_frame_size / (state.sample_rate / 100)) as u32;

    // inference
    let output = if state.skip_inference {
        let output_start = input_buffer_16k_view.len() - state.model_return_size;
        input_buffer_16k_view.slice(s![output_start..]).to_owned()
    } else if let Some(engine) = state.engine.as_mut() {
        match engine.infer(
            input_buffer_16k_view,
            state.sample_frame_16k_size,
            state.pitch_shift,
            skip_head,
            state.model_return_length as u32
        ) {
            Ok(output) => {
                output
                // let skip_head = state.extra_frame_size / (state.sample_rate / 100);
                // let flow_head = if skip_head > 24 { skip_head - 24 } else { 0 };
                // let dec_head = skip_head - flow_head;
                // let end = state.model_return_size + dec_head;
                // output.slice(s![dec_head..end]).to_owned()
            },
            Err(e) => {
                eprintln!("Error: {:?}", e);

                match e {
                    RvcAdapterError::IoError(e) => {
                        RvcInferenceFilter::restart_rvc_engine_inner(state);
                    },
                    _ => (),
                }

                return ndarray::Array1::zeros(state.sample_frame_size);
            }
        }
    } else {
        return ndarray::Array1::zeros(state.sample_frame_size);
    };

    if output.len() != state.model_return_size {
        eprintln!(
            "Model output size mismatch: {} != {}",
            output.len(),
            state.model_return_size
        );
        // return ndarray::Array1::zeros(state.sample_frame_size);
    }

    let mut output = {
        let output = output.into_raw_vec();
        let output_sample = &[&output];
        let output_buffer = &mut [&mut state.output_buffer[..]];

        let result = state
            .upsampler
            .process_into_buffer(output_sample, output_buffer, None);
        if let Err(e) = result {
            panic!("Error: {:?}", e);
        }
        let (csi, cso) = result.unwrap();
        ndarray::ArrayViewMut1::from_shape((cso,), &mut state.output_buffer)
            .unwrap()
    };

    if state.rms_mix_rate < 1. {
        envelop_mixing(
            input_buffer_view.slice(s![state.extra_frame_size..]),
            output.view_mut(),
            state.sample_rate,
            state.rms_mix_rate,
        )
    }

    // sola
    let sola_offset = get_sola_offset(
        output.view(),
        state.sola_buffer.view(),
        state.sola_buffer_frame_size,
        state.sola_search_frame_size,
    )
    .unwrap();

    let mut output = output.slice_mut(s![sola_offset..]);

    // TODO: phase vocoder
    Zip::from(output.slice_mut(s![..state.sola_buffer_frame_size]))
        .and(state.fade_in_window.view())
        .and(state.sola_buffer.view())
        .and(state.fade_out_window.view())
        .for_each(|output_sola_buffer_view, fade_in, sola, fade_out| {
            *output_sola_buffer_view = *output_sola_buffer_view * fade_in + sola * fade_out;
        });


    // self.sola_buffer.assign(&output[self.sample_frame_size..(self.sample_frame_size + self.sola_buffer_frame_size)]);
    state.sola_buffer.assign(&output.slice(s![
        state.sample_frame_size..(state.sample_frame_size + state.sola_buffer_frame_size)
    ]));

    // output.iter().for_each(|sample| self.output.push_back(*sample));
    output.slice(s![..state.sample_frame_size]).into_owned()
}

fn thread_loop(shared_state: Arc<RvcInferenceSharedState>, has_input: Parker) {
    let mut input_sample: Vec<f32> = {
        let state = shared_state.state.lock();
        Vec::with_capacity(state.sample_frame_size * 2)
    };

    let mut output_sample: Vec<f32> = Vec::with_capacity(input_sample.capacity());

    let mut frame_buffer: VecDeque<Frame> = VecDeque::with_capacity(300);

    while shared_state
        .running
        .load(std::sync::atomic::Ordering::Relaxed)
    {
        let mut state = match shared_state.state.try_lock() {
            Some(state) => state,
            None => {
                continue;
            }
        };

        let sample_frame_size = state.sample_frame_size;
        while input_sample.len() < sample_frame_size {
            let frame = shared_state.input.pop();
            if let Some(frame) = frame {
                input_sample.extend_from_slice(&frame.data);
                frame_buffer.push_back(frame);
            } else {
                has_input.park_timeout(Duration::from_secs(2));
                if !shared_state
                    .running
                    .load(std::sync::atomic::Ordering::Relaxed) {
                    return;
                }
            }
        }

        let output_frame = process_one_frame(&input_sample[..sample_frame_size], &mut state);
        output_sample.extend_from_slice(&output_frame.as_slice().unwrap());

        let mut output_head = 0;
        while let Some(mut frame) = frame_buffer.pop_front() {
            let frame_len = frame.data.len();
            if output_head + frame_len >= output_sample.len() {
                frame_buffer.push_front(frame);
                break;
            }
            let output = &output_sample[output_head..output_head + frame_len];
            frame.data.copy_from_slice(output);
            let res = shared_state.output.force_push(frame);
            output_head += frame_len;
        }
        
        input_sample.copy_within(sample_frame_size.., 0);
        input_sample.truncate(input_sample.len() - sample_frame_size);
        output_sample.copy_within(output_head.., 0);
        output_sample.truncate(output_sample.len() - output_head);

    }
}

impl RvcInferenceFilter {
    fn start_thread(&mut self) {
        if self.thread_handle.is_none() {
            eprintln!("Starting thread...");
            self.shared_state.running.store(true, std::sync::atomic::Ordering::Relaxed);
            let shared_state = self.shared_state.clone();
            let parker = Parker::new();
            let unparker = parker.unparker().clone();
            let handle = std::thread::spawn(move || thread_loop(shared_state, parker));
            self.thread_handle.replace(handle);
            self.has_input.replace(unparker);
        }
    }

    fn stop_thread(&mut self) {
        if let Some(handle) = self.thread_handle.take() {
            eprintln!("Stopping thread...");
            self.shared_state
                .running
                .store(false, std::sync::atomic::Ordering::Relaxed);
            self.has_input.take();
            match handle.join() {
                Ok(_) => (),
                Err(e) => {
                    println!("Error joining thread: {:?}", e);
                }
            }
        }
    }

    fn restart_rvc_engine(&mut self) {
        let mut state = self.shared_state.state.lock();
        Self::restart_rvc_engine_inner(&mut state);
    }

    fn restart_rvc_engine_inner(state: &mut RvcInferenceState) {
        let binary_path = unsafe { BINARY_PATH.as_ref().unwrap().parent().unwrap().join("rvc-rpc.exe") };
        let infer_data_path = unsafe { DATA_PATH.as_ref().unwrap() }.join("rvcinfer");

        let rvc = match state.model_path.clone() {
            Some(path) => Some(RvcInfer::new(binary_path, state.model_version, state.pitch_algorithm, path, infer_data_path)),
            None => None,
        };

        state.engine = rvc;
    }

    fn clear_state(&mut self) {
        let mut state = self.shared_state.state.lock();
        state.input_buffer.fill(0_f32);
        state.input_buffer_16k.fill(0_f32);
        state.sola_buffer.fill(0_f32);
        state.output_buffer.fill(0_f32);
    }
}

impl Drop for RvcInferenceFilter {
    fn drop(&mut self) {
        self.stop_thread();
    }
}

impl Module for RvcInferenceModule {
    fn new(context: ModuleRef) -> Self {

        let binary_path = PathBuf::from(context.binary_path().unwrap().as_str());
        let data_path = PathBuf::from(context.data_path().unwrap().as_str());


        unsafe {
            BINARY_PATH = Some(binary_path);
            DATA_PATH = Some(data_path);
        };

        Self { context }
    }

    fn get_ctx(&self) -> &ModuleRef {
        &self.context
    }

    fn load(&mut self, load_context: &mut LoadContext) -> bool {
        let source = load_context
            .create_source_builder::<RvcInferenceFilter>()
            .enable_get_name()
            .enable_update()
            .enable_get_properties()
            .enable_filter_audio()
            .enable_activate()
            .enable_deactivate()
            .build();

        load_context.register_source(source);

        true
    }

    fn description() -> ObsString {
        obs_string!("A filter that uses Retrieval-based Voice Conversion to change your voice.")
    }
    fn name() -> ObsString {
        obs_string!("Retrieval Voice Conversion")
    }
    fn author() -> ObsString {
        obs_string!("Joe")
    }
}

obs_register_module!(RvcInferenceModule);
