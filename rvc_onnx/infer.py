import re
import os
import sys
import time
import onnx
import json
import faiss
import torch
import codecs
import shutil
import librosa
import logging
import warnings
import requests
import parselmouth
import onnxruntime
import logging.handlers

import numpy as np
import soundfile as sf
import torch.nn.functional as F

from tqdm import tqdm
from scipy import signal
from rvc_onnx.fairseq import checkpoint_utils
from pydub import AudioSegment, silence

warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())

from lib.f0_method.FCPE import FCPE
from lib.f0_method.RMVPE import RMVPE
from lib.f0_method.WORLD import PYWORLD
from lib.f0_method.CREPE import predict, mean, median

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

for l in ["torch", "faiss", "httpx", "fairseq", "httpcore", "faiss.loader", "numba.core", "urllib3"]:
    logging.getLogger(l).setLevel(logging.ERROR)

def hf_download_file(url, output_path=None):
    """Downloads a file from Hugging Face Hub."""
    url = url.replace("/blob/", "/resolve/").replace("?download=true", "").strip()

    if output_path is None: output_path = os.path.basename(url)
    else: output_path = os.path.join(output_path, os.path.basename(url)) if os.path.isdir(output_path) else output_path

    response = requests.get(url, stream=True, timeout=300)

    if response.status_code == 200:        
        progress_bar = tqdm(total=int(response.headers.get("content-length", 0)), ncols=100, unit="byte")

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                progress_bar.update(len(chunk))
                f.write(chunk)

        progress_bar.close()
        return output_path
    else: raise ValueError(f"Failed to download file. Status code: {response.status_code}")

def check_predictors(method):
    """Checks and downloads necessary predictor models."""
    def download(predictor_name):
        if not os.path.exists(os.path.join(predictor_name)): hf_download_file(codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/cerqvpgbef/", "rot13") + predictor_name, predictor_name)

    model_dict = {**dict.fromkeys(["rmvpe", "rmvpe-legacy"], "rmvpe.pt"), **dict.fromkeys(["rmvpe-onnx", "rmvpe-legacy-onnx"], "rmvpe.onnx"), **dict.fromkeys(["fcpe", "fcpe-legacy"], "fcpe.pt"), **dict.fromkeys(["fcpe-onnx", "fcpe-legacy-onnx"], "fcpe.onnx"), **dict.fromkeys(["crepe-full", "mangio-crepe-full"], "crepe_full.pth"), **dict.fromkeys(["crepe-full-onnx", "mangio-crepe-full-onnx"], "crepe_full.onnx"), **dict.fromkeys(["crepe-large", "mangio-crepe-large"], "crepe_large.pth"), **dict.fromkeys(["crepe-large-onnx", "mangio-crepe-large-onnx"], "crepe_large.onnx"), **dict.fromkeys(["crepe-medium", "mangio-crepe-medium"], "crepe_medium.pth"), **dict.fromkeys(["crepe-medium-onnx", "mangio-crepe-medium-onnx"], "crepe_medium.onnx"), **dict.fromkeys(["crepe-small", "mangio-crepe-small"], "crepe_small.pth"), **dict.fromkeys(["crepe-small-onnx", "mangio-crepe-small-onnx"], "crepe_small.onnx"), **dict.fromkeys(["crepe-tiny", "mangio-crepe-tiny"], "crepe_tiny.pth"), **dict.fromkeys(["crepe-tiny-onnx", "mangio-crepe-tiny-onnx"], "crepe_tiny.onnx"), **dict.fromkeys(["harvest", "dio"], "world.pth")}

    if "hybrid" in method:
        methods_str = re.search("hybrid\[(.+)\]", method)
        if methods_str: methods = [method.strip() for method in methods_str.group(1).split("+")]
        for method in methods:
            if method in model_dict: download(model_dict[method])
    elif method in model_dict: download(model_dict[method])

def check_embedders(hubert):
    """Checks and downloads necessary embedder models."""
    if hubert in ["contentvec_base", "hubert_base", "japanese_hubert_base", "korean_hubert_base", "chinese_hubert_base", "Hidden_Rabbit_last", "portuguese_hubert_base"]:
        model_path = hubert + ".pt"
        if not os.path.exists(model_path): hf_download_file(codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/rzorqqref/", "rot13") + f"{hubert}.pt", model_path)

def load_audio(file_path):
    """Loads an audio file and resamples it to 16kHz."""
    try:
        file_path = file_path.strip(" ").strip("\"").strip("\n").strip("\"").strip(" ")
        if not os.path.isfile(file_path): raise FileNotFoundError(f"File not found: {file_path}")

        audio, sr = sf.read(file_path)

        if len(audio.shape) > 1: audio = librosa.to_mono(audio.T)
        if sr != 16000: audio = librosa.resample(audio, orig_sr=sr, target_sr=16000, res_type="soxr_vhq")
    except Exception as e:
        raise RuntimeError(f"Error loading audio file: {e}")
    return audio.flatten()

def process_audio(file_path, output_path):
    """Splits audio into non-silent chunks."""
    try:
        song = pydub_convert(AudioSegment.from_file(file_path))
        cut_files, time_stamps = [], []

        for i, (start_i, end_i) in enumerate(silence.detect_nonsilent(song, min_silence_len=750, silence_thresh=-70)):
            chunk = song[start_i:end_i]

            if len(chunk) > 10:
                chunk_file_path = os.path.join(output_path, f"chunk{i}.wav")
                if os.path.exists(chunk_file_path): os.remove(chunk_file_path)

                chunk.export(chunk_file_path, format="wav")

                cut_files.append(chunk_file_path)
                time_stamps.append((start_i, end_i))
            else: print(f"Part {i} skipped due to short length {len(chunk)}ms")

        print(f"Total cut parts: {len(cut_files)}")
        return cut_files, time_stamps
    except Exception as e:
        raise RuntimeError(f"Error splitting audio file: {e}")

def merge_audio(files_list, time_stamps, original_file_path, output_path, format):
    """Merges audio chunks back into a single file."""
    try:
        def extract_number(filename):
            match = re.search(r"_(\\d+)", filename)
            return int(match.group(1)) if match else 0

        total_duration = len(AudioSegment.from_file(original_file_path))
        combined = AudioSegment.empty() 
        current_position = 0 

        for file, (start_i, end_i) in zip(sorted(files_list, key=extract_number), time_stamps):
            if start_i > current_position: combined += AudioSegment.silent(duration=start_i - current_position)  
            combined += AudioSegment.from_file(file)  
            current_position = end_i

        if current_position < total_duration: combined += AudioSegment.silent(duration=total_duration - current_position)
        combined.export(output_path, format=format)
        return output_path
    except Exception as e:
        raise RuntimeError(f"Error merging audio files: {e}")

def pydub_convert(audio):
    """Converts pydub AudioSegment to numpy array."""
    samples = np.frombuffer(audio.raw_data, dtype=np.int16)
    if samples.dtype != np.int16: samples = (samples * 32767).astype(np.int16)
    return AudioSegment(samples.tobytes(), frame_rate=audio.frame_rate, sample_width=samples.dtype.itemsize, channels=audio.channels)

def run_batch_convert(params):
    """Runs batch conversion for audio segments."""
    path, audio_temp, export_format, cut_files, pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0_method, pth_path, index_path, f0_autotune, f0_autotune_strength, clean_audio, clean_strength, embedder_model, resample_sr = params["path"], params["audio_temp"], params["export_format"], params["cut_files"], params["pitch"], params["filter_radius"], params["index_rate"], params["volume_envelope"], params["protect"], params["hop_length"], params["f0_method"], params["pth_path"], params["index_path"], params["f0_autotune"], params["f0_autotune_strength"], params["clean_audio"], params["clean_strength"], params["embedder_model"], params["resample_sr"]

    segment_output_path = os.path.join(audio_temp, f"output_{cut_files.index(path)}.{export_format}")
    if os.path.exists(segment_output_path): os.remove(segment_output_path)
    
    VoiceConverter().convert_audio(pitch=pitch, filter_radius=filter_radius, index_rate=index_rate, volume_envelope=volume_envelope, protect=protect, hop_length=hop_length, f0_method=f0_method, audio_input_path=path, audio_output_path=segment_output_path, model_path=pth_path, index_path=index_path, f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength, clean_audio=clean_audio, clean_strength=clean_strength, export_format=export_format, embedder_model=embedder_model, resample_sr=resample_sr)
    os.remove(path)

    if os.path.exists(segment_output_path): return segment_output_path
    else: raise FileNotFoundError(f"File not found: {segment_output_path}")

def run_convert_script(pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0_method, input_path, output_path, pth_path, index_path, f0_autotune, f0_autotune_strength, clean_audio, clean_strength, export_format, embedder_model, resample_sr, split_audio):
    """Runs the main audio conversion script."""
    check_predictors(f0_method)
    check_embedders(embedder_model)

    cvt = VoiceConverter()
    start_time = time.time()

    if not pth_path or not os.path.exists(pth_path) or os.path.isdir(pth_path) or not pth_path.endswith((".pth", ".onnx")): raise FileNotFoundError("Invalid model file!")

    processed_segments = []
    audio_temp = os.path.join("audios_temp")
    if not os.path.exists(audio_temp) and split_audio: os.makedirs(audio_temp, exist_ok=True)

    if os.path.isdir(input_path):
        try:
            print(f"Input is a directory, converting all audio files inside.")
            audio_files = [f for f in os.listdir(input_path) if f.lower().endswith(("wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"))]

            if not audio_files: raise FileNotFoundError("No audio files found!")
            print(f"Found {len(audio_files)} audio files.")

            for audio in audio_files:
                audio_path = os.path.join(input_path, audio)
                output_audio = os.path.join(input_path, os.path.splitext(audio)[0] + f"_output.{export_format}")

                if split_audio:
                    try:
                        cut_files, time_stamps = process_audio(audio_path, audio_temp)
                        params_list = [{"path": path, "audio_temp": audio_temp, "export_format": export_format, "cut_files": cut_files, "pitch": pitch, "filter_radius": filter_radius, "index_rate": index_rate, "volume_envelope": volume_envelope, "protect": protect, "hop_length": hop_length, "f0_method": f0_method, "pth_path": pth_path, "index_path": index_path, "f0_autotune": f0_autotune, "f0_autotune_strength": f0_autotune_strength, "clean_audio": clean_audio, "clean_strength": clean_strength, "embedder_model": embedder_model, "resample_sr": resample_sr} for path in cut_files]
                        
                        with tqdm(total=len(params_list), desc="Converting audio", ncols=100, unit="a") as pbar:
                            for params in params_list:
                                results = run_batch_convert(params)
                                processed_segments.append(results)
                                pbar.update(1)

                        merge_audio(processed_segments, time_stamps, audio_path, output_audio, export_format)
                    except Exception as e:
                        raise RuntimeError(f"Error during batch conversion: {e}")
                    finally:
                        if os.path.exists(audio_temp): shutil.rmtree(audio_temp, ignore_errors=True)
                else:
                    try:
                        print(f"Starting conversion for \'{audio_path}\'...")
                        if os.path.exists(output_audio): os.remove(output_audio)

                        with tqdm(total=1, desc="Converting audio", ncols=100, unit="a") as pbar:
                            cvt.convert_audio(pitch=pitch, filter_radius=filter_radius, index_rate=index_rate, volume_envelope=volume_envelope, protect=protect, hop_length=hop_length, f0_method=f0_method, audio_input_path=audio_path, audio_output_path=output_audio, model_path=pth_path, index_path=index_path, f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength, clean_audio=clean_audio, clean_strength=clean_strength, export_format=export_format, embedder_model=embedder_model, resample_sr=resample_sr)
                            pbar.update(1)
                    except Exception as e:
                        raise RuntimeError(f"Error converting file: {e}")

            elapsed_time = time.time() - start_time
            print(f"Batch conversion completed in {elapsed_time:.2f} seconds. Output: {output_path.replace('wav', export_format)}")
        except Exception as e:
            raise RuntimeError(f"Error during batch conversion: {e}")
    else:
        print(f"Starting conversion for \'{input_path}\'...")
        if not os.path.exists(input_path): raise FileExistsError("Input file not found!")
        if os.path.exists(output_path): os.remove(output_path)

        if split_audio:
            try:              
                cut_files, time_stamps = process_audio(input_path, audio_temp)
                params_list = [{"path": path, "audio_temp": audio_temp, "export_format": export_format, "cut_files": cut_files, "pitch": pitch, "filter_radius": filter_radius, "index_rate": index_rate, "volume_envelope": volume_envelope, "protect": protect, "hop_length": hop_length, "f0_method": f0_method, "pth_path": pth_path, "index_path": index_path, "f0_autotune": f0_autotune, "f0_autotune_strength": f0_autotune_strength, "clean_audio": clean_audio, "clean_strength": clean_strength, "embedder_model": embedder_model, "resample_sr": resample_sr} for path in cut_files]
                
                with tqdm(total=len(params_list), desc="Converting audio", ncols=100, unit="a") as pbar:
                    for params in params_list:
                        results = run_batch_convert(params)
                        processed_segments.append(results)
                        pbar.update(1)

                merge_audio(processed_segments, time_stamps, input_path, output_path.replace("wav", export_format), export_format)
            except Exception as e:
                raise RuntimeError(f"Error during batch conversion: {e}")
            finally:
                if os.path.exists(audio_temp): shutil.rmtree(audio_temp, ignore_errors=True)
        else:
            try:
                with tqdm(total=1, desc="Converting audio", ncols=100, unit="a") as pbar:
                    cvt.convert_audio(pitch=pitch, filter_radius=filter_radius, index_rate=index_rate, volume_envelope=volume_envelope, protect=protect, hop_length=hop_length, f0_method=f0_method, audio_input_path=input_path, audio_output_path=output_path, model_path=pth_path, index_path=index_path, f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength, clean_audio=clean_audio, clean_strength=clean_strength, export_format=export_format, embedder_model=embedder_model, resample_sr=resample_sr)
                    pbar.update(1)
            except Exception as e:
                raise RuntimeError(f"Error converting file: {e}")

        elapsed_time = time.time() - start_time
        print(f"Conversion of {input_path} to {output_path.replace('wav', export_format)} completed in {elapsed_time:.2f}s")

def change_rms(source_audio, source_rate, target_audio, target_rate, rate):
    """Adjusts the RMS of the target audio to match the source audio."""
    rms2 = F.interpolate(torch.from_numpy(librosa.feature.rms(y=target_audio, frame_length=target_rate // 2 * 2, hop_length=target_rate // 2)).float().unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze()
    return (target_audio * (torch.pow(F.interpolate(torch.from_numpy(librosa.feature.rms(y=source_audio, frame_length=source_rate // 2 * 2, hop_length=source_rate // 2)).float().unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze(), 1 - rate) * torch.pow(torch.maximum(rms2, torch.zeros_like(rms2) + 1e-6), rate - 1)).numpy())

def get_providers():
    """Returns available ONNX Runtime providers."""
    ort_providers = onnxruntime.get_available_providers()
    providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in ort_providers else ["CPUExecutionProvider"]
    return providers

def device_config(device):
    """Configures the device for ONNX Runtime."""
    if device == "cpu": return onnxruntime.SessionOptions(), ["CPUExecutionProvider"]
    elif device == "cuda":
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        return options, ["CUDAExecutionProvider"]
    else: raise ValueError(f"Unsupported device: {device}")

class Autotune:
    """Autotune F0 values based on a reference frequency list."""
    def __init__(self, ref_freqs):
        self.ref_freqs = ref_freqs

    def autotune_f0(self, f0, strength):
        f0_autotuned = np.copy(f0)
        for i in range(f0.shape[0]):
            if f0[i] > 0:
                closest_freq_idx = np.argmin(np.abs(self.ref_freqs - f0[i]))
                closest_freq = self.ref_freqs[closest_freq_idx]
                f0_autotuned[i] = f0[i] * (1 - strength) + closest_freq * strength
        return f0_autotuned

class VoiceConverter:
    """Main class for voice conversion."""
    def __init__(self):
        self.net_g = None
        self.hps = None
        self.hubert_model = None
        self.onnx_session = None
        self.onnx_options = None
        self.providers = None

    def load_model(self, model_path, device="cpu"):
        """Loads the RVC model (PyTorch or ONNX)."""
        if model_path.endswith(".pth"):
            self.hps = json.load(open("configs/config.json", encoding="utf-8"))
            self.net_g = SynthesizerONNX(
                self.hps["data"]["filter_length"],
                self.hps["data"]["n_channels"],
                self.hps["data"]["inter_channels"],
                self.hps["model"]["hidden_channels"],
                self.hps["model"]["filter_channels"],
                self.hps["model"]["n_heads"],
                self.hps["model"]["n_layers"],
                self.hps["model"]["kernel_size"],
                self.hps["model"]["p_dropout"],
                self.hps["model"]["resblock_kernel_sizes"],
                self.hps["model"]["resblock_dilation_sizes"],
                self.hps["model"]["upsample_rates"],
                self.hps["model"]["upsample_initial_channel"],
                self.hps["model"]["upsample_kernel_sizes"],
                self.hps["model"]["n_speakers"],
                self.hps["model"]["gin_channels"],
                self.hps["model"]["use_f0"],
                self.hps["model"]["vocoder"],
                checkpointing=False
            )
            self.net_g.eval()
            self.net_g.load_state_dict(torch.load(model_path, map_location="cpu")["model"])
            self.net_g.remove_weight_norm()
        elif model_path.endswith(".onnx"):
            self.onnx_options, self.providers = device_config(device)
            self.onnx_session = onnxruntime.InferenceSession(model_path, self.onnx_options, providers=self.providers)
            model_inputs = self.onnx_session.get_inputs()
            self.hps = {"data": {"filter_length": model_inputs[0].shape[2] if len(model_inputs[0].shape) == 3 else model_inputs[0].shape[1]}}
        else:
            raise ValueError("Unsupported model format. Please provide a .pth or .onnx file.")

    def load_hubert(self, embedder_model, device="cpu"):
        """Loads the HuBERT embedder model."""
        if embedder_model == "hubert_base":
            self.hubert_model = torch.hub.load("s3prl/s3prl", "hubert_base")
        elif embedder_model == "contentvec_base":
            self.hubert_model = torch.hub.load("s3prl/s3prl", "contentvec_base")
        elif embedder_model == "japanese_hubert_base":
            self.hubert_model = torch.hub.load("s3prl/s3prl", "japanese_hubert_base")
        elif embedder_model == "korean_hubert_base":
            self.hubert_model = torch.hub.load("s3prl/s3prl", "korean_hubert_base")
        elif embedder_model == "chinese_hubert_base":
            self.hubert_model = torch.hub.load("s3prl/s3prl", "chinese_hubert_base")
        elif embedder_model == "Hidden_Rabbit_last":
            self.hubert_model = torch.hub.load("s3prl/s3prl", "Hidden_Rabbit_last")
        elif embedder_model == "portuguese_hubert_base":
            self.hubert_model = torch.hub.load("s3prl/s3prl", "portuguese_hubert_base")
        else:
            raise ValueError(f"Unsupported embedder model: {embedder_model}")
        self.hubert_model.eval()

    def get_f0(self, audio, f0_method, hop_length, sr):
        """Extracts F0 from audio using the specified method."""
        if f0_method == "crepe":
            f0, pd = predict(audio, sr, hop_length=hop_length, fmin=50, fmax=1100)
            f0 = mean(f0)
        elif f0_method == "crepe-tiny":
            f0, pd = predict(audio, sr, hop_length=hop_length, fmin=50, fmax=1100, model="tiny")
            f0 = mean(f0)
        elif f0_method == "crepe-small":
            f0, pd = predict(audio, sr, hop_length=hop_length, fmin=50, fmax=1100, model="small")
            f0 = mean(f0)
        elif f0_method == "crepe-medium":
            f0, pd = predict(audio, sr, hop_length=hop_length, fmin=50, fmax=1100, model="medium")
            f0 = mean(f0)
        elif f0_method == "crepe-large":
            f0, pd = predict(audio, sr, hop_length=hop_length, fmin=50, fmax=1100, model="large")
            f0 = mean(f0)
        elif f0_method == "crepe-full":
            f0, pd = predict(audio, sr, hop_length=hop_length, fmin=50, fmax=1100, model="full")
            f0 = mean(f0)
        elif f0_method == "harvest":
            f0, t = PYWORLD.harvest(audio, sr, f0_min=50, f0_max=1100, frame_period=hop_length / sr * 1000)
        elif f0_method == "dio":
            f0, t = PYWORLD.dio(audio, sr, f0_min=50, f0_max=1100, frame_period=hop_length / sr * 1000)
        elif f0_method == "rmvpe":
            model = RMVPE("rmvpe.pt", is_half=False, device="cpu")
            f0 = model.infer_from_audio(audio, sr, hop_length=hop_length)
        elif f0_method == "rmvpe-onnx":
            model = RMVPE("rmvpe.onnx", is_half=False, device="cpu", onnx=True)
            f0 = model.infer_from_audio(audio, sr, hop_length=hop_length)
        elif f0_method == "fcpe":
            model = FCPE("fcpe.pt", is_half=False, device="cpu")
            f0 = model.infer_from_audio(audio, sr, hop_length=hop_length)
        elif f0_method == "fcpe-onnx":
            model = FCPE("fcpe.onnx", is_half=False, device="cpu", onnx=True)
            f0 = model.infer_from_audio(audio, sr, hop_length=hop_length)
        elif "hybrid" in f0_method:
            methods_str = re.search("hybrid\[(.+)\]", f0_method)
            if methods_str: methods = [method.strip() for method in methods_str.group(1).split("+")]
            f0_list = []
            for method in methods:
                f0_list.append(self.get_f0(audio, method, hop_length, sr))
            f0 = np.array(f0_list).mean(axis=0)
        else:
            raise ValueError(f"Unsupported f0 method: {f0_method}")

        return f0

    def convert_audio(self, pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0_method, audio_input_path, audio_output_path, model_path, index_path, f0_autotune, f0_autotune_strength, clean_audio, clean_strength, export_format, embedder_model, resample_sr):
        """Converts the input audio using the loaded RVC model."""
        if self.net_g is None and self.onnx_session is None: self.load_model(model_path)
        if self.hubert_model is None: self.load_hubert(embedder_model)

        audio = load_audio(audio_input_path)
        if clean_audio: audio = signal.filtfilt(bh, ah, audio)

        if resample_sr != 0: audio = librosa.resample(audio, orig_sr=16000, target_sr=resample_sr, res_type="soxr_vhq")

        audio_length = len(audio)
        f0 = self.get_f0(audio, f0_method, hop_length, 16000)
        f0 = f0 * 2**(pitch / 12)

        if f0_autotune: f0 = Autotune(ref_freqs=librosa.note_to_hz([
            "C2", "C#2", "D2", "D#2", "E2", "F2", "F#2", "G2", "G#2", "A2", "A#2", "B2",
            "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", "G#3", "A3", "A#3", "B3",
            "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4",
            "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5"
        ])).autotune_f0(f0, f0_autotune_strength)

        f0_min = 50
        f0_max = 1100
        f0[f0 < f0_min] = 0
        f0[f0 > f0_max] = 0

        if protect > 0.5: f0[f0 > 0] = f0[f0 > 0] * 0.9

        audio_torch = torch.from_numpy(audio).float().unsqueeze(0)
        f0_torch = torch.from_numpy(f0).float().unsqueeze(0)

        c = self.get_hubert_features(audio_torch)
        c = F.interpolate(c, size=f0_torch.shape[1], mode="linear", align_corners=False)

        if index_path and os.path.exists(index_path):
            big_npy = np.load(index_path)
            index = faiss.read_index(index_path.replace(".npy", ".index"))
            rate = index_rate

            if rate != 0:
                score, ix = index.search(c.cpu().numpy().astype(np.float32), k=8)
                weight = np.expand_dims(np.power(1 / score, 2), 1).repeat(c.shape[1], axis=1)
                weight = weight / np.sum(weight, axis=2, keepdims=True)
                c = torch.from_numpy(np.sum(big_npy[ix] * weight, axis=2)).transpose(1, 0).unsqueeze(0).to(c.dtype)

        c = c.transpose(1, 2)

        if self.net_g:
            if self.hps["model"]["use_f0"] == 1:
                audio_output = self.net_g(c, f0_torch).cpu().detach().numpy()[0]
            else:
                audio_output = self.net_g(c).cpu().detach().numpy()[0]
        elif self.onnx_session:
            model_inputs = self.onnx_session.get_inputs()
            model_outputs = self.onnx_session.get_outputs()

            if len(model_inputs) == 2:
                audio_output = self.onnx_session.run([model_outputs[0].name], {model_inputs[0].name: c.numpy(), model_inputs[1].name: f0_torch.numpy()})[0][0]
            else:
                audio_output = self.onnx_session.run([model_outputs[0].name], {model_inputs[0].name: c.numpy()})[0][0]
        else:
            raise RuntimeError("No model loaded. Please load a model first.")

        audio_output = change_rms(audio, 16000, audio_output, 16000, volume_envelope)
        sf.write(audio_output_path, audio_output, 16000, format=export_format)

    def get_hubert_features(self, audio):
        """Extracts HuBERT features from audio."""
        with torch.no_grad():
            if self.hubert_model.device != audio.device: self.hubert_model = self.hubert_model.to(audio.device)
            feats = self.hubert_model(audio.squeeze(0))
            feats = feats.transpose(1, 0)
        return feats

from lib.generator.synthesizers import SynthesizerONNX


