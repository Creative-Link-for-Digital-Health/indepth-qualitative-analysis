import os
# Set env var to disable strict loading before importing torch (if possible)
# This is a fallback for libraries that might respect it
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import whisperx
import gc
import torch
import streamlit as st
from whisperx.diarize import DiarizationPipeline
import omegaconf
import typing

# Monkey-patch torch.load to default weights_only=False
# This resolves errors with PyTorch 2.6 breaking changes on legacy checkpoints
if hasattr(torch, 'load'):
    try:
        _original_torch_load = torch.load
        def _safe_torch_load(*args, **kwargs):
            # Check if weights_only is specified. If not, default to False to emulate PyTorch < 2.6
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
                print("DEBUG: torch.load monkey-patch applied: forcing weights_only=False")
            return _original_torch_load(*args, **kwargs)
        torch.load = _safe_torch_load
        print("DEBUG: Successfully monkey-patched torch.load to handle legacy WhisperX checkpoints.")
    except Exception as e:
        print(f"DEBUG: Failed to monkey-patch torch.load: {e}")

# Register safe globals as a secondary measure
torch.serialization.add_safe_globals([
    omegaconf.listconfig.ListConfig, 
    omegaconf.dictconfig.DictConfig,
    omegaconf.base.ContainerMetadata,
    typing.Any,
    list,
    dict,
    set,
])

@st.cache_resource
def load_whisper_model(device="cuda", compute_type="float16", model_size="base"):
    """
    Loads and caches the WhisperX model.
    """
    try:
        print(f"DEBUG: Checking PyTorch environment...")
        print(f"DEBUG: Torch version: {torch.__version__}")
        print(f"DEBUG: CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"DEBUG: CUDA device count: {torch.cuda.device_count()}")
            print(f"DEBUG: Current device: {torch.cuda.current_device()}")
            print(f"DEBUG: Device name: {torch.cuda.get_device_name(0)}")
        else:
            print("DEBUG: No CUDA devices found.")

        # Fallback logic
        if device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
            print("DEBUG: Changed device to 'cpu'")
            
            # float16 is usually not supported on CPU for these models, switch to int8
            if compute_type == "float16":
                print("WARNING: compute_type 'float16' not supported on CPU. Falling back to 'int8'.")
                compute_type = "int8"
        
        print(f"DEBUG: Attempting to load WhisperX model. Size: {model_size}, Device: {device}, Compute: {compute_type}")
        model = whisperx.load_model(model_size, device, compute_type=compute_type)
        print("DEBUG: WhisperX model loaded successfully.")
        return model
    except Exception as e:
        print(f"DEBUG: Failed to load WhisperX model. Error: {e}")
        st.error(f"Error loading WhisperX model: {e}")
        return None

@st.cache_resource
def load_diarization_model(device="cuda", use_auth_token=None):
    """
    Loads and caches the proper speaker diarization pipeline.
    """
    try:
        # Note: 'use_auth_token' might be needed if using pyannote/speaker-diarization directly via HF
        diarize_model = DiarizationPipeline(use_auth_token=use_auth_token, device=device)
        return diarize_model
    except Exception as e:
        st.error(f"Error loading Diarization model: {e}")
        return None

def transcribe_and_diarize(audio_file_path, hf_token=None, model_size="base", device="cuda", batch_size=4, compute_type="float16"):
    """
    Transcribes audio, aligns it, and performs speaker diarization.
    """
    # Validating Trace
    if device == "cuda" and not torch.cuda.is_available():
        print("DEBUG: Force switching device to 'cpu' in pipeline because CUDA is unavailable.")
        device = "cpu"
        if compute_type == "float16":
            compute_type = "int8"
            print("DEBUG: Forced compute_type to 'int8' for CPU compatibility.")
            
    # 1. Transcribe with original whisper
    # 1. Transcribe with original whisper
    print("DEBUG: [Step 1] Loading Whisper model...")
    model = load_whisper_model(device, compute_type, model_size)
    if not model:
        print("DEBUG: Failed to load Whisper model.")
        return None

    print(f"DEBUG: [Step 1] Loading audio file: {audio_file_path}")
    audio = whisperx.load_audio(audio_file_path)
    
    print(f"DEBUG: [Step 1] Transcribing audio (batch_size={batch_size})...")
    result = model.transcribe(audio, batch_size=batch_size, language="en")
    print("DEBUG: [Step 1] Transcription complete.")
    
    # 2. Align whisper output
    print(f"DEBUG: [Step 2] Loading alignment model for language: {result['language']}")
    try:
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        print("DEBUG: [Step 2] Alignment model loaded. Aligning...")
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        print("DEBUG: [Step 2] Alignment complete.")
        
        # Clean up alignment model to free memory
        del model_a
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"DEBUG: Alignment failed: {e}")
        print("DEBUG: Continuing without alignment...")

    # 3. Diarize
    print("DEBUG: [Step 3] Loading diarization model...")
    diarize_model = load_diarization_model(device, use_auth_token=hf_token)
    if not diarize_model:
        st.warning("Diarization model failed to load. Returning transcription without diarization.")
        print("DEBUG: Diarization model skipped/failed.")
        return result

    print("DEBUG: [Step 3] Running diarization...")
    try:
        diarize_segments = diarize_model(audio)
        print("DEBUG: [Step 3] Diarization complete. Assigning speakers...")
        
        # 4. Assign Speaker Labels
        result = whisperx.assign_word_speakers(diarize_segments, result)
        print("DEBUG: [Step 4] Speaker assignment complete.")
    except Exception as e:
        print(f"DEBUG: Diarization execution failed: {e}")
        st.error(f"Diarization failed: {e}")

    return result

def format_diarization_output(result):
    """
    Formats the WhisperX result into a readable string with speaker labels.
    """
    output_text = []
    current_speaker = None
    
    for segment in result["segments"]:
        speaker = segment.get("speaker", "Unknown Speaker")
        text = segment["text"].strip()
        start = segment["start"]
        end = segment["end"]
        
        # Simple formatting: [Speaker] (Time): Text
        time_str = f"[{format_timestamp(start)} - {format_timestamp(end)}]"
        
        output_text.append(f"**{speaker}** {time_str}:\n{text}\n")
            
    return "\n".join(output_text)

def format_timestamp(seconds):
    """Formats seconds into MM:SS"""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
    return f"{int(m):02d}:{int(s):02d}"
