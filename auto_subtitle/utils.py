import ffmpeg
import whisperx
import torch
import json
import tempfile
import os
from typing import Iterator, TextIO

def get_audio(path, output_path=tempfile.gettempdir()):
    print(f"Extracting audio from {filename(path)}...")
    output_file = os.path.join(output_path, f"{filename(path)}.wav")

    if os.path.exists(output_file):
        print(f"Audio already extracted at {output_file}, reusing it.")
        return output_file

    ffmpeg.input(path).output(
        output_file,
        acodec="pcm_s16le", ac=1, ar="16k"
    ).run(quiet=True, overwrite_output=True)

    return output_file

def generate_whisperx_json(audio_path, output_json_path="work", model_size="small.en"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_file = os.path.join(output_json_path, f"{filename(audio_path)}.json")
    if os.path.exists(output_file):
        print(f"WhisperX JSON already exists at {output_file}, reusing it.")
        return output_file

    print(f"Loading WhisperX model {model_size} on {device}...")
    model = whisperx.load_model(model_size, device=device, compute_type="float32")

    print(f"Transcribing {audio_path}...")
    result = model.transcribe(audio_path)

    print("Aligning words for word-level timestamps...")
    alignment_model, metadata = whisperx.load_align_model(language_code="en", device=device)
    aligned_result = whisperx.align(result["segments"], alignment_model, metadata, audio_path, device=device)

    print(f"Saving aligned output to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(aligned_result, f, ensure_ascii=False, indent=2)

    print("✅ WhisperX JSON generated successfully.")
    return output_file

def str2bool(string):
    string = string.lower()
    str2val = {"true": True, "false": False}

    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(
            f"Expected one of {set(str2val.keys())}, got {string}")


def format_timestamp(seconds: float, always_include_hours: bool = False):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def write_srt(transcript: Iterator[dict], file: TextIO):
    for i, segment in enumerate(transcript, start=1):
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start'], always_include_hours=True)} --> "
            f"{format_timestamp(segment['end'], always_include_hours=True)}\n"
            f"{segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )


def filename(path):
    return os.path.splitext(os.path.basename(path))[0]