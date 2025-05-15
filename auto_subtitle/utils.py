import ffmpeg
from moviepy import VideoClip
import whisperx
import json
import tempfile

import os
import requests
from typing import Iterator, TextIO
from .face_tracking import FacePoint



def center_crop_to_aspect_ratio(clip: VideoClip, target_w: int, target_h: int) -> VideoClip:
    if clip.aspect_ratio == target_w / target_h:
        return clip
    
    from moviepy.video.fx import Crop
    
    w, h = clip.size

    if w > target_w:
        # Too wide → crop width
        x1 = (w - target_w) // 2
        return Crop(x1=x1, width=target_w).apply(clip)
    else:
        # Already in target aspect ratio
        return clip

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

def generate_and_write_whisperx_json(audio_path, output_json_path="work", model_size="small.en"):
    output_file = os.path.join(output_json_path, f"{filename(audio_path)}.json")
    if os.path.exists(output_file):
        print(f"WhisperX JSON already exists at {output_file}, reusing it.")
        return output_file
    
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading WhisperX model {model_size} on {device}...")
    model = whisperx.load_model(model_size, device=device, compute_type="float32")

    print(f"Transcribing {audio_path}...")
    result = model.transcribe(audio_path)

    print("Aligning words for word-level timestamps...")
    alignment_model, metadata = whisperx.load_align_model(language_code="en", device=device)
    aligned_result = whisperx.align(result["segments"], alignment_model, metadata, audio_path, device=device)
    
    for segment in aligned_result["segments"]:
        add_broll_score(segment["text"], json_segment=segment)

    print(f"Saving augmented output to {output_file}...")
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

def is_vertical(clip):
    return clip.h > clip.w

def generate_b_roll_image(prompt: str, output_path: str, vertical: bool = True):
    from openai import OpenAI
    client = OpenAI()

    try:
        print(f"🎨 Generating B-roll for: {prompt[:80]}...")
        orientation = "portrait" if vertical else "landscape"
        response = client.images.generate(
            model="dall-e-3",
            prompt=(
                f"{orientation.capitalize()} orientation. "
                f"A high-quality image for use as B-roll in a video about serious topics like mental health, philosophy, and psychology. "
                f"No text. If humans or animals are shown, ensure anatomical correctness (no extra or missing limbs). "
                f"Visual tone: introspective, cinematic, grounded. "
                f"{prompt}"
                ),
            #prompt=f"An image in {orientation} orientation for use as B-roll in a video for a channel focusing on serious content around mental health, philosophy and psychology, avoiding written text. If portraying humans or animals, make sure they are anotomically correct without extra or missing limbs. Prompt: {prompt}",
            n=1,
            size="1024x1792" if vertical else "1792x1024",
            quality="standard",
            response_format="url",
        )
        image_url = response.data[0].url
        img_data = requests.get(image_url).content
        with open(output_path, "wb") as f:
            f.write(img_data)
        print(f"✅ Saved B-roll to {output_path}")
    except Exception as e:
        print(f"❌ Failed to generate B-roll: {e}")

def add_broll_score(prompt: str, json_segment: dict = None) -> bool:
    system_prompt = (
        "You are a helpful assistant. Rate how visually suitable the following sentence is "
        "for a single cinematic image, on a scale from 0 (not visual at all) to 10 (very visual). "
        "Only return the number."
    )
    from openai import OpenAI
    client = OpenAI()

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        score_text = response.choices[0].message.content.strip()
        score = int(score_text)

        if json_segment is not None:
            json_segment["b_roll_score"] = score
    except Exception as e:
        print(f"⚠️ Could not score B-roll suitability: {e}")
        return False
