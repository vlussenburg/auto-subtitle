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
    
    if not os.path.exists(output_file):    
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading WhisperX model {model_size} on {device}...")
        model = whisperx.load_model(model_size, device=device, compute_type="float32")

        print(f"Transcribing {audio_path}...")
        result = model.transcribe(audio_path)

        aligned_result = align_words(audio_path, result, device=device)
    else:
        print(f"WhisperX JSON already exists at {output_file}, reusing it.")
        aligned_result = json.load(open(output_file, "r"))
    
    if not all("b_roll_score" in segment for segment in aligned_result.get("segments", [])):
        for segment in aligned_result["segments"]:
            print(f"Scoring B-roll suitability for segment: {segment['text'][:80]}...")
            answer = determine_broll_score(segment["text"])
            print(f"Answer: {answer}")
            segment["b_roll_score"] =  answer.get("score", 0)
            segment["b_roll_prompt"] = answer.get("prompt", None)
            segment["emotional_tone"] = answer.get("emotional_tone", None)
    else:
        print("✅ B-roll scores already added to all segments, skipping scoring.")

    print(f"Saving augmented output to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(aligned_result, f, ensure_ascii=False, indent=2)

    print("✅ WhisperX JSON generated successfully.")
    return output_file


def align_words(audio_path, result, device="cpu"):
    print("Aligning words for word-level timestamps...")
    alignment_model, metadata = whisperx.load_align_model(language_code="en", device=device)
    aligned_result = whisperx.align(result["segments"], alignment_model, metadata, audio_path, device=device)
    return aligned_result

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
        size = "1024x1792" if vertical else "1792x1024"
        response = client.images.generate(
            model="dall-e-3",
            prompt = (
                f"{orientation.capitalize()} orientation.\n"
                "Generate a cinematic-quality B-roll image for a video on topics like mental health, philosophy, or personal growth.\n"
                "The image should be composed specifically for a vertical portrait frame." if vertical else "The image should be composed specifically for a wide landscape frame." + "\n"
                "Avoid text, symbols, or watermarks.\n"
                "If humans or animals are depicted, ensure anatomical correctness (no extra or missing limbs).\n"
                "The style should be introspective, grounded, and visually rich.\n"
                f"Scene description: {prompt}"
            ),
            n=1,
            size=size,
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

def determine_broll_score(segment_text: str) -> tuple[int, str | None]:
    system_prompt = """
    You are a visual assistant trained to select B-roll and supplemental imagery to enhance podcast or YouTube content. 

    Your job is to evaluate whether a given segment can be represented effectively by a single cinematic image. If so, suggest a strong, metaphorical or thematic visual — not a literal one.

    Focus on abstract or emotionally resonant ideas. Avoid clichés and generic imagery. Use simple, evocative phrases like: “waves crashing,” “man standing in a doorway,” or “fog rolling over mountains.”

    Never suggest text overlays. Assume the final use is silent B-roll under voiceover.
    """
    
    user_prompt = f"""
    Evaluate the following transcript segment and rate how suitable it is for representing with a single cinematic B-roll image.

    Return a JSON object with:
    - a key `"score"` (an integer from 0 to 10), indicating how visual the moment is.
    - a key `"emotional_tone"` (a single word, lower case) that is null or captures the emotional tone of the moment, adhering strictly Plutchik’s Wheel of Emotions ("joy", "sadness", "anticipation", "fear", "anger", "disgust", "surprise", "trust"). Don't use any other words or phrases.
    - a key `"prompt"` only if the score is 8 or above — this should describe a cinematic visual metaphor for the moment, not a literal rephrasing.

    Segment:
    `{segment_text}`

    Examples:
    {{ "score": 8, "emotional_tone": "anticipation", "prompt": "A person journaling in a quiet forest clearing, with sunlight breaking through the trees." }},
    {{ "score": 3, "emotional_tone": null, "prompt": null }}
    """

    # In the determine_broll_score function, replace the previous selection with:
    return json.loads(ask_openai(system_prompt, user_prompt))

def ask_openai(system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI()
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            answer = response.choices[0].message.content.strip()
            return answer
        except Exception as e:
            print(f"⚠️ Error: {e}")
            return None