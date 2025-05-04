import os
import json
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoFileClip, ImageClip, TextClip, CompositeVideoClip

import json

EMOJI_SIZE = 128

def load_inverted_emoji_index(path="external/emojis.json"):
    with open(path) as f:
        raw = json.load(f)

    inverted = {}
    for emoji, keywords in raw.items():
        for keyword in keywords:
            keyword = keyword.lower()
            inverted.setdefault(keyword, []).append(emoji)
    return inverted

EMOJI_DB = load_inverted_emoji_index()

def get_emojis_for_word(word):
    if len(word) >= 5:
        return EMOJI_DB.get(word.lower(), [])

FONT_PATH = "/System/Library/Fonts/Apple Color Emoji.ttc"
EMOJI_RENDER_DIR = "apple_emojis"
SUBTITLE_FONT = "Bangers"
os.makedirs(EMOJI_RENDER_DIR, exist_ok=True)

def emoji_to_filename(emoji):
    return '-'.join(f"{ord(c):x}" for c in emoji) + ".png"

def render_emoji_to_png(emoji, path, size=EMOJI_SIZE):
    font = ImageFont.truetype(FONT_PATH, size, encoding='unic')
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), emoji, font=font, embedded_color=True)
    image.save(path)

def get_emoji_overlay(video_width, word, start, end, y_position):
    center_x = (video_width // 2) - (EMOJI_SIZE // 2)
    effect_cycle = [
        # Smooth vertical bounce (3Hz)
        {
            "position": lambda t: ("center", y_position + 20 * np.sin(6 * np.pi * t)),
            "scale": lambda t: 1.0
        },
        # Zoom in
        {
            "position": lambda t: (center_x, y_position),
            "scale": lambda t: 0.5 + 0.5 * t
        },
        # Springy decay bounce (damped sine wave)
        {
            "position": lambda t: ("center", y_position + 40 * np.exp(-1.5 * t) * np.sin(6 * np.pi * t)),
            "scale": lambda t: 1.0
        },
        # Pop bounce
        {
            "position": lambda t: (center_x, y_position),
            "scale": lambda t: 1.0 + 0.3 * np.exp(-4 * t) * np.sin(10 * np.pi * t)
        },
        # Slide in from left
        {
            "position": lambda t: (int(video_width * 0.1) + int(video_width * 0.5 * t), y_position),
            "scale": lambda t: 1.0
        },
    ]
    
    emojis = get_emojis_for_word(word.lower())
    if not emojis:
        return None

    emoji = random.choice(emojis)
    emoji_path = path_to_emoji(emoji)
    fx = random.choice(effect_cycle)

    emoji_clip = (ImageClip(emoji_path)
                    .with_start(start)
                    .with_end(end)
                    .with_position(fx["position"])
                    .resized(fx["scale"]))

    print(emoji, fx)
    return emoji_clip

def build_overlays(video_clip, whisperx_json_path):
    overlays = []
    with open(whisperx_json_path, 'r') as f:
        data = json.load(f)

    segments = data.get("segments", [])

    if not segments:
        print("⚠️ No segments found for subtitles.")
        return overlays
    
    for segment in segments:
        for word_info in segment.get("words", []):
            word = word_info["word"]
            start = word_info["start"]
            end = word_info["end"]
            
            safe_y_ratio = 0.78  # visually safe from UI

            safe_width_ratio = 0.9
            safe_height_ratio = 0.08
                        
            caption_height = int(video_clip.h * safe_height_ratio)
            word_clip = TextClip(text=word, 
                                  font="./Bangers-Regular.ttf", 
                                  method='caption', 
                                  size=(int(video_clip.w * safe_width_ratio), caption_height),
                                  horizontal_align="center",
                                  vertical_align="center",
                                  stroke_color="black",
                                  stroke_width=4,
                                  color="white")

            word_clip = word_clip.with_start(start)
            word_clip = word_clip.with_end(end)

            # Convert Y-ratio to pixel position
            y_position = int(video_clip.h * safe_y_ratio)
            word_clip = word_clip.with_position(("center", y_position))

            overlays.append(word_clip)
            
            emoji_end = max(end, start + 1)
            emoji_y_position = y_position - caption_height
            emoji_overlay = get_emoji_overlay(video_clip.w, word.lower(), start, emoji_end, emoji_y_position)
            if emoji_overlay:
                overlays.append(emoji_overlay)

    return overlays

def path_to_emoji(emoji):
    emoji_path = os.path.join(EMOJI_RENDER_DIR, emoji_to_filename(emoji))
    if not os.path.exists(emoji_path):
        render_emoji_to_png(emoji, emoji_path)
    return emoji_path

def compose_video_with_overlays(video_path, whisperx_json_path, output_path):
    video_clip = VideoFileClip(video_path)
    #video_clip = video_clip.with_subclip(0, 5);

    #emoji_clips = build_emoji_overlays(video_clip, whisperx_json_path)
    subtitle_clips = build_overlays(video_clip, whisperx_json_path)

    final = CompositeVideoClip([video_clip] + subtitle_clips)
    dev_mode = True

    final.write_videofile(
        output_path + "/out.mp4",
        codec="libx264" if dev_mode else "h264_videotoolbox",
        audio_codec="aac",
        preset="ultrafast" if dev_mode else "medium",
        fps=12 if dev_mode else 30,
        threads=4
    )