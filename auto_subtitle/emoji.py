import os
import json
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoFileClip, ImageClip, TextClip, CompositeVideoClip

import json

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

def render_emoji_to_png(emoji, path, size=96):
    font = ImageFont.truetype(FONT_PATH, size, encoding='unic')
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), emoji, font=font, embedded_color=True)
    image.save(path)

def get_emoji_overlay(word, start, end, y_position):
    emojis = get_emojis_for_word(word.lower())
    if not emojis:
        return None
    
    effect_cycle = [
        lambda t: ("center", y_position + 20 * np.sin(2 * np.pi * t)),
        lambda t: ("center", y_position + 5 * np.sin(20 * np.pi * t)),
        lambda t: ("center", y_position + 40 * np.exp(-3 * t) * np.sin(10 * np.pi * t)),
        lambda t: (100 + 300 * t, y_position),
    ]
    random.shuffle(effect_cycle)

    effect_index = 0 
    fx = effect_cycle[effect_index]
    effect_index = (effect_index + 1) % len(effect_cycle)

    emoji = random.choice(emojis)
    emoji_path = path_to_emoji(emoji)

    emoji_clip = (ImageClip(emoji_path)
                    .with_start(start)
                    .with_end(end)
                    .with_position(fx))

    return emoji_clip

def build_subtitle_overlays(video_clip, whisperx_json_path):
    subtitles = []
    with open(whisperx_json_path, 'r') as f:
        data = json.load(f)

    segments = data.get("segments", [])

    if not segments:
        print("⚠️ No segments found for subtitles.")
        return subtitles
    
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

            subtitles.append(word_clip)

            overlay = get_emoji_overlay(word.lower(), start, end, y_position - caption_height)
            if overlay:
                subtitles.append(overlay)            
            # if subtitles.__len__() > 5:
            #     return subtitles

    return subtitles

def path_to_emoji(emoji):
    emoji_path = os.path.join(EMOJI_RENDER_DIR, emoji_to_filename(emoji))
    if not os.path.exists(emoji_path):
        render_emoji_to_png(emoji, emoji_path)
    return emoji_path

def compose_video_with_overlays(video_path, whisperx_json_path, output_path):
    video_clip = VideoFileClip(video_path)
    #video_clip = video_clip.with_subclip(0, 5);

    #emoji_clips = build_emoji_overlays(video_clip, whisperx_json_path)
    subtitle_clips = build_subtitle_overlays(video_clip, whisperx_json_path)

    final = CompositeVideoClip([video_clip] + subtitle_clips)
    dev_mode = False

    final.write_videofile(
        output_path + "/out.mp4",
        codec="libx264" if dev_mode else "h264_videotoolbox",
        audio_codec="aac",
        preset="ultrafast" if dev_mode else "medium",
        fps=12 if dev_mode else 30,
        threads=4
    )