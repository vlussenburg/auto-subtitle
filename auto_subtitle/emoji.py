import os
import json
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoFileClip, ImageClip, CompositeVideoClip

FONT_PATH = "/System/Library/Fonts/Apple Color Emoji.ttc"
EMOJI_RENDER_DIR = "apple_emojis"
os.makedirs(EMOJI_RENDER_DIR, exist_ok=True)

def emoji_to_filename(emoji):
    return '-'.join(f"{ord(c):x}" for c in emoji) + ".png"

def render_emoji_to_png(emoji, path, size=96):
    font = ImageFont.truetype(FONT_PATH, size, encoding='unic')
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), emoji, font=font, embedded_color=True)
    image.save(path)

def add_emoji_overlays(video_path, highlight_json, output_path):
    overlays = []
    video = VideoFileClip(video_path)

    with open(highlight_json, 'r') as f:
        highlights = json.load(f)

    clip_start = highlights[0]["start_time"]

    effect_index = 0
    effect_cycle = [
        lambda t: ("center", 200 + 20 * np.sin(2 * np.pi * t)),  # bounce
        lambda t: ("center", 200 + 5 * np.sin(20 * np.pi * t)),  # jitter
        lambda t: ("center", 200 + 40 * np.exp(-3 * t) * np.sin(10 * np.pi * t)),  # spring
        lambda t: (100 + 300 * t, 150),                          # left to right
    ]
    random.shuffle(effect_cycle)

    for highlight in highlights:
        for e in highlight.get("emojis", []):
            emoji = e["emoji"]
            filename = emoji_to_filename(emoji)
            emoji_path = os.path.join(EMOJI_RENDER_DIR, filename)

            if not os.path.exists(emoji_path):
                print(f"🎨 Rendering {emoji} to {emoji_path}")
                render_emoji_to_png(emoji, emoji_path)

            start = e["start_time"] - clip_start + 2
            end = e["end_time"] - clip_start + 2

            fx = effect_cycle[effect_index]
            effect_index += 1
            if effect_index >= len(effect_cycle):
                effect_index = 0
                random.shuffle(effect_cycle)

            emoji_clip = (ImageClip(emoji_path)
                          .with_start(start)
                          .with_end(end)
                          .with_position(fx))
            overlays.append(emoji_clip)

    final = CompositeVideoClip([video] + overlays)
    final.write_videofile(output_path, codec="h264_videotoolbox", audio_codec="aac", preset="ultrafast")