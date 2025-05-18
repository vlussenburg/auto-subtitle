import os
import math
import json
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoFileClip, ImageClip, TextClip, CompositeVideoClip
from slugify import slugify
from .utils import *
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

def render_emoji_to_png(emoji, video_clip, path):
    size = get_emoji_size(video_clip)
    font = ImageFont.truetype(FONT_PATH, size, encoding='unic')
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), emoji, font=font, embedded_color=True)
    image.save(path)

def get_emoji_size(video_clip, scale=0.05):
    preferred_sizes = [32, 64, 96, 160, 256]
    target = video_clip.h * scale
    size = min(preferred_sizes, key=lambda s: abs(s - target))
    return size

def get_emoji_overlay(video_clip, word, start, end, y_position):
    center_x = (video_clip.w // 2) - (get_emoji_size(video_clip) // 2)
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
            "position": lambda t: (int(video_clip.w * 0.1) + int(video_clip.w * 0.5 * t), y_position),
            "scale": lambda t: 1.0
        },
    ]
    
    emojis = get_emojis_for_word(word.lower())
    if not emojis:
        return None

    emoji = random.choice(emojis)
    emoji_path = path_to_emoji(emoji, video_clip)
    fx = random.choice(effect_cycle)

    emoji_clip = (ImageClip(emoji_path)
                    .with_start(start)
                    .with_end(end)
                    .with_position(fx["position"])
                    .resized(fx["scale"]))
    return emoji_clip
    
def generate_b_roll_overlay(image_path, start, end, video_size):
    duration = min(max(end - start, 3), 5)
    img_clip = ImageClip(image_path, duration=duration)
    img_w, img_h = img_clip.size
    target_w, target_h = video_size
    
    # Compute scale to cover frame, whether 9:16 or 16:9
    target_aspect = target_w / target_h
    img_aspect = img_w / img_h
    
    if img_aspect > target_aspect:
        # Image is wider → scale based on height
        scale_to_cover = target_h / img_h
    else:
        # Image is taller/narrower → scale based on width
        scale_to_cover = target_w / img_w

    # Compute initial scale so the image is big enough for zooming
    zoom_margin = 1.25
    base_scale = scale_to_cover * zoom_margin

    def zoom(t):
        return base_scale * (1 + 0.20 * (t / duration))

    return (
        img_clip
        .resized(lambda t: zoom(t))
        .with_position("center")  # safe as long as image always covers frame
        .with_start(start)
        .with_duration(duration)
        .with_opacity(1)
    )

def build_overlays(video_clip, whisperx_json_path):
    overlays = []
    with open(whisperx_json_path, 'r') as f:
        data = json.load(f)

    segments = data.get("segments", [])
    
    if not segments:
        print("⚠️ No segments found for subtitles.")
        return overlays
    
    broll_overlay = find_broll_segment_and_generate_broll_overlay(video_clip, segments)
    if broll_overlay: overlays.extend(broll_overlay)
    
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
            emoji_overlay = get_emoji_overlay(video_clip, word.lower(), start, emoji_end, emoji_y_position)
            if emoji_overlay:
                overlays.append(emoji_overlay)

    return overlays

def find_broll_segment_and_generate_broll_overlay(video_clip, segments):
    # Calculate how many segments to keep
    video_duration = video_clip.duration  # in seconds
    num_top_segments = int(math.ceil(video_duration / 30))  # 1 per 30s
    
    eligible_segments = [s for s in segments if s.get("b_roll_score", 0) >= 8 and s.get("b_roll_prompt") and s.get("emotional_tone")]
    
    # Sort and take top X
    top_segments = sorted(
        eligible_segments, key=lambda s: s["b_roll_score"], reverse=True
    )[:num_top_segments]
    
    overlays = []
    for segment in top_segments:
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        aspect_str = "9x16" if is_vertical(video_clip) else "16x9"
        image_path = "broll_images/" + slugify(segment["b_roll_prompt"][:100]) + "_" + aspect_str + ".png"
        if not os.path.exists(image_path):
            generate_b_roll_image(segment["b_roll_prompt"], image_path, is_vertical(video_clip))

        if os.path.exists(image_path):
            overlays.append(generate_b_roll_overlay(image_path, start, end, (video_clip.w, video_clip.h)))

    return overlays

def path_to_emoji(emoji, video_clip):
    emoji_path = os.path.join(EMOJI_RENDER_DIR, emoji_to_filename(emoji))
    if not os.path.exists(emoji_path):
        render_emoji_to_png(emoji, video_clip, emoji_path)
    return emoji_path