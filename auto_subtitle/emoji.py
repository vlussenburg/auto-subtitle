import os
import json
import numpy as np
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip

TWEMOJI_PATH = "external/twemoji/assets/72x72"

def emoji_to_filename(emoji):
    return '-'.join(f"{ord(c):x}" for c in emoji) + ".png"

def add_emoji_overlays(video_path, highlight_json, output_path):
    overlays = []
    video = VideoFileClip(video_path)

    with open(highlight_json, 'r') as f:
        highlights = json.load(f)

    for highlight in highlights:
        segment = highlight["segment"]
        for e in segment.get("emojis", []):
            emoji = e["emoji"]
            filename = emoji_to_filename(emoji)
            emoji_path = os.path.join(TWEMOJI_PATH, filename)

            if not os.path.exists(emoji_path):
                print(f"❌ Skipping emoji {emoji}: file not found at {emoji_path}")
                continue

            start = e["start_time"]
            end = e["end_time"]
            duration = end - start

            # Animate position or bounce
            emoji_clip = (ImageClip(emoji_path)
                          .set_start(start)
                          .set_duration(duration)
                          .resize(0.15)
                          .set_position(lambda t: ("center", 200 + 20 * np.sin(2 * np.pi * t))))
            overlays.append(emoji_clip)

    final = CompositeVideoClip([video] + overlays)
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")