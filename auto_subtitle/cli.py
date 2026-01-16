import os
import json
import argparse
from .utils import *
from moviepy import VideoFileClip, CompositeVideoClip, concatenate_videoclips
from .emoji import build_overlays
from dotenv import load_dotenv
import numpy as np
from .face_tracking import track_face_centers


def load_emoji_map(video_path):
    """Load emoji_map from sidecar .meta.json file if it exists."""
    base, _ = os.path.splitext(video_path)
    sidecar = base + '.meta.json'

    if os.path.exists(sidecar):
        with open(sidecar) as f:
            data = json.load(f)
            emoji_map = data.get("emoji_map", {})
            if emoji_map:
                print(f"Loaded emoji_map from {sidecar}: {emoji_map}")
            return emoji_map
    return {}

OUTPUT_DIR = ""
WORK_DIR = "work"

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", type=str, nargs='+',
                        help="paths to video files to transcribe (provide a list of video paths)")
    parser.add_argument("--model", default="small",
                        help="name of the Whisper model to use")
    parser.add_argument("--output_dir", "-o", type=str,
                        default="subtitled", help="directory to save the outputs")
    parser.add_argument("--verbose", type=str2bool, default=False,
                        help="whether to print out the progress and debug messages")
    parser.add_argument("--skip-vertical", action="store_true",
                        help="whether to skip output in 9x16 aspect ratio")
    parser.add_argument("--skip-horizontal", action="store_true",
                        help="whether to skip output in 16x9 aspect ratio")
    parser.add_argument("--task", type=str, default="transcribe", choices=[
                        "transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")

    load_dotenv()

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    OUTPUT_DIR = args.pop("output_dir")
    skip_vertical: bool = args.pop("skip_vertical")
    skip_horizontal: bool = args.pop("skip_horizontal")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)

    video_paths = args.pop("video")

    if not skip_vertical:
        create_subtitled_video(video_paths, False)
    if not skip_horizontal:
        create_subtitled_video(video_paths, True)

def create_subtitled_video(video_paths, horizontal: bool):
    dev_mode = False
    aspect_str = "9x16" if not horizontal else "16x9"
    output_file = f"{filename(video_paths[-1])}_{aspect_str}.mp4"

    if not dev_mode and os.path.exists(output_file):
        return

    clips = []
    for idx, video_path in enumerate(video_paths):
        video_clip = VideoFileClip(video_path)

        orig_video_w, orig_video_h = video_clip.size
        clip_fps = video_clip.fps
        target_w, target_h = (1920, 1080) if horizontal else (1080, 1920)
        scale = target_h / orig_video_h
        video_clip = video_clip.resized(height=target_h)
        scaled_w, scaled_h = int(orig_video_w * scale), int(orig_video_h * scale)

        face_points = track_face_centers(video_path, work_dir=WORK_DIR)

        # this reference is needed in the make_cropped_frame def
        resized_clip = video_clip.resized(height=target_h)

        # Build a frame function that crops around the (scaled) face position
        def make_frame(t):
            frame = resized_clip.get_frame(t)  # HxWx3 uint8 (RGB)

            # Pick the face point for this timestamp
            i = min(int(t * clip_fps), len(face_points) - 1)
            face_x, face_y = face_points[i].x, face_points[i].y

            # Scale to the resized/working space
            fx = face_x * scale
            fy = face_y * scale

            # Top-left crop corner (clamped to valid range)
            cx = int(np.clip(fx - (target_w // 2), 0, scaled_w - target_w))
            cy = int(np.clip(fy - (target_h // 2), 0, scaled_h - target_h))

            # Return the cropped frame
            return frame[cy:cy + target_h, cx:cx + target_w]

        # Create a new clip from the frame function
        cropped_clip = (
            VideoClip(make_frame, duration=resized_clip.duration)
            .with_fps(clip_fps)                   # or resized_clip.fps
            .with_audio(resized_clip.audio)       # keep original audio
        )

        # Replace/assign if your code expects `video_clip` to be the result
        video_clip = cropped_clip

        audio_path = get_audio(video_path, WORK_DIR)
        whisperx_json_path = generate_and_write_whisperx_json(audio_path, WORK_DIR)

        # Load emoji_map from sidecar if available
        emoji_map = load_emoji_map(video_path)

        # Captions
        subtitle_clips = build_overlays(video_clip, whisperx_json_path, emoji_map)

        clip = CompositeVideoClip([video_clip] + subtitle_clips)
        clips.append(clip)

    final = concatenate_videoclips(clips)

    if dev_mode:
        final.write_videofile(
            os.path.join(OUTPUT_DIR, output_file),
            fps=12,
            bitrate="800k",         # Lower bitrate for faster rendering
            preset="ultrafast",     # ffmpeg preset (requires ffmpeg installed)
            audio_codec="aac",      # ensure audio is included
            codec="h264_videotoolbox",        # Use H.264 codec for video
            ffmpeg_params=["-movflags", "+faststart"],
        )
    else:
        final.write_videofile(
            os.path.join(OUTPUT_DIR, output_file),
            fps=30,
            audio_codec="aac",
            ffmpeg_params=["-movflags", "+faststart"],
        )

if __name__ == '__main__':
    main()
