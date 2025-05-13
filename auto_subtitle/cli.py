import os
import argparse
from .utils import *
from moviepy import VideoFileClip, CompositeVideoClip, concatenate_videoclips
from .emoji import build_overlays
from dotenv import load_dotenv
from .face_tracking import track_face_centers

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
        create_subtitled_video(video_paths, True)
    if not skip_horizontal:
        create_subtitled_video(video_paths, False)

def create_subtitled_video(video_paths, horizontal: bool):
    clips = []
    for idx, video_path in enumerate(video_paths):
        video_clip = VideoFileClip(video_path)
        
        video_w, video_h = video_clip.size
        clip_fps = video_clip.fps
        target_w, target_h = (1920, 1080) if horizontal else (1080, 1920)
        video_clip = center_crop_to_aspect_ratio(video_clip, target_w / target_h)
        
        face_points = track_face_centers(video_path, work_dir=WORK_DIR)

        def pos(t):
            i = min(int(t * clip_fps), len(face_points) - 1)
            face_x, face_y = face_points[i].x, face_points[i].y
            
            # how far to pan the frame so the face ends up centered in target
            cx = int(face_x - target_w // 2)
            cy = int(face_y - target_h // 2)

            # clamp to valid bounds (can't show black edges)
            cx = min(max(cx, 0), video_w - target_w)
            cy = min(max(cy, 0), video_h - target_h)

            # position to apply to full-size video
            return (-cx, -cy)
        
        #video_clip = video_clip.with_position(pos)
        #video_clip = video_clip.resized(width=target_w, height=target_h)

        audio_path = get_audio(video_path, WORK_DIR)
        whisperx_json_path = generate_and_write_whisperx_json(audio_path, WORK_DIR)
        
        # Captions
        subtitle_clips = build_overlays(video_clip, whisperx_json_path)

        clip = CompositeVideoClip([video_clip] + subtitle_clips)
        clips.append(clip)
    
    final = concatenate_videoclips(clips)
    
    aspect_str = "9x16" if not horizontal else "16x9"
    output_file = f"{filename(video_paths[-1])}_{aspect_str}.mp4"
    dev_mode = True
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
