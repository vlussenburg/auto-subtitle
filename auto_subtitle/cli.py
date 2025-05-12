import os
import argparse
import openai
import tempfile
from .utils import filename, str2bool, generate_whisperx_json, get_audio, center_crop_to_9x16 
from moviepy import VideoFileClip, ImageClip, TextClip, CompositeVideoClip, concatenate_videoclips
from .emoji import build_overlays
from dotenv import load_dotenv

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", type=str, nargs='+',
                        help="paths to video files to transcribe (provide a list of video paths)")
    parser.add_argument("--model", default="small",
                        help="name of the Whisper model to use")
    parser.add_argument("--output_dir", "-o", type=str,
                        default="subtitled", help="directory to save the outputs")
    parser.add_argument("--output_srt", type=str2bool, default=False,
                        help="whether to output the .srt file along with the video files")
    parser.add_argument("--srt_only", type=str2bool, default=False,
                        help="only generate the .srt file and not create overlayed video")
    parser.add_argument("--verbose", type=str2bool, default=False,
                        help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=[
                        "transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default="auto", choices=["auto","af","am","ar","as","az","ba","be","bg","bn","bo","br","bs","ca","cs","cy","da","de","el","en","es","et","eu","fa","fi","fo","fr","gl","gu","ha","haw","he","hi","hr","ht","hu","hy","id","is","it","ja","jw","ka","kk","km","kn","ko","la","lb","ln","lo","lt","lv","mg","mi","mk","ml","mn","mr","ms","mt","my","ne","nl","nn","no","oc","pa","pl","ps","pt","ro","ru","sa","sd","si","sk","sl","sn","so","sq","sr","su","sv","sw","ta","te","tg","th","tk","tl","tr","tt","uk","ur","uz","vi","yi","yo","zh"], 
    help="What is the origin language of the video? If unset, it is detected automatically.")

    load_dotenv()
    
    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    output_srt: bool = args.pop("output_srt")
    srt_only: bool = args.pop("srt_only")
    language: str = args.pop("language")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("work", exist_ok=True)

    video_paths = args.pop("video")
    clips = []
    for idx, video_path in enumerate(video_paths):
        video_clip = VideoFileClip(video_path)
        video_clip = center_crop_to_9x16(video_clip)
        
        audio_path = get_audio(video_path, "work")
        whisperx_json_path = generate_whisperx_json(audio_path, "work")
        subtitle_clips = build_overlays(video_clip, whisperx_json_path)

        clip = CompositeVideoClip([video_clip] + subtitle_clips)
        clips.append(clip)
        
    final = concatenate_videoclips(clips)
    
    output_file = os.path.join(output_dir, f"{filename(video_path)}.mp4")
    dev_mode = False
    if dev_mode:
        final.write_videofile(
            output_file,
            fps=12,
            bitrate="800k",         # Lower bitrate for faster rendering
            preset="ultrafast",     # ffmpeg preset (requires ffmpeg installed)
            audio_codec="aac",      # ensure audio is included
            codec="h264_videotoolbox",        # Use H.264 codec for video
        )
    else:
        final.write_videofile(
            output_file,
            fps=30,
            audio_codec="aac"
        )

if __name__ == '__main__':
    main()
