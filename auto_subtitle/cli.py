import os
import argparse
import openai
import tempfile
from .utils import filename, str2bool, generate_whisperx_json, get_audio
from .emoji import compose_video_with_overlays


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", type=str,
                        help="path to video files to transcribe")
    parser.add_argument("--model", default="small",
                        help="name of the Whisper model to use")
    parser.add_argument("--output_dir", "-o", type=str,
                        default=".", help="directory to save the outputs")
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

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    output_srt: bool = args.pop("output_srt")
    srt_only: bool = args.pop("srt_only")
    language: str = args.pop("language")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("work", exist_ok=True)

    # if model_name.endswith(".en"):
    #     warnings.warn(
    #         f"{model_name} is an English-only model, forcing English detection.")
    #     args["language"] = "en"
    # # if translate task used and language argument is set, then use it
    # elif language != "auto":
    #     args["language"] = language
        
    # model = stable_whisper.load_model(model_name)
    video_path = args.pop("video")
    audio_path = get_audio(video_path, "work")
    whisperx_json_path = generate_whisperx_json(audio_path, "work")

    #modified_subtitles = modify_subtitles(subtitles, output_srt, output_dir)

    # for path, ass_path in subtitles.items():
    #     out_path = os.path.join(output_dir, f"{filename(path)}.mp4")

    #     print(f"Adding subtitles to {filename(path)}...")

    #     video = ffmpeg.input(path)
    #     audio = video.audio

    #     ffmpeg.concat(
    #         video.filter('subtitles', ass_path), audio, v=1, a=1
    #     ).output(out_path, vcodec="h264_videotoolbox").run(quiet=True, overwrite_output=True)

    #     print(f"Saved subtitled video to {os.path.abspath(out_path)}.")

    #     # Add animated emoji overlays if highlight JSON exists

    #     emoji_output_path = os.path.join(output_dir, f"{filename(path)}_emoji.mp4")
    #     print(f"Adding emoji overlays to {filename(path)}...")
    compose_video_with_overlays(video_path, whisperx_json_path, "subitled")
    #     print(f"✅ Saved emoji video to {emoji_output_path}")

def modify_subtitles(subtitles, output_srt, output_dir):

    client = openai.OpenAI(api_key="")
    modified_subtitles = {}
    srt_path = output_dir if output_srt else tempfile.gettempdir()

    for path, ass_path in subtitles.items():
        with open(ass_path, "r", encoding="utf-8") as file:
            subtitle_text = file.read()

            print(
                f"Modifying subtitles with gpt-4o for {filename(ass_path)}... This might take a while."
            )

            prompt = f"""
                You are an expert in subtitle styling for viral social media content. 

                I need you to modify the following subtitle transcript into a **TikTok-style subtitle file** in **ASS (Advanced SubStation Alpha) format**. Follow these exact specifications:

                1️⃣ **FONT & OUTLINE**  
                - Use a **bold, comic-style font** (e.g., Impact or Arial Black).  
                - All text should have a **thick black outline** for visibility.  

                2️⃣ **COLOR HIGHLIGHTING**  
                - Most words should be **white with a black outline**.  
                - Highlight **important words** (e.g., emotional, power words) in **red**.  

                3️⃣ **POSITIONING**  
                - Place the subtitles **near the speaker's mouth**, **centered horizontally**.  
                - Adjust positioning slightly per sentence for a natural effect.  

                4️⃣ **ANIMATION EFFECTS**  
                - Use the `\\t()` tag in ASS to make key words **pop** (slight scaling effect).  
                - Add subtle `\\move()` effects where necessary.  

                5️⃣ **EMOJIS**  
                - Add a **relevant emoji every few sentences** to match the tone.  
                - Example: If the sentence is about thinking, add 🧠.  
                - If it's exciting, add 🚀.  

                6️⃣ **OUTPUT FORMAT**  
                - The output should be a **fully formatted .ass file** with proper `[Script Info]`, `[V4+ Styles]`, and `[Events]` sections.  
                - **Return only the .ass file content**, no explanations, no markdown, no backticks.

                Here is the transcript:  
                ```{subtitle_text}```
            """

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )

            # Save the modified subtitles
            out_file = os.path.join(srt_path, f"{filename(path)}_modified.ass")
            with open(out_file, "w", encoding="utf-8") as file:
                file.write(response.choices[0].message.content.strip())

            print(f"Saved modified subtitle to {os.path.abspath(out_file)}.")

            modified_subtitles[path] = out_file

    return modified_subtitles

# def get_subtitles(audio_paths: list, output_srt: bool, output_dir: str, model: stable_whisper):
#     subtitles_path = {}

#     for path, audio_path in audio_paths.items():
#         srt_path = output_dir if output_srt else tempfile.gettempdir()
#         #srt_file = os.path.join(srt_path, f"{filename(path)}.srt")
#         ass_file = os.path.join(srt_path, f"{filename(path)}.ass")
        
#         print(
#             f"Generating subtitles for {filename(path)}... This might take a while."
#         )

#         warnings.filterwarnings("ignore")
#         transcribe = model.transcribe(audio_path, regroup=True, fp16=torch.cuda.is_available())

#         # **Split subtitles naturally for TikTok style**
#         transcribe.split_by_gap(0.5)  # Split when there's a 0.5s silence
#         transcribe.split_by_length(max_words=1)
#         #transcribe.merge_by_gap(0.2, max_words=2)

#         #transcribe.to_srt_vtt(str(srt_file), word_level=True)
        
#         # Save the SSA file with styling
#         transcribe.to_ass(
#             ass_file,
#             word_level=True,
#             primary_color="FFFFFF",
#             secondary_color="FFFFFF",
#             highlight_color="0000FF",
#             font="Bangers",
#             font_size=28,
#             border_style=1,
#             outline=1,
#             shadow=1,
#             Alignment=5,
#         )

#         warnings.filterwarnings("default")

#         # with open(srt_file, "w", encoding="utf-8") as srt:
#         #     write_srt(result["segments"], file=srt)

#         subtitles_path[path] = ass_file

#     return subtitles_path

if __name__ == '__main__':
    main()
