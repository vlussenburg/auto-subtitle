import os
import argparse
import openai
import tempfile
from .utils import filename, str2bool, generate_whisperx_json, get_audio
from .emoji import compose_video_with_overlays
from dotenv import load_dotenv

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

    load_dotenv()
    
    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    output_srt: bool = args.pop("output_srt")
    srt_only: bool = args.pop("srt_only")
    language: str = args.pop("language")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("work", exist_ok=True)

    video_path = args.pop("video")
    audio_path = get_audio(video_path, "work")
    whisperx_json_path = generate_whisperx_json(audio_path, "work")

    compose_video_with_overlays(video_path, whisperx_json_path, "subtitled")

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

if __name__ == '__main__':
    main()
