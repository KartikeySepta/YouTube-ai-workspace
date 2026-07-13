import argparse
import json
import os
from dotenv import load_dotenv
from typing import Any

from yt_dlp import YoutubeDL


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def extract_audio_and_metadata(video_url: str, output_name: str = "temp_stream") -> tuple[str, dict[str, Any]]:
    """Download the audio stream and extract expanded YouTube metadata."""
    print(f"📡 Extracting audio and expanded metadata from: {video_url}")
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
        "outtmpl": output_name,
        "quiet": True,
        "no_warnings": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        # extract_info with download=True gets the metadata dict AND triggers the download
        info = ydl.extract_info(video_url, download=True)

    expected_file = f"{output_name}.mp3"
    if not os.path.exists(expected_file):
        raise FileNotFoundError("Audio extraction failed. Check your FFmpeg path configuration.")

    metadata = {
        "video_id": info.get("id"),
        "title": info.get("title"),
        "channel": info.get("uploader"),
        "channel_url": info.get("uploader_url"),
        "subscriber_count": info.get("channel_follower_count"),
        "duration_seconds": info.get("duration"),
        "view_count": info.get("view_count"),
        "like_count": info.get("like_count"),
        "comment_count": info.get("comment_count"),
        "upload_date": info.get("upload_date"),
        "tags": info.get("tags", []),
        "categories": info.get("categories", []),
        "thumbnail_url": info.get("thumbnail"),
        "is_live": info.get("is_live", False),
        "language": info.get("language"),
        "description": info.get("description"),
    }

    return expected_file, metadata


def run_cloud_transcription(audio_path: str) -> str:
    """Transcribe audio with Google Gemini."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        raise ValueError("Cloud transcription requires a valid GEMINI_API_KEY environment variable.")

    from google import genai

    print("☁️ Initializing Google Gemini Engine...")
    client = genai.Client(api_key=GEMINI_API_KEY)

    print("📤 Uploading audio payload to Google temporary cloud storage...")
    uploaded_file = client.files.upload(file=audio_path)

    print("🤖 Processing speech-to-text with Gemini Flash...")
    prompt = (
        "Transcribe this audio clip with complete detail. Provide timestamps in format [MM:SS] "
        "and intelligently separate dialogue whenever different speakers talk."
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, uploaded_file],
    )

    print("🧹 Cleaning up remote cloud storage file...")
    client.files.delete(name=uploaded_file.name)
    return response.text


def run_local_transcription(audio_path: str, model_size: str) -> str:
    """Transcribe audio locally with Faster Whisper."""
    from faster_whisper import WhisperModel

    print(f"⚙️ Initializing Local GPU Engine (Model: '{model_size}', Precision: 'int8')...")
    try:
        model = WhisperModel(model_size, device="cuda", compute_type="int8")
    except Exception as e:
        print(f"⚠️ CUDA/GPU warning occurred: {e}")
        print("🔄 Falling back to CPU mode...")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

    print("🎙️ Processing speech local matrices...")
    segments, info = model.transcribe(audio_path, beam_size=5)

    formatted_lines = []
    for segment in segments:
        minutes_start = int(segment.start // 60)
        seconds_start = int(segment.start % 60)
        timestamp = f"[{minutes_start:02d}:{seconds_start:02d}]"
        formatted_lines.append(f"{timestamp} {segment.text.strip()}")

    return "\n".join(formatted_lines)


def append_payload_to_json(output: str, final_payload: dict[str, Any]) -> None:
    """Append a video payload to a JSON file while preserving existing data."""
    if os.path.exists(output):
        with open(output, "r", encoding="utf-8") as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    if not isinstance(existing_data, list):
        existing_data = [existing_data]

    existing_data.append(final_payload)

    with open(output, "w", encoding="utf-8") as file:
        json.dump(existing_data, file, indent=4, ensure_ascii=False)

    print(f"\n💾 Added new video to: {output}")


def process_video(
    url: str,
    engine: str = "local",
    model: str = "small",
    output: str | None = None,
) -> dict[str, Any]:
    """Run the complete YouTube metadata and transcription workflow."""
    local_audio_file = None

    try:
        local_audio_file, video_metadata = extract_audio_and_metadata(url)

        if engine == "cloud":
            transcript_text = run_cloud_transcription(local_audio_file)
        else:
            transcript_text = run_local_transcription(local_audio_file, model)

        final_payload = {
            "metadata": video_metadata,
            "transcript": transcript_text,
        }

        if output:
            append_payload_to_json(output, final_payload)

        return final_payload

    except Exception as e:
        print(f"💥 Pipeline Execution Failed: {e}")
        raise

    finally:
        if local_audio_file and os.path.exists(local_audio_file):
            os.remove(local_audio_file)
            print("✨ System clean.")


# ==========================================
# CLI ORCHESTRATION ROUTINE
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid YouTube Audio Transcript Scraper CLI")
    parser.add_argument("url", help="The full YouTube video URL to process.")
    parser.add_argument("--engine", choices=["cloud", "local"], default="local", help="Processing engine to use.")
    parser.add_argument("--model", choices=["base", "small"], default="small", help="Whisper model size (local engine only).")
    parser.add_argument("--output", help="Optional path to save the structured JSON output (e.g., data.json).")

    args = parser.parse_args()
    result = process_video(
        url=args.url,
        engine=args.engine,
        model=args.model,
        output=args.output,
    )

    if not args.output:
        print(json.dumps(result, indent=4, ensure_ascii=False))
