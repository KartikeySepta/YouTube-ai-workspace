import os
from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel

# A short 30-second test video to verify the pipeline quickly
VIDEO_URL = "https://youtu.be/4NVWPOAMfN0?si=oeQB6JcdHjuD6XfS" 
AUDIO_OUTPUT = "pipeline_test_audio"

print("1. Downloading audio stream via yt-dlp...")
ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '128',
    }],
    'outtmpl': AUDIO_OUTPUT,
    'quiet': True,
    'no_warnings': True,
}

with YoutubeDL(ydl_opts) as ydl:
    ydl.download([VIDEO_URL])

downloaded_file = f"{AUDIO_OUTPUT}.mp3"

if not os.path.exists(downloaded_file):
    print("❌ Download Phase Failed: Audio file was not created by FFmpeg.")
    exit()

print("2. Loading AI model into VRAM...")
model = WhisperModel("small", device="cuda", compute_type="int8")

print("3. Transcribing live stream data...")
try:
    segments, info = model.transcribe(downloaded_file, beam_size=5)
    
    print("\n--- PIPELINE TRANSCRIPT OUTPUT ---")
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    print("----------------------------------")
    
    print("\n✅ SUCCESS! Your full local end-to-end scraper pipeline is working.")

except Exception as e:
    print(f"\n❌ FAILED during pipeline execution: {e}")

finally:
    # Cleanup downloaded file
    if os.path.exists(downloaded_file):
        os.remove(downloaded_file)