from faster_whisper import WhisperModel
import os

# Change this if your file has a different extension (e.g., 'test.m4a')
AUDIO_FILE = "test.mp3" 

if not os.path.exists(AUDIO_FILE):
    print(f"❌ Error: Please place a file named '{AUDIO_FILE}' in this folder first!")
    exit()

print("1. Loading AI model into GTX 1650 VRAM...")
model = WhisperModel("small", device="cuda", compute_type="int8")

print(f"2. Attempting to decode and transcribe '{AUDIO_FILE}'...")
try:
    segments, info = model.transcribe(AUDIO_FILE, beam_size=5)
    
    print("3. Processing text output...")
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        
    print("\n✅ SUCCESS! Local audio file decoding works perfectly on your GPU.")
except Exception as e:
    print(f"\n❌ FAILED during audio decoding: {e}")