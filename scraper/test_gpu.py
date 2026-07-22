from faster_whisper import WhisperModel
import numpy as np

print("1. Loading AI model into GTX 1650 VRAM...")
model = WhisperModel("small", device="cuda", compute_type="int8")

print("2. Generating a dummy audio matrix (1 second of silence)...")
# We feed it raw numpy data so we don't even need an audio file yet
dummy_audio = np.zeros(16000, dtype=np.float32)

print("3. Executing cuBLAS matrix math...")
try:
    segments, info = model.transcribe(dummy_audio)
    # We must iterate through the generator to force the math to process
    for s in segments:
        pass
    print("\n✅ SUCCESS! Your GPU and cuBLAS libraries are perfectly linked.")
except Exception as e:
    print(f"\n❌ FAILED during math execution: {e}")