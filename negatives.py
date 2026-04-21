import sounddevice as sd
import soundfile as sf
import os
import time

# Configuration
SR = 16000
DURATION = 2.0  # 2 seconds per clip
SAMPLES = int(SR * DURATION)
OUTPUT_FOLDER = "dataset/raw_other"
DEVICE_ID = 2  # Your Razer Seiren Mini

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print(f"--- STARTING MIC DUMP TO {OUTPUT_FOLDER} ---")
print("Speak naturally, read a book, or play a YouTube video nearby.")
print("Press Ctrl+C to stop recording.")

try:
    file_count = 0
    while True:
        timestamp = int(time.time() * 1000)
        filename = f"{OUTPUT_FOLDER}/mic_dump_{timestamp}.wav"
        
        print(f"Recording clip {file_count}...", end="\r")
        
        # Record 2 seconds of audio
        recording = sd.rec(SAMPLES, samplerate=SR, channels=1, device=DEVICE_ID, dtype='float32')
        sd.wait() # Wait for the 2 seconds to finish
        
        # Save the file
        sf.write(filename, recording, SR)
        file_count += 1

except KeyboardInterrupt:
    print(f"\nStopped. Saved {file_count} clips to {OUTPUT_FOLDER}.")