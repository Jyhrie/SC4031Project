import sounddevice as sd
import soundfile as sf
import os
import time

SR = 16000
DURATION = 2.0
SAMPLES = int(SR * DURATION)
OUTPUT_FOLDER = "dataset/raw_hey_home"
DEVICE_ID = 1 

if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

print("--- POSITIVE DATA COLLECTION ---")
print("Every time you see 'SPEAK', say 'Hey Home!' clearly.")

for i in range(50):  # Let's start with 50 real samples
    print(f"\n[{i+1}/50] Get ready...")
    time.sleep(1)
    print(">>> SPEAK NOW! <<<")
    
    recording = sd.rec(SAMPLES, samplerate=SR, channels=1, device=DEVICE_ID, dtype='float32')
    sd.wait()
    
    filename = f"{OUTPUT_FOLDER}/real_voice_{i}.wav"
    sf.write(filename, recording, SR)
    print("Done.")

print("\nAwesome. Now run your Librosa Augmentation script to turn these 50 into 500!")