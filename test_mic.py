import sounddevice as sd
import numpy as np
import sys

# Based on your previous scan: 2 is the Razer Seiren Mini
DEVICE_ID = 2
SAMPLE_RATE = 16000  # Standard for wake-word models
CHANNELS = 1

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    
    # Calculate volume level
    volume_norm = np.linalg.norm(indata) * 20
    
    # Create a bar (max 50 chars)
    bar_length = int(min(volume_norm, 50))
    bar = "█" * bar_length + "-" * (50 - bar_length)
    
    # \r returns the cursor to the start of the line for a clean animation
    sys.stdout.write(f"\rMic Level: [{bar}] {volume_norm:.2f} ")
    sys.stdout.flush()

print(f"--- Testing Razer Seiren Mini (Device {DEVICE_ID}) ---")
print("Press Ctrl+C to exit\n")

try:
    with sd.InputStream(device=DEVICE_ID, 
                        channels=CHANNELS, 
                        samplerate=SAMPLE_RATE, 
                        callback=audio_callback):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\n\nTest stopped by user.")
except Exception as e:
    print(f"\n\nAn error occurred: {e}")
    print("\nTip: If you get a 'Device not found' error, run 'python3 -m sounddevice' to check the ID again.")