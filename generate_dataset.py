import asyncio
import edge_tts
import os
import random

async def generate_seeds():
    voices = [v['ShortName'] for v in await edge_tts.list_voices() if v['Locale'].startswith('en-')]
    for folder in ["dataset/raw_hey_home", "dataset/raw_other"]:
        if not os.path.exists(folder): os.makedirs(folder)
    
    print("Generating seed audio...")
    for i in range(100):
        voice = random.choice(voices)
        # Hey Home (Positive)
        await edge_tts.Communicate("Hey Home!", voice).save(f"dataset/raw_hey_home/seed_{i}.mp3")
        # Other (Negative)
        phrase = random.choice(["Hey Phone", "Stay Home", "Way Home", "Hey Bone"])
        await edge_tts.Communicate(phrase, voice).save(f"dataset/raw_other/seed_{i}.mp3")
    print("Seeds generated!")

if __name__ == "__main__":
    asyncio.run(generate_seeds())