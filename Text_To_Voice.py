
import os
import requests
import winsound

# Fix numba/librosa permission issue
os.environ["NUMBA_CACHE_DIR"] = os.path.join(os.path.expanduser("~"), ".numba_cache")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

print("Starting Python NPC voice test...", flush=True)

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "Chatbot"   

TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
OUTPUT_WAV = r"C:\Users\Chilong\Desktop\NewApp\Text To Voice\output.wav"


def ask_ollama(prompt):
    print("Asking Ollama...", flush=True)

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "num_predict": 120
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()

    data = response.json()
    reply = data.get("response", "").strip()

    return reply


print("Loading Qwen TTS model...", flush=True)

model = Qwen3TTSModel.from_pretrained(
    TTS_MODEL,
    device_map="cuda:0",
    dtype=torch.float16
)

print("TTS model loaded.", flush=True)

while True:
    user_text = input("\nYou: ")

    if user_text.lower() in ["quit", "exit", "q"]:
        print("Closing.")
        break

    npc_reply = ask_ollama(user_text)

    print("\nNPC:", npc_reply, flush=True)

    print("Generating voice...", flush=True)

    wavs, sr = model.generate_custom_voice(
        text=npc_reply,
        language="English",
        speaker="serena"
    )

    sf.write(OUTPUT_WAV, wavs[0], sr, subtype="PCM_16")

    print("Done! Saved:", OUTPUT_WAV, flush=True)
    print("Playing voice...", flush=True)

    winsound.PlaySound(OUTPUT_WAV, winsound.SND_FILENAME)

    print("Voice finished.", flush=True)
