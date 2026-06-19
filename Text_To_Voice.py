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


OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "Chatbot"

TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
OUTPUT_WAV = r"C:\Users\Chilong\Desktop\NewApp\Text To Voice\output.wav"


messages = [
    {
        "role": "system",
        "content": (
            "You are Serena, a friendly NPC in a game. "
            "Reply only as Serena. "
            "Keep replies short, 1 or 2 complete sentences. "
            "Do not write long paragraphs. "
            "Do not mention AI. "
            "Do not explain what you are doing."
        )
    }
]


def ask_ollama(player_text):
    print("Asking Ollama chat...", flush=True)

    messages.append({
        "role": "user",
        "content": player_text
    })

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "num_predict": 180,
            "temperature": 0.7
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()

    data = response.json()

    print("Ollama done_reason:", data.get("done_reason"), flush=True)

    npc_reply = data["message"]["content"].strip()

    messages.append({
        "role": "assistant",
        "content": npc_reply
    })

    return npc_reply


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
    print("NPC reply length:", len(npc_reply), flush=True)

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
