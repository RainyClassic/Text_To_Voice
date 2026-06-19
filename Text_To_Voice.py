
import sys
import os

os.environ["NUMBA_CACHE_DIR"] = os.path.join(os.path.expanduser("~"), ".numba_cache")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

print("Script started", flush=True)

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

print("Script started", flush=True)

if len(sys.argv) < 2:
    print("Missing text. Use: python Text_To_Voice.py \"hello\"", flush=True)
    sys.exit(1)

text = sys.argv[1]
print("Speaking:", text, flush=True)
print("Current folder:", os.getcwd(), flush=True)
print("Will save to:", os.path.abspath("output.wav"), flush=True)

print("Importing torch...", flush=True)
import torch

print("Importing soundfile...", flush=True)
import soundfile as sf

print("Importing Qwen TTS...", flush=True)
from qwen_tts import Qwen3TTSModel

print("Loading model...", flush=True)
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.float16
)

print("Model loaded.", flush=True)

print("Generating speech...", flush=True)
wavs, sr = model.generate_custom_voice(
    text=text,
    language="English",
    speaker="serena"
)

print("Saving WAV...", flush=True)
sf.write("output.wav", wavs[0], sr)

print("Done!", flush=True)
print("Saved:", os.path.abspath("output.wav"), flush=True)