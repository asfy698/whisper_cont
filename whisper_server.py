from fastapi import FastAPI, UploadFile, File
import whisper
import torch
import numpy as np
import tempfile
import scipy.io.wavfile as wav
import time
import csv

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = whisper.load_model("small", device=device)
print(r"""
 /$$      /$$ /$$   /$$ /$$$$$$  /$$$$$$  /$$$$$$$  /$$$$$$$$ /$$$$$$$        /$$$$$$$                                /$$                    
| $$  /$ | $$| $$  | $$|_  $$_/ /$$__  $$| $$__  $$| $$_____/| $$__  $$      | $$__  $$                              |__/                    
| $$ /$$$| $$| $$  | $$  | $$  | $$  \__/| $$  \ $$| $$      | $$  \ $$      | $$  \ $$ /$$   /$$ /$$$$$$$  /$$$$$$$  /$$ /$$$$$$$   /$$$$$$ 
| $$/$$ $$ $$| $$$$$$$$  | $$  |  $$$$$$ | $$$$$$$/| $$$$$   | $$$$$$$/      | $$$$$$$/| $$  | $$| $$__  $$| $$__  $$| $$| $$__  $$ /$$__  $$
| $$$$_  $$$$| $$__  $$  | $$   \____  $$| $$____/ | $$__/   | $$__  $$      | $$__  $$| $$  | $$| $$  \ $$| $$  \ $$| $$| $$  \ $$| $$  \ $$
| $$$/ \  $$$| $$  | $$  | $$   /$$  \ $$| $$      | $$      | $$  \ $$      | $$  \ $$| $$  | $$| $$  | $$| $$  | $$| $$| $$  | $$| $$  | $$
| $$/   \  $$| $$  | $$ /$$$$$$|  $$$$$$/| $$      | $$$$$$$$| $$  | $$      | $$  | $$|  $$$$$$/| $$  | $$| $$  | $$| $$| $$  | $$|  $$$$$$$
|__/     \__/|__/  |__/|______/ \______/ |__/      |________/|__/  |__/      |__/  |__/ \______/ |__/  |__/|__/  |__/|__/|__/  |__/ \____  $$
                                                                                                                                    /$$  \ $$
                                                                                                                                   |  $$$$$$/
                                                                                                                                    \______/ """)
print("DEVICE:",torch.cuda.get_device_name(0))
print("get_device:",torch.cuda.get_device_capability(0))
print("memory allocated:",torch.cuda.memory_allocated())
print("memory reserved:",torch.cuda.memory_reserved())
def log_csv(row):
    with open("whisper_logs.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):

    start = time.time()

    raw = await file.read()
    audio = np.frombuffer(raw, dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        wav.write(tmp.name, 16000, audio)
        result = model.transcribe(tmp.name)

    end = time.time()

    text = result["text"].strip()

    log_csv(["whisper", start, end, end-start, text])

    return {
        "text": text,
        "start": start,
        "end": end,
        "latency": end - start
    }