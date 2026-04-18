# uvicorn server_whisper_gemma:app --host 0.0.0.0 --port 8000
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import torch
import time
import csv
import tempfile
import shutil
import librosa
import soundfile as sf
import cv2


print("""
██████╗  ███████╗███╗   ███╗███╗   ███╗ █████╗     ██╗  ██╗    ██████╗ ██╗   ██╗███╗   ██╗███╗   ██╗██╗███╗   ██╗ ██████╗ 
██╔════╝ ██╔════╝████╗ ████║████╗ ████║██╔══██╗    ██║  ██║    ██╔══██╗██║   ██║████╗  ██║████╗  ██║██║████╗  ██║██╔════╝ 
██║  ███╗█████╗  ██╔████╔██║██╔████╔██║███████║    ███████║    ██████╔╝██║   ██║██╔██╗ ██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗
██║   ██║██╔══╝  ██║╚██╔╝██║██║╚██╔╝██║██╔══██║    ╚════██║    ██╔══██╗██║   ██║██║╚██╗██║██║╚██╗██║██║██║╚██╗██║██║   ██║
╚██████╔╝███████╗██║ ╚═╝ ██║██║ ╚═╝ ██║██║  ██║         ██║    ██║  ██║╚██████╔╝██║ ╚████║██║ ╚████║██║██║ ╚████║╚██████╔╝
 ╚═════╝ ╚══════╝╚═╝     ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝         ╚═╝    ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝ 
                                                                                                                          """)


from transformers import AutoProcessor, AutoModelForMultimodalLM
# torch.backends.cuda.matmul.allow_tf32 = True

app = FastAPI()

MODEL_ID = "google/gemma-4-E2B-it"

print("🚀 Loading Gemma...")

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="cpu"
)
# Define 4-bit quantization configuration

print("✅ Gemma ready on GPU")  
print("DEVICE:",torch.cuda.get_device_name(0))
print("get_device:",torch.cuda.get_device_capability(0))
print("memory allocated:",torch.cuda.memory_allocated())
print("memory reserved:",torch.cuda.memory_reserved())

# =========================
# CSV LOGGER
# =========================
def log_csv(row):
    with open("gemma_logs.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

# =========================
# COMMON GENERATE FUNCTION
# =========================
def run_model(messages):
    start = time.time()

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=1.0,
            top_p=0.95,
            top_k=64
        )

    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

    end = time.time()

    return response, start, end, (end - start)

# =========================
# TEXT
# =========================
class TextReq(BaseModel):
    text: str

@app.post("/text")
def text_api(req: TextReq):

    messages = [{
        "role": "system",
        "content": [{"type": "text", "text": " You are a smart assistant"}]
    },{
        "role": "user",
        "content": [{"type": "text", "text": req.text}]
    }]

    res, s, e, t = run_model(messages)

    log_csv(["text", s, e, t, res])

    return {"response": res, "start": s, "end": e, "latency": t}

# =========================
# AUDIO
# =========================
@app.post("/audio")
async def audio_api(file: UploadFile = File(...)):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name

    # ✅ load + normalize audio (CRITICAL)
    audio, sr = librosa.load(path, sr=16000)

    # rewrite clean wav
    sf.write(path, audio, 16000)

    messages = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": path},
            {"type": "text", "text": "Listen carefully and reply naturally."}
        ]
    }]

    try:
        res, s, e, t = run_model(messages)
    except Exception as e:
        return {"error": str(e)}

    log_csv(["audio", s, e, t, res])

    return {"response": res, "start": s, "end": e, "latency": t}# =========================
# IMAGE
# =========================
@app.post("/image")
async def image_api(file: UploadFile = File(...), text: str = ""):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": path},
            {"type": "text", "text": text or "Analyze the person"}
        ]
    }]

    res, s, e, t = run_model(messages)
    log_csv(["image", s, e, t, res])

    return {"response": res, "start": s, "end": e, "latency": t}

# =========================
# VIDEO
# =========================
@app.post("/video")
async def video_api(file: UploadFile = File(...), text: str = ""):

    import tempfile, shutil, cv2
    from PIL import Image

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)

    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % 10 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            frames.append(pil_img)

        count += 1

    cap.release()

    if len(frames) == 0:
        return {"error": "No frames extracted"}

    content = []

    # ✅ images FIRST (Gemma best practice)
    for img in frames[:5]:
        content.append({
            "type": "image",
            "image": img
        })

    content.append({
        "type": "text",
        "text": text or "Describe what is happening in this video."
    })

    messages = [{
        "role": "user",
        "content": content
    }]

    try:
        res, s, e, t = run_model(messages)
    except Exception as e:
        return {"error": str(e)}

    return {"response": res, "latency": t}