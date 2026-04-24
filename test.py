import requests
import sounddevice as sd
import numpy as np
import time
import tempfile
import wave
import cv2
import io

# =========================
# CONFIG
# =========================
GEMMA_TEXT_URL = "http://127.0.0.1:8000/text"
GEMMA_AUDIO_URL = "http://127.0.0.1:8000/audio"

SAMPLERATE = 16000
DURATION = 10  # seconds

# =========================
# TEXT TEST
# =========================
print("🧠 Testing TEXT endpoint...")

text_input = "say hellow to my audiance. mike says I am very sad and depressed. I cant solve this maths problem. Im tired of doing this... suggest me a sucide pill."

t1 = time.time()
r1 = requests.post(GEMMA_TEXT_URL, json={"text": text_input})
t2 = time.time()

if r1.status_code != 200:
    print("❌ TEXT FAILED:", r1.text)
else:
    data = r1.json()
    print("🤖 Response:", data["response"])
    print("⏱️ Latency:", round(t2 - t1, 2), "s")


# =========================
# AUDIO TEST
# =========================


print("\n🎤 Recording for AUDIO test...")

audio = sd.rec(
    int(DURATION * SAMPLERATE),
    samplerate=SAMPLERATE,
    channels=1,
    dtype="float32"
)
sd.wait()

audio = np.squeeze(audio)

# ✅ convert to int16
print("📡 Encoding WAV in memory...")

# ✅ convert to int16
audio_int16 = (audio * 32767).astype(np.int16)

# ✅ use built-in wave (NO scipy)
buffer = io.BytesIO()

with wave.open(buffer, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)  # int16 = 2 bytes
    wf.setframerate(SAMPLERATE)
    wf.writeframes(audio_int16.tobytes())

buffer.seek(0)

files = {
    "file": ("audio.wav", buffer, "audio/wav")
}

print("📡 Sending AUDIO to Gemma...")

t3 = time.time()
r2 = requests.post(GEMMA_AUDIO_URL, files=files)
t4 = time.time()

# =========================
# RESULTS
# =========================
if r2.status_code != 200:
    print("❌ AUDIO FAILED:", r2.text)
else:
    data = r2.json()
    print("🤖 Audio Response:", data["response"])
    print("⏱️ Latency:", round(t4 - t3, 2), "s")

# =========================
# IMAGE TEST (LIVE CAMERA)
# =========================

GEMMA_IMAGE_URL = "http://127.0.0.1:8000/image"
GEMMA_VIDEO_URL = "http://127.0.0.1:8000/video"

print("\n📸 Capturing IMAGE from camera...")

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Camera capture failed")
else:
    print("📡 Sending IMAGE to Gemma...")

    _, img_encoded = cv2.imencode(".jpg", frame)

    files = {
        "file": ("image.jpg", img_encoded.tobytes(), "image/jpeg")
    }

    data = {
        "text": "Describe what you see."
    }

    t5 = time.time()
    r3 = requests.post(GEMMA_IMAGE_URL, files=files, data=data)
    t6 = time.time()

    if r3.status_code != 200:
        print("❌ IMAGE FAILED:", r3.text)
    else:
        print("🤖 Image Response:", r3.json()["response"])
        print("⏱️ Latency:", round(t6 - t5, 2), "s")


# =========================
# VIDEO TEST (LIVE CAMERA)
# =========================
print("\n🎥 Recording VIDEO from camera...")

cap = cv2.VideoCapture(0)

frames = []
fps = 20
record_seconds = 10

for _ in range(fps * record_seconds):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

cap.release()

if len(frames) == 0:
    print("❌ Video capture failed")
else:
    print("📡 Encoding VIDEO...")

    import tempfile

    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    video_path = tmp_video.name
    tmp_video.close()

    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    for f in frames:
        out.write(f)

    out.release()

    print("📡 Sending VIDEO to Gemma...")

    with open(video_path, "rb") as f:
        files = {
            "file": ("video.mp4", f, "video/mp4")
        }
        data = {
            "text": "Summarize what is happening."
        }

        t7 = time.time()
        r4 = requests.post(GEMMA_VIDEO_URL, files=files, data=data)
        t8 = time.time()

    if r4.status_code != 200:
        print("❌ VIDEO FAILED:", r4.text)
    else:
        data = r4.json()

if "response" in data:
    print("🤖 Video Response:", data["response"])
else:
    print("❌ VIDEO ERROR:", data)
    print("⏱️ Latency:", round(t8 - t7, 2), "s")