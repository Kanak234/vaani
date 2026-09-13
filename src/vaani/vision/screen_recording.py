"""Local screen-recording analysis using OpenCV + a local Ollama vision model.

The analyzer samples frames from a recording, sends only those local images to
Ollama, and returns a compact visual/context report. No recording is uploaded.
"""
from __future__ import annotations

import base64
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RecordingAnalysis:
    path: str
    duration_s: float
    width: int
    height: int
    fps: float
    frames_sampled: int
    report: str


def _ollama_chat(model: str, prompt: str, images: list[str]) -> str:
    payload = json.dumps({
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt, "images": images}],
    }).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data.get("message", {}).get("content", "")).strip()


def analyze_recording(path: str | Path, *, model: str = "qwen3-vl:4b",
                      max_frames: int = 6) -> RecordingAnalysis:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Screen analysis requires OpenCV. Install Vaani with the [screen] extra."
        ) from exc

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open recording: {source}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 else 0.0
        count = max(1, min(max_frames, frame_count or max_frames))
        positions = [int(i * max(frame_count - 1, 0) / max(count - 1, 1)) for i in range(count)]
        images: list[str] = []
        for position in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = capture.read()
            if not ok:
                continue
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                images.append(base64.b64encode(encoded.tobytes()).decode("ascii"))
    finally:
        capture.release()

    if not images:
        raise RuntimeError("No readable video frames were found in the recording.")

    prompt = (
        "Analyze this screen recording using the sampled frames. Identify the visible "
        "application/context, important on-screen text, user intent or task, questions "
        "being discussed if inferable, and concrete facts that should influence a useful "
        "meeting response. Do not invent details that are not visible. Return a concise "
        "context report suitable for another local assistant to use when drafting a spoken "
        "response. Mention uncertainty when the frames do not support a conclusion."
    )
    report = _ollama_chat(model, prompt, images)
    if not report:
        raise RuntimeError("The local vision model returned an empty analysis.")
    return RecordingAnalysis(str(source), duration, width, height, fps, len(images), report)
