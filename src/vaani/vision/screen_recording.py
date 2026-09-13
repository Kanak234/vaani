"""Local screen-recording analysis using OpenCV + a local Ollama vision model."""
from __future__ import annotations

import base64
import json
import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_log = logging.getLogger(__name__)

# Known vision-capable model families (ordered by preference)
_VISION_MODEL_PATTERNS = (
    "qwen3-vl", "qwen2.5-vl", "qwen2-vl", "llava", "moondream",
    "bakllava", "cogvlm", "minicpm-v", "internvl",
)


@dataclass(slots=True)
class RecordingAnalysis:
    path: str
    duration_s: float
    width: int
    height: int
    fps: float
    frames_sampled: int
    report: str
    model_used: str = ""


def _ollama_host() -> str:
    return os.environ.get("VAANI_OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def detect_vision_models(host: str | None = None) -> list[str]:
    """Query Ollama for installed models that are likely vision-capable."""
    h = host or _ollama_host()
    try:
        req = urllib.request.Request(f"{h}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        names = [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        return []

    vision_models: list[str] = []
    for pattern in _VISION_MODEL_PATTERNS:
        for name in names:
            if pattern in name.lower().split(":")[0]:
                vision_models.append(name)
    return vision_models


def select_vision_model(host: str | None = None) -> str | None:
    """Select the best available vision model, or None if unavailable."""
    models = detect_vision_models(host)
    if models:
        _log.info("Vision models available: %s; selected: %s", models, models[0])
        return models[0]
    _log.warning("No vision-capable model found in Ollama")
    return None


def _ollama_chat(model: str, prompt: str, images: list[str]) -> str:
    host = _ollama_host()
    payload = json.dumps({
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt, "images": images}],
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data.get("message", {}).get("content", "")).strip()


def vision_available(host: str | None = None) -> bool:
    """Check whether screen analysis is available (vision model installed)."""
    return bool(detect_vision_models(host))


def analyze_recording(path: str | Path, *, model: str | None = None,
                      max_frames: int = 6) -> RecordingAnalysis:
    """Analyze a screen recording using a local vision model.

    Parameters
    ----------
    path : path to video file
    model : Ollama vision model name, or None for auto-detection
    max_frames : maximum frames to sample from the recording

    Raises
    ------
    RuntimeError
        If OpenCV is missing, no vision model is available, or analysis fails.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Screen analysis requires OpenCV. Install Vaani with the [screen] extra."
        ) from exc

    # Auto-detect vision model if not specified
    if model is None:
        model = select_vision_model()
        if model is None:
            raise RuntimeError(
                "No vision-capable model found in Ollama. "
                "Install a vision model (e.g. 'ollama pull qwen2.5-vl:3b') "
                "to enable screen analysis."
            )

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
    return RecordingAnalysis(str(source), duration, width, height, fps,
                            len(images), report, model_used=model)
