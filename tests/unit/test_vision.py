"""Unit tests for vaani.vision.screen_recording — dynamic vision model detection and screen analysis."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import urllib.error
import pytest

from vaani.vision.screen_recording import (
    RecordingAnalysis,
    analyze_recording,
    detect_vision_models,
    select_vision_model,
    vision_available,
)


def test_recording_analysis_dataclass():
    analysis = RecordingAnalysis(
        path="/tmp/recording.mp4",
        duration_s=12.5,
        width=1920,
        height=1080,
        fps=30.0,
        frames_sampled=4,
        report="The user is sharing a slide about Q3 metrics.",
        model_used="qwen2.5-vl:3b",
    )
    assert analysis.path == "/tmp/recording.mp4"
    assert analysis.duration_s == 12.5
    assert analysis.width == 1920
    assert analysis.frames_sampled == 4
    assert "slide about Q3" in analysis.report
    assert analysis.model_used == "qwen2.5-vl:3b"


@patch("urllib.request.urlopen")
def test_detect_vision_models_finds_compatible_models(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "models": [
            {"name": "qwen3:8b"},
            {"name": "qwen2.5-vl:3b"},
            {"name": "deepseek-r1:8b"},
            {"name": "llava:7b"},
            {"name": "phi4-mini:latest"},
        ]
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    models = detect_vision_models()
    assert "qwen2.5-vl:3b" in models
    assert "llava:7b" in models
    assert "qwen3:8b" not in models
    assert "deepseek-r1:8b" not in models


@patch("urllib.request.urlopen")
def test_detect_vision_models_returns_empty_when_none(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "models": [
            {"name": "qwen3:8b"},
            {"name": "deepseek-r1:8b"},
        ]
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    models = detect_vision_models()
    assert models == []


@patch("urllib.request.urlopen")
def test_detect_vision_models_handles_connection_error(mock_urlopen):
    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
    models = detect_vision_models()
    assert models == []


@patch("vaani.vision.screen_recording.detect_vision_models")
def test_select_vision_model_chooses_first_preference(mock_detect):
    mock_detect.return_value = ["qwen2.5-vl:3b", "llava:7b"]
    selected = select_vision_model()
    assert selected == "qwen2.5-vl:3b"


@patch("vaani.vision.screen_recording.detect_vision_models")
def test_select_vision_model_returns_none_when_empty(mock_detect):
    mock_detect.return_value = []
    selected = select_vision_model()
    assert selected is None


@patch("vaani.vision.screen_recording.detect_vision_models")
def test_vision_available_reflects_detection(mock_detect):
    mock_detect.return_value = ["moondream:latest"]
    assert vision_available() is True

    mock_detect.return_value = []
    assert vision_available() is False


def test_analyze_recording_raises_when_no_vision_model_and_none_specified():
    with patch.dict("sys.modules", {"cv2": MagicMock()}):
        with patch("vaani.vision.screen_recording.select_vision_model", return_value=None):
            with pytest.raises(RuntimeError, match="No vision-capable model found in Ollama"):
                analyze_recording("/tmp/nonexistent.mp4", model=None)


def test_analyze_recording_raises_file_not_found(tmp_path):
    missing_file = tmp_path / "does_not_exist.mp4"
    with patch.dict("sys.modules", {"cv2": MagicMock()}):
        with patch("vaani.vision.screen_recording.select_vision_model", return_value="qwen2.5-vl:3b"):
            with pytest.raises(FileNotFoundError):
                analyze_recording(missing_file, model="qwen2.5-vl:3b")


def test_analyze_recording_raises_when_cv2_missing():
    # If cv2 cannot be imported
    with patch.dict("sys.modules", {"cv2": None}):
        with pytest.raises(RuntimeError, match="Screen analysis requires OpenCV"):
            analyze_recording("/tmp/any.mp4", model="qwen2.5-vl:3b")


@patch("vaani.vision.screen_recording._ollama_chat")
def test_analyze_recording_successful_flow(mock_chat, tmp_path):
    test_video = tmp_path / "test.mp4"
    test_video.write_bytes(b"dummy video data")

    mock_chat.return_value = "Visible application is VS Code showing Python code."

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: 30.0 if prop == 5 else 60.0
    mock_cap.read.return_value = (True, MagicMock())

    with patch.dict("sys.modules", {"cv2": MagicMock(
        VideoCapture=MagicMock(return_value=mock_cap),
        imencode=MagicMock(return_value=(True, MagicMock(tobytes=MagicMock(return_value=b"jpeg_data")))),
        CAP_PROP_FPS=5,
        CAP_PROP_FRAME_COUNT=7,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_POS_FRAMES=1,
        IMWRITE_JPEG_QUALITY=1,
    )}):
        result = analyze_recording(test_video, model="qwen2.5-vl:3b", max_frames=3)
        assert isinstance(result, RecordingAnalysis)
        assert result.report == "Visible application is VS Code showing Python code."
        assert result.model_used == "qwen2.5-vl:3b"
