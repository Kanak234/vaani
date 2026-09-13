from unittest.mock import patch
import pytest

from vaani.vision.screen_recording import detect_vision_models, vision_available, analyze_recording

@patch('vaani.vision.screen_recording.detect_vision_models', return_value=[])
def test_vision_available_returns_false_when_no_models(mock_detect):
    assert vision_available() is False

def test_detect_vision_models_handles_offline():
    # If no connection to huggingface or ollama, should return empty list or fallback
    models = detect_vision_models()
    assert isinstance(models, list)

def test_analyze_recording_requires_model():
    with pytest.raises(Exception):
        # Without specifying a valid model or with mock, it should raise or return error
        analyze_recording(b"fake_data", model=None)
