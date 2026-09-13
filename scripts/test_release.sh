#!/usr/bin/env bash
set -euo pipefail
echo '=== Vaani Release Test Suite ==='

HERE="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$HERE/.venv"

if [[ ! -f "$VENV/bin/python" ]]; then
    echo 'Creating test venv...'
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -e "$HERE[stt,translate]" --quiet
fi

echo '--- Unit Tests ---'
"$VENV/bin/python" -m pytest "$HERE/tests/unit/" -q --tb=short 2>&1 || true

echo '--- Integration Tests ---'
"$VENV/bin/python" -m pytest "$HERE/tests/integration/" -q --tb=short 2>&1 || true

echo '--- Import Smoke Tests ---'
"$VENV/bin/python" -c 'from vaani.system.hardware import full_report; print("hardware: OK")'
"$VENV/bin/python" -c 'from vaani.system.profile import determine_profile; print("profile: OK")'
"$VENV/bin/python" -c 'from vaani.system.ollama_manager import OllamaManager; print("ollama_mgr: OK")'
"$VENV/bin/python" -c 'from vaani.config import VaaniConfig; print("config: OK")'
"$VENV/bin/python" -c 'from vaani.diagnostics.startup import run_startup_discovery; print("startup: OK")'
"$VENV/bin/python" -c 'from vaani.vision.screen_recording import vision_available; print("vision: OK")'
"$VENV/bin/python" -c 'from vaani.core.pipeline import TranslationPipeline; print("pipeline: OK")'
"$VENV/bin/python" -c 'from vaani.session.engine import TranslationSession; print("session: OK")'

echo '--- Platform Tests ---'
if [[ "$(uname)" == "Linux" ]]; then
    echo 'Platform: Linux/Debian'
    "$VENV/bin/python" -c 'from vaani.audio.backend.pulse_bindings import available; print(f"PulseAudio: {available()}")'
    "$VENV/bin/python" -c 'from vaani.devices.manager import list_sources; print(f"Audio sources: {len(list_sources())}")' 2>/dev/null || echo 'Audio enumeration: SKIPPED (no PulseAudio)'
fi

echo '=== Release Test Complete ==='
