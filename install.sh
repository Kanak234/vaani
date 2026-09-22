#!/usr/bin/env bash
#
# Vaani one-step installer.
#
# The acceptance criterion this exists for: a fresh clone, then ./install.sh,
# then `vaani` works with no manual steps.
#
# WHY THERE ARE TWO INSTALL SCRIPTS. packaging/install.sh predates this one and
# does a different, narrower job: desktop integration (a launcher symlink, an
# icon, .desktop entries). It explicitly REFUSES to run without an existing
# virtualenv. That is the right behaviour for what it is, so it is called from
# here rather than duplicated or replaced -- this script builds everything it
# needs and then hands over.
#
# WHY MODELS ARE FETCHED DURING INSTALL. Without it the first `vaani meeting`
# prints one line and then sits silent for minutes while huggingface_hub pulls
# ~1.5 GB, with no progress reaching the terminal. During Phase 2 testing that
# was mistaken for a hang three times before a stack dump showed it was a
# download. A first-time user would conclude the same and kill it. A long wait
# during an install is expected; a long silence on first launch is a bug.
#
# EVERYTHING IS IDEMPOTENT. Re-running is safe and is the normal way to repair a
# half-finished install: an existing venv is reused, already-downloaded models
# are skipped, and symlinks are replaced rather than duplicated.
#
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"

# Defaults chosen for the machines this targets. --cpu matters because the
# laptop may be running Nouveau/NVK rather than the proprietary driver, where
# the CUDA extra installs several hundred MB that can never be used.
WITH_CUDA=1
WITH_VOICE=1
FETCH_MODELS=1
PYTHON_BIN=""

usage() {
  cat <<'USAGE'
Usage: ./install.sh [options]

  --cpu           skip the CUDA libraries (no NVIDIA GPU, or Nouveau/NVK)
  --no-voice      skip XTTS (~1.8 GB). The fallback voice still works,
                  but it will not be YOUR voice.
  --no-models     install code only; models download on first run instead
  --python PATH   use a specific interpreter (default: newest >= 3.12 found)
  -h, --help      show this

Re-running is safe: it reuses the venv and skips models already downloaded.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu)        WITH_CUDA=0; shift ;;
    --no-voice)   WITH_VOICE=0; shift ;;
    --no-models)  FETCH_MODELS=0; shift ;;
    --python)     PYTHON_BIN="${2:-}"; shift 2 ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '\033[33m    WARNING: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31m    ERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. preflight
say "Checking this machine"

if [[ "$(uname -s)" != "Linux" ]]; then
  # Windows has its own path (packaging/windows, scripts/build_windows.ps1).
  die "this installer is for Linux. See packaging/windows/ for Windows."
fi

# Find an interpreter. pyproject requires >= 3.12; newest first, because the
# repo is developed on 3.14 and older versions see less testing.
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.14 python3.13 python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      version="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
      major="${version%%.*}"; minor="${version##*.}"
      if [[ "$major" -eq 3 && "$minor" -ge 12 ]]; then
        PYTHON_BIN="$(command -v "$candidate")"
        break
      fi
    fi
  done
fi
[[ -n "$PYTHON_BIN" ]] || die "no Python >= 3.12 found. Install one, or pass --python PATH"
info "python    $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"

# PipeWire/PulseAudio is not optional: the virtual microphone IS the product.
# Checked here rather than at first run so the user learns now, not mid-meeting.
if command -v pactl >/dev/null 2>&1; then
  if pactl info >/dev/null 2>&1; then
    info "audio     pactl OK ($(pactl info 2>/dev/null | sed -n 's/^Server Name: //p'))"
  else
    warn "pactl is installed but cannot reach a sound server."
    warn "The virtual microphone will not work until it can."
  fi
else
  warn "pactl not found. Vaani needs PipeWire (or PulseAudio) for the virtual mic."
  warn "On Debian/Ubuntu:  sudo apt install pipewire-pulse pulseaudio-utils"
fi

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  info "gpu       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
else
  if [[ "$WITH_CUDA" -eq 1 ]]; then
    warn "no working nvidia-smi. Installing CUDA libraries anyway (--cpu skips them)."
    warn "Vaani runs on CPU regardless; it is slower, not broken."
  fi
fi

# ------------------------------------------------------------------ 2. venv
say "Creating the virtual environment"
if [[ -x "$VENV/bin/python" ]]; then
  info "reusing existing venv at $VENV"
else
  # uv when present: an order of magnitude faster, and it is already used by
  # this project's own docs. Falling back to venv keeps a fresh clone working
  # on a machine that has never heard of uv.
  if command -v uv >/dev/null 2>&1; then
    info "creating with uv"
    uv venv --python "$PYTHON_BIN" "$VENV"
  else
    info "creating with python -m venv"
    "$PYTHON_BIN" -m venv "$VENV"
  fi
fi
VPY="$VENV/bin/python"
[[ -x "$VPY" ]] || die "venv creation failed: no interpreter at $VPY"

# ------------------------------------------------------------------ 3. deps
say "Installing Vaani and its dependencies"
EXTRAS="stt,translate"
[[ "$WITH_VOICE" -eq 1 ]] && EXTRAS="$EXTRAS,voice"
[[ "$WITH_CUDA"  -eq 1 ]] && EXTRAS="$EXTRAS,cuda"
info "extras: [$EXTRAS]"

if command -v uv >/dev/null 2>&1; then
  VIRTUAL_ENV="$VENV" uv pip install -e "$HERE[$EXTRAS]"
else
  "$VPY" -m pip install --upgrade pip >/dev/null
  "$VPY" -m pip install -e "$HERE[$EXTRAS]"
fi

# Import check before going any further: a package that installed but cannot
# import is worth catching here rather than after a multi-GB model download.
"$VPY" -c "import vaani" || die "vaani installed but will not import"
info "import OK"

# ---------------------------------------------------------------- 4. models
if [[ "$FETCH_MODELS" -eq 1 ]]; then
  say "Downloading models (slow once, instant afterwards)"
  FETCH_ARGS=()
  [[ "$WITH_VOICE" -eq 1 ]] && FETCH_ARGS+=(--voice)
  # Not fatal. Code that installed correctly is still useful, and the models
  # will download on first run - which is merely slow, not broken.
  if ! "$VPY" -m vaani fetch-models "${FETCH_ARGS[@]}"; then
    warn "some models did not download. They will be fetched on first run."
  fi
else
  info "skipping model download (--no-models)"
fi

# --------------------------------------------------------- 5. virtual mic
say "Checking the virtual microphone"
# Create and immediately destroy it. This is the one part of the stack that
# touches the system's audio graph, and a failure here is the difference
# between "Vaani works" and "my meeting app cannot see Vaani".
if "$VPY" -m vaani virtualmic create >/dev/null 2>&1; then
  info "created OK"
  "$VPY" -m vaani virtualmic destroy >/dev/null 2>&1 || true
  info "removed again (it is created automatically when you start a meeting)"
else
  warn "could not create the virtual microphone."
  warn "Run './.venv/bin/python -m vaani doctor' to see why."
fi

# ------------------------------------------------- 6. desktop integration
say "Installing the launcher and desktop entry"
# packaging/install.sh is the narrower, older script: launcher symlink, icon,
# .desktop files. It needs the venv that now exists.
if [[ -x "$HERE/packaging/install.sh" ]]; then
  "$HERE/packaging/install.sh"
else
  warn "packaging/install.sh missing or not executable; skipping desktop entry"
fi

# ---------------------------------------------------------------- 7. verify
say "Verifying"
if "$VPY" -m vaani models >/dev/null 2>&1; then
  info "'vaani models' runs"
else
  warn "'vaani models' failed. Try: ./.venv/bin/python -m vaani doctor"
fi

# ------------------------------------------------------------------- done
say "Installed"
cat <<NEXT
    Try it:
      vaani models          what Vaani will use, and why
      vaani doctor          diagnose audio and models
      vaani enroll          record your voice (needed for YOUR voice)
      vaani meeting         start meeting mode

    In your meeting app, choose the microphone:
      Vaani Virtual Microphone

    If 'vaani' is not found, add ~/.local/bin to your PATH:
      echo 'export PATH="\$HOME/.local/bin:\$PATH"' >> ~/.bashrc && exec bash

    Optional - physical hold-to-talk (vaani meeting --input-mode ptt_hold)
    needs raw keyboard access, which a fresh install does not have:
      ./.venv/bin/python -m pip install evdev
      sudo usermod -aG input \$USER     # then log out and back in
    Without it, 'vaani talk' (toggle) works with no special permissions.
NEXT
