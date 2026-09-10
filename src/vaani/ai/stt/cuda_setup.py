"""Make the pip-installed NVIDIA libraries loadable.

`nvidia-cublas-cu12` and `nvidia-cudnn-cu12` install their shared objects under
`site-packages/nvidia/*/lib`, which is not on the dynamic loader's search path.
CTranslate2 dlopen()s them by bare soname, so without help it fails with:

    RuntimeError: Library libcublas.so.12 is not found or cannot be loaded

The failure is nastier than it looks, and this is why it gets its own module:

  * `ctranslate2.get_cuda_device_count()` still returns 1 -- it only asks the
    driver whether a GPU exists.
  * `WhisperModel(..., device="cuda")` still SUCCEEDS. Constructing the model does
    not touch cuBLAS.
  * The error only appears on the first encode, i.e. on the user's first utterance,
    mid-meeting.

All three were observed on the target machine. So we preload the libraries
explicitly, before CTranslate2 is imported, and report honestly when we cannot.

Measured effect on this machine (RTX 3050 Mobile, real 11 s speech):

    small  CPU int8          RTF 0.262
    small  CUDA int8_float16 RTF 0.025   (10.5x faster)
    medium CUDA int8_float16 RTF 0.061   (still 4x faster than small on CPU)
"""
from __future__ import annotations

import ctypes
import os
import site
import sys
from pathlib import Path

#: Load order matters: cuDNN links against cuBLAS.
_LIB_DIRS = ("nvidia/cublas/lib", "nvidia/cuda_nvrtc/lib", "nvidia/cudnn/lib")
_PRELOAD = ("libcublas.so.12", "libcublasLt.so.12", "libcudnn.so.9")

_done = False
_result: tuple[bool, str] = (False, "not attempted")


def _site_dirs() -> list[Path]:
    dirs: list[Path] = []
    try:
        dirs += [Path(p) for p in site.getsitepackages()]
    except Exception:
        pass
    user = site.getusersitepackages() if hasattr(site, "getusersitepackages") else None
    if isinstance(user, str):
        dirs.append(Path(user))
    # Covers venvs where getsitepackages() is not populated as expected.
    for p in sys.path:
        if p.endswith("site-packages"):
            dirs.append(Path(p))
    seen, out = set(), []
    for d in dirs:
        if d not in seen and d.is_dir():
            seen.add(d)
            out.append(d)
    return out


def ensure_cuda_libraries() -> tuple[bool, str]:
    """Best-effort: make CUDA libraries loadable. Idempotent.

    Returns (ok, detail). Never raises -- a CPU fallback is always available, and
    a hard failure here would take down a session that could have run.
    """
    global _done, _result
    if _done:
        return _result

    found: list[Path] = []
    for base in _site_dirs():
        for rel in _LIB_DIRS:
            d = base / rel
            if d.is_dir():
                found.append(d)

    if not found:
        _done, _result = True, ("no pip-installed CUDA libraries found; "
                                "install nvidia-cublas-cu12 and nvidia-cudnn-cu12 "
                                "for GPU acceleration")
        return _result

    # Help anything that re-execs or dlopens later in the process.
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [str(d) for d in found] + ([existing] if existing else [])
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(parts)

    # The part that actually works in THIS process: once a soname is loaded into
    # the global namespace, later dlopen() calls for it resolve immediately.
    loaded, failed = [], []
    for soname in _PRELOAD:
        for d in found:
            candidate = d / soname
            if candidate.exists():
                try:
                    ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                    loaded.append(soname)
                except OSError as exc:
                    failed.append(f"{soname}: {exc}")
                break

    if not loaded:
        _done, _result = True, f"CUDA libraries present but none loaded: {failed}"
        return _result

    _done, _result = True, f"preloaded {', '.join(loaded)}"
    return _result


def cuda_available() -> tuple[bool, str]:
    """Whether CUDA can actually run inference -- not merely whether a GPU exists.

    Model construction is deliberately NOT used as the test: it succeeds even when
    cuBLAS is missing. Only a real encode proves the path works.
    """
    ensure_cuda_libraries()
    try:
        import ctranslate2
    except ImportError:
        return False, "ctranslate2 not installed"
    if ctranslate2.get_cuda_device_count() < 1:
        return False, "no CUDA device"
    try:
        import numpy as np
        from faster_whisper import WhisperModel
        model = WhisperModel("tiny", device="cuda", compute_type="int8_float16")
        list(model.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1)[0])
        del model
        return True, "verified by inference"
    except Exception as exc:
        msg = str(exc)
        lib = next((t for t in msg.split() if t.startswith("lib") and ".so" in t), None)
        return False, (f"GPU present but unusable (missing {lib})" if lib
                       else f"GPU present but unusable: {msg[:80]}")
