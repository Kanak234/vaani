import os
import platform as pyplatform
from pathlib import Path

def is_windows() -> bool:
    """Return True if the current operating system is Windows."""
    return os.name == 'nt'

def is_linux() -> bool:
    """Return True if the current operating system is Linux."""
    return os.name == 'posix' and pyplatform.system() == 'Linux'

def _ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist and return it."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return path

def data_dir() -> Path:
    """Get the base data directory for the application."""
    if is_windows():
        base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
        return _ensure_dir(base / 'Vaani')
    else:
        base = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share'))
        return _ensure_dir(base / 'vaani')

def config_dir() -> Path:
    """Get the configuration directory for the application."""
    if is_windows():
        base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
        return _ensure_dir(base / 'Vaani' / 'config')
    else:
        base = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config'))
        return _ensure_dir(base / 'vaani')

def runtime_dir() -> Path:
    """Get the runtime directory for transient files."""
    if is_windows():
        username = os.environ.get('USERNAME', 'default')
        base = Path(os.environ.get('TEMP', Path.home() / 'AppData' / 'Local' / 'Temp'))
        return _ensure_dir(base / f'vaani-{username}')
    else:
        uid = os.getuid() if hasattr(os, 'getuid') else 1000
        base = Path(os.environ.get('XDG_RUNTIME_DIR', f'/tmp'))
        return _ensure_dir(base / f'vaani-{uid}')

def log_dir() -> Path:
    """Get the logging directory."""
    return _ensure_dir(data_dir() / 'logs')

def models_dir() -> Path:
    """Get the models directory, respecting environment variables."""
    if 'VAANI_MODELS_DIR' in os.environ:
        return _ensure_dir(Path(os.environ['VAANI_MODELS_DIR']))
    if 'OLLAMA_MODELS' in os.environ:
        return _ensure_dir(Path(os.environ['OLLAMA_MODELS']))
    
    return _ensure_dir(data_dir() / 'models')

def cache_dir() -> Path:
    """Get the cache directory."""
    if is_windows():
        base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
        return _ensure_dir(base / 'Vaani' / 'cache')
    else:
        base = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
        return _ensure_dir(base / 'vaani')
