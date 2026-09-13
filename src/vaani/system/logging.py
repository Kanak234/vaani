import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional
from vaani.system.platform import log_dir as default_log_dir

class StructuredFormatter(logging.Formatter):
    def _sanitize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = {}
        sensitive_keys = {'audio', 'samples', 'password', 'secret', 'token', 'key'}
        for k, v in data.items():
            if k.lower() in sensitive_keys:
                continue
            sanitized[k] = v
        return sanitized

    def format(self, record: logging.LogRecord) -> str:
        output = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "component": getattr(record, 'component', record.name),
            "level": record.levelname,
            "message": record.getMessage()
        }
        
        extra = {}
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord(None, None, "", 0, "", (), None).__dict__ and key != 'component':
                extra[key] = value
                
        if extra:
            output["extra"] = self._sanitize(extra)
            
        if record.exc_info:
            output["exc_info"] = self.formatException(record.exc_info)
            
        return json.dumps(output)

class ConsoleFormatter(logging.Formatter):
    COLORS = {
        'ERROR': '\033[91m',   # Red
        'WARNING': '\033[93m', # Yellow
        'INFO': '\033[0m',     # Default
        'DEBUG': '\033[90m'    # Grey
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        time_str = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        level_name = record.levelname
        component = getattr(record, 'component', record.name)
        msg = record.getMessage()
        
        # Simple heuristic to check if terminal supports color
        supports_color = True # Defaulting to True for modern terminals
        
        if supports_color and level_name in self.COLORS:
            color = self.COLORS[level_name]
            return f"[{time_str}] {color}{level_name}{self.RESET} {component}: {msg}"
        return f"[{time_str}] {level_name} {component}: {msg}"

def setup_logging(*, level: str = 'INFO', log_dir: Optional[Path] = None, component: str = 'vaani') -> logging.Logger:
    logger = logging.getLogger('vaani')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid duplicate handlers if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    ldir = log_dir or default_log_dir()
    log_file = ldir / 'vaani.log'

    # File Handler
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    file_handler.setFormatter(StructuredFormatter())
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ConsoleFormatter())
    logger.addHandler(console_handler)
    
    # Extra attribute wrapper
    logger = logging.LoggerAdapter(logger, {'component': component})
    return logger.logger # Returning raw logger for typical use

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"vaani.{name}")
