import logging
import sys
from typing import Optional


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "message": record.getMessage(),
            "name": record.name,
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
        }
        for key, value in record.__dict__.items():
            if key.startswith("ctx_"):
                payload[key[4:]] = value
        return json_dumps(payload)


def json_dumps(data) -> str:
    try:
        import json

        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return str(data)


def get_logger(name: str = "app", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())

    logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False
    return logger


__all__ = ["get_logger"]
