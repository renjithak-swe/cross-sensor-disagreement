"""JSON log lines on stderr, each tagged with the run identifier."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def __init__(self, run_id: str) -> None:
        super().__init__()
        self._run_id = run_id

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "run_id": self._run_id,
            "message": record.getMessage(),
        }
        payload.update({key: value for key, value in record.__dict__.items() if key not in _RESERVED})
        if record.exc_info:
            payload["error"] = repr(record.exc_info[1])
        return json.dumps(payload, default=str)


def configure_logging(run_id: str, level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter(run_id))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("rasterio", "urllib3", "pystac_client", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
