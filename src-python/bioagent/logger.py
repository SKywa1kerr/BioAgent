"""Structured JSON-line logging for the BioAgent Python sidecar.

Emits one JSON object per log record on stderr so that the Electron-side
"Export debug log" feature produces a greppable / jq-able artifact without
requiring any third-party dependencies.

Each line has the shape::

    {"ts": "2026-05-06T12:34:56.789012+00:00",
     "level": "info",
     "logger": "bioagent.alignment",
     "event": "<message>",
     "extra": {"phase": "init", "ms": 320}}

Standard library only. No structlog / rich / loguru.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

__all__ = ["configure", "get_logger", "JsonFormatter"]


# Attributes that the stdlib `logging.LogRecord` sets itself. Anything outside
# this set on `record.__dict__` was added by the caller via `extra=...` and
# belongs in the JSON `extra` bucket.
_RESERVED_RECORD_KEYS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


class JsonFormatter(logging.Formatter):
    """Formatter that renders a `LogRecord` as a single JSON line.

    Uses `record.created` (which is `time.time()` at record construction)
    so timestamps reflect when the log call happened, not when the record
    was formatted.
    """

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        payload: dict = {
            "ts": ts,
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }

        extra = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_RECORD_KEYS and not key.startswith("_")
        }
        if extra:
            payload["extra"] = extra

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = record.stack_info

        # `default=str` keeps the formatter total: any non-JSON-native value
        # in extra (e.g. Path, datetime) degrades gracefully to its str() form
        # rather than raising and dropping the log line.
        return json.dumps(payload, ensure_ascii=False, default=str)


_HANDLER_MARK = "_bioagent_json_handler"


def configure() -> None:
    """Install a JSON-line stderr handler on the root logger.

    Idempotent: a second call is a no-op (does not stack handlers).
    """

    root = logging.getLogger()

    for existing in root.handlers:
        if getattr(existing, _HANDLER_MARK, False):
            return  # already configured

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(JsonFormatter())
    setattr(handler, _HANDLER_MARK, True)
    root.addHandler(handler)

    # Default to INFO so lifecycle traces survive. Don't clobber a stricter
    # level already set by the embedding application.
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)


def get_logger(name: str = "bioagent") -> logging.Logger:
    """Return a named logger. After `configure()` it emits JSON lines."""

    return logging.getLogger(name)
