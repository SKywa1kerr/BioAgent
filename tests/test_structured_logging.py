"""Verify the JSON-line logging shim emits valid JSON per record.

The Electron-side "Export debug log" feature appends raw stderr lines to a
file; downstream tooling treats each line as a JSON object. This test
locks down that contract.
"""
from __future__ import annotations

import json
import logging
from io import StringIO

from bioagent.logger import JsonFormatter, configure, get_logger


def _attach_buffer(name: str) -> tuple[logging.Logger, StringIO, logging.Handler]:
    """Attach a fresh StringIO handler with the JSON formatter to `name`."""
    log = get_logger(name)
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter())
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    # Don't double-emit through root (which configure() pointed at stderr).
    log.propagate = False
    return log, buf, handler


def test_emits_valid_json_per_line():
    configure()
    log, buf, handler = _attach_buffer("bioagent.test.json")
    try:
        log.info("hello", extra={"phase": "x", "ms": 320})
    finally:
        log.removeHandler(handler)

    line = buf.getvalue().strip()
    assert "\n" not in line, "log call must produce exactly one line"
    obj = json.loads(line)
    assert obj["level"] == "info"
    assert obj["event"] == "hello"
    assert obj["logger"] == "bioagent.test.json"
    assert obj["extra"]["phase"] == "x"
    assert obj["extra"]["ms"] == 320
    assert "ts" in obj and obj["ts"].endswith("+00:00")


def test_warning_and_error_levels_round_trip():
    configure()
    log, buf, handler = _attach_buffer("bioagent.test.levels")
    try:
        log.warning("rate limited", extra={"attempt": 2})
        log.error("boom")
    finally:
        log.removeHandler(handler)

    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 2
    warn_obj = json.loads(lines[0])
    err_obj = json.loads(lines[1])
    assert warn_obj["level"] == "warning"
    assert warn_obj["event"] == "rate limited"
    assert warn_obj["extra"] == {"attempt": 2}
    assert err_obj["level"] == "error"
    assert err_obj["event"] == "boom"
    assert "extra" not in err_obj  # no extras emitted -> key omitted


def test_configure_is_idempotent():
    """Calling configure() multiple times must not stack handlers on root."""
    configure()
    root = logging.getLogger()
    before = sum(1 for h in root.handlers if getattr(h, "_bioagent_json_handler", False))
    configure()
    configure()
    after = sum(1 for h in root.handlers if getattr(h, "_bioagent_json_handler", False))
    assert before == 1
    assert after == 1


def test_non_json_extra_value_does_not_break_output():
    """Path objects etc. should degrade to str, not raise."""
    from pathlib import Path

    configure()
    log, buf, handler = _attach_buffer("bioagent.test.coerce")
    try:
        log.info("path event", extra={"path": Path("/tmp/x")})
    finally:
        log.removeHandler(handler)

    obj = json.loads(buf.getvalue().strip())
    assert obj["event"] == "path event"
    # Path should have been coerced via default=str
    assert isinstance(obj["extra"]["path"], str)
    assert "x" in obj["extra"]["path"]


def test_exception_info_included():
    configure()
    log, buf, handler = _attach_buffer("bioagent.test.exc")
    try:
        try:
            raise ValueError("nope")
        except ValueError:
            log.exception("caught")
    finally:
        log.removeHandler(handler)

    obj = json.loads(buf.getvalue().strip())
    assert obj["event"] == "caught"
    assert obj["level"] == "error"
    assert "ValueError" in obj["exc_info"]
    assert "nope" in obj["exc_info"]
