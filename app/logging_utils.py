"""Structured JSON-lines logging. One line per request, with latency, so benchmark numbers
can be derived from real logs (see bench/summarize_logs.py)."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        out.update(getattr(record, "fields", {}))
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def setup_logging(log_dir: str) -> logging.Logger:
    logger = logging.getLogger("rag")
    for h in list(logger.handlers):  # idempotent: re-point handlers if called again (tests, reloads)
        logger.removeHandler(h)
        h.close()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = JsonFormatter()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    try:
        os.makedirs(log_dir, exist_ok=True)
        fh = RotatingFileHandler(os.path.join(log_dir, "requests.jsonl"), maxBytes=20_000_000,
                                 backupCount=5)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError:
        logger.warning("file logging disabled (cannot write to %s)", log_dir)
    return logger


def log_event(msg: str, **fields) -> None:
    logging.getLogger("rag").info(msg, extra={"fields": fields})
