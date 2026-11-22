"""Shared helpers for eval and benchmarks: environment capture and safe results.json updates."""
from __future__ import annotations

import json
import os
import platform
import tempfile
import time
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results" / "results.json"


def collect_env() -> dict:
    cpu = platform.processor() or ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    mem_gb = None
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                mem_gb = round(int(line.split()[1]) / 1024 / 1024, 2)
                break
    except OSError:
        pass
    return {"platform": platform.platform(), "python": platform.python_version(), "cpu_model": cpu,
            "logical_cpus": os.cpu_count(), "ram_gb": mem_gb}


def update_results(section: str, payload: dict, path: Path = RESULTS) -> None:
    """Replace ``section`` in results.json atomically, leaving every other section untouched."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    payload = {**payload, "status": "ok", "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    data[section] = payload
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
