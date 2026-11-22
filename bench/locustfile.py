"""Locust alternative (interactive UI / distributed load):
  DOC=eval/docs/yourdoc.pdf locust -f bench/locustfile.py --host http://127.0.0.1:8000 \
      --users 20 --spawn-rate 5 --run-time 60s --headless --csv results/locust
Reports p50/p95, failures and req/s per endpoint. TTFT is only measured by bench/load_test.py."""
import json
import os
import random
from pathlib import Path

from locust import HttpUser, between, task

DOC = os.environ["DOC"]
QUESTIONS = ["Summarize the main topic of the document."]
if os.getenv("QUESTIONS"):
    QUESTIONS = [json.loads(x)["question"] for x in Path(os.environ["QUESTIONS"]).read_text().splitlines() if x.strip()]


class DocUser(HttpUser):
    wait_time = between(0, 0)

    def on_start(self):
        with open(DOC, "rb") as f:
            self.client.post("/api/upload", files={"file": (Path(DOC).name, f)})

    @task
    def ask(self):
        self.client.post("/api/chat", json={"message": random.choice(QUESTIONS)}, name="/api/chat")
