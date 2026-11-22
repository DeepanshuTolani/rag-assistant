import os

bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
# Session state (history + per-session FAISS index) lives in process memory, so use ONE worker
# with many threads. Scale vertically, or add sticky routing + external state before adding workers.
workers = 1
worker_class = "gthread"
threads = int(os.getenv("GUNICORN_THREADS", "16"))
timeout = 120          # > LLM read timeout so streams are not killed mid-answer
graceful_timeout = 30
accesslog = None       # request logs come from the app's structured JSON logger
errorlog = "-"
preload_app = False
