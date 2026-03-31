"""gunicorn.conf.py — Gunicorn configuration for production deployment.

PORT is read from the environment so Railway (and other PaaS platforms) can
inject the port at runtime.  Never hardcode a port value here.

Usage::

    gunicorn app.main:app -c gunicorn.conf.py
"""

import os

# Bind to the platform-injected PORT; fall back to 8000 for local runs.
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# 2 workers is conservative for a CPU-bound pipeline on a small dyno.
# Increase if the host has more cores and memory.
workers = 2

# UvicornWorker gives async support required by FastAPI.
worker_class = "uvicorn.workers.UvicornWorker"

# Long timeout for OHLCV + strategy computation (yfinance fallback is slow).
timeout = 120

# Restart workers after this many requests to prevent memory leaks.
max_requests = 500
max_requests_jitter = 50

# Access log to stdout so Railway captures it.
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info").lower()
