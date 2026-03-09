# Gunicorn Configuration File
import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
backlog = 2048

# Worker processes
# To avoid Out of Memory (OOM) errors on free tiers, hardcode to 1 worker
workers = int(os.getenv('WEB_CONCURRENCY', '1'))
worker_class = 'gthread'
threads = 2
worker_connections = 1000
timeout = 30
keepalive = 2
preload_app = True

# Logging
accesslog = '-'
errorlog = '-'
loglevel = os.getenv('LOG_LEVEL', 'info')
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = 'ckd_prediction_api'

# Server mechanics
daemon = False
pidfile = 'gunicorn.pid'
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (if needed)
# keyfile = 'path/to/keyfile'
# certfile = 'path/to/certfile'
