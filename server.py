# server.py - ASGI app export for uvicorn (DO NOT RENAME)
# Supervisor runs: uvicorn server:app --host 0.0.0.0 --port 8001
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'previsit.settings')
django.setup()

from previsit.asgi import application as app
