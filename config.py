"""
Configuration module for the Handbook Generator application.
Loads environment variables and provides centralized config access.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── API Keys ────────────────────────────────────────────────
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
XAI_MODEL = os.getenv("XAI_MODEL", "grok-4-1")

# ─── Supabase ────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# ─── LightRAG Settings ──────────────────────────────────────
LIGHTRAG_WORKING_DIR = os.getenv("LIGHTRAG_WORKING_DIR", "./lightrag_data")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ─── Generation Settings ─────────────────────────────────────
MAX_TOKENS_PER_CHUNK = 4096
HANDBOOK_TARGET_WORDS = 20000
PARAGRAPH_MIN_WORDS = 200
PARAGRAPH_MAX_WORDS = 1000
TEMPERATURE = 0.7
MAX_RETRIES = 3

# ─── Upload Settings ─────────────────────────────────────────
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(LIGHTRAG_WORKING_DIR, exist_ok=True)
