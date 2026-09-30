"""Vercel entrypoint for the existing FastAPI application."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

if os.environ.get("VERCEL"):
    os.environ.setdefault("CODELENS_INDEX_DIR", str(ROOT / "data" / "indexes"))
    os.environ.setdefault("CODELENS_VERSION_DIR", str(ROOT / "data" / "versions"))
    # The checked-in demo corpus can be searched, but serverless instances must not
    # accept arbitrary filesystem indexing requests or depend on ephemeral writes.
    os.environ.setdefault("CODELENS_READ_ONLY", "true")

from app.main import app  # noqa: E402
