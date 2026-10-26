"""Explicit repository location, independent of the working directory."""
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("FAKTENRETENTION_PROJECT", Path(__file__).resolve().parents[2])).resolve()
