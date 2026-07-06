"""Central configuration for data paths and scraping behavior."""

from __future__ import annotations

import os
from pathlib import Path

# Root for all cached data. Override with SPORTS_LLM_DATA_DIR.
DATA_DIR = Path(os.environ.get("SPORTS_LLM_DATA_DIR", "data"))
STATSBOMB_CACHE = DATA_DIR / "statsbomb"
FBREF_CACHE = DATA_DIR / "fbref"

# FBref (sports-reference.com) bot policy allows at most 10 requests/minute.
# Exceeding it earns an IP block of up to a day, so we stay at 6s+ between hits.
FBREF_MIN_REQUEST_INTERVAL = 6.0
FBREF_USER_AGENT = (
    "sports-llm/0.1 (research project; respects robots.txt; contact via GitHub)"
)

# Claude model used for the analysis layer.
ANTHROPIC_MODEL = "claude-opus-4-8"


def ensure_dirs() -> None:
    for d in (STATSBOMB_CACHE, FBREF_CACHE):
        d.mkdir(parents=True, exist_ok=True)
