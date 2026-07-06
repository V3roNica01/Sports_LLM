"""FBref scraping — polite, rate-limited, cached.

FBref covers current seasons and betting-relevant tables (fixtures, league
tables, squad stats) that the StatsBomb open data doesn't. Sports Reference's
bot policy caps traffic at 10 requests/minute; we enforce a 6s minimum gap
between requests and cache every page to disk, so a re-run costs zero hits.
"""

from __future__ import annotations

import hashlib
import io
import time

import pandas as pd
import requests

from sports_llm.config import (
    FBREF_CACHE,
    FBREF_MIN_REQUEST_INTERVAL,
    FBREF_USER_AGENT,
    ensure_dirs,
)

BASE = "https://fbref.com"

# FBref competition ids for the big-five leagues + a few extras.
COMPETITIONS = {
    "premier-league": (9, "Premier-League"),
    "la-liga": (12, "La-Liga"),
    "serie-a": (11, "Serie-A"),
    "bundesliga": (20, "Bundesliga"),
    "ligue-1": (13, "Ligue-1"),
    "championship": (10, "Championship"),
    "mls": (22, "Major-League-Soccer"),
}

_last_request_time = 0.0


def _fetch_html(url: str) -> str:
    """Fetch a page, respecting the rate limit and an on-disk cache."""
    global _last_request_time
    ensure_dirs()
    cache_path = FBREF_CACHE / (hashlib.sha1(url.encode()).hexdigest() + ".html")
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    wait = FBREF_MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request_time)
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(url, headers={"User-Agent": FBREF_USER_AGENT}, timeout=30)
    _last_request_time = time.monotonic()
    resp.raise_for_status()
    cache_path.write_text(resp.text, encoding="utf-8")
    return resp.text


def _read_tables(url: str) -> list[pd.DataFrame]:
    # FBref hides some tables inside HTML comments; strip the markers so
    # pandas can see them all.
    html = _fetch_html(url).replace("<!--", "").replace("-->", "")
    return pd.read_html(io.StringIO(html))


def fixtures(competition: str, season: str | None = None) -> pd.DataFrame:
    """Scores & fixtures table for a league.

    competition: key from COMPETITIONS (e.g. "premier-league")
    season: e.g. "2023-2024"; omit for the current season.
    """
    comp_id, slug = COMPETITIONS[competition]
    if season:
        url = f"{BASE}/en/comps/{comp_id}/{season}/schedule/{season}-{slug}-Scores-and-Fixtures"
    else:
        url = f"{BASE}/en/comps/{comp_id}/schedule/{slug}-Scores-and-Fixtures"
    df = _read_tables(url)[0]
    df = df.dropna(how="all").reset_index(drop=True)
    # Drop repeated header rows FBref embeds mid-table
    if "Wk" in df.columns:
        df = df[df["Wk"].astype(str) != "Wk"].reset_index(drop=True)
    return df


def league_table(competition: str, season: str | None = None) -> pd.DataFrame:
    """Current league standings, including xG columns where available."""
    comp_id, slug = COMPETITIONS[competition]
    if season:
        url = f"{BASE}/en/comps/{comp_id}/{season}/{season}-{slug}-Stats"
    else:
        url = f"{BASE}/en/comps/{comp_id}/{slug}-Stats"
    return _read_tables(url)[0]
