"""StatsBomb open-data ingestion.

Uses statsbombpy against the free open-data repo (no credentials needed).
Everything is cached to parquet under data/statsbomb/ so repeated runs
don't re-download.

License note: free for research/non-commercial use; published work must
credit StatsBomb as the data source.
"""

from __future__ import annotations

import time

import pandas as pd
import requests
from statsbombpy import sb

from sports_llm.config import STATSBOMB_CACHE, ensure_dirs


def _with_retry(loader, attempts: int = 5) -> pd.DataFrame:
    """Retry on 429/5xx with exponential backoff — GitHub's raw host
    rate-limits bulk season downloads."""
    for attempt in range(attempts):
        try:
            return loader()
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status not in (429, 500, 502, 503) or attempt == attempts - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("unreachable")


def _cached(name: str, loader) -> pd.DataFrame:
    ensure_dirs()
    path = STATSBOMB_CACHE / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = _with_retry(loader)
    try:
        df.to_parquet(path, index=False)
    except Exception:
        # StatsBomb frames contain mixed-type object columns (e.g.
        # home_manager_id is "4711, 3626" for co-managed teams, int elsewhere)
        # and nested lists. Stringify object columns so the cache write works.
        safe = df.copy()
        for col in safe.columns[safe.dtypes == object]:
            safe[col] = safe[col].astype(str)
        safe.to_parquet(path, index=False)
    return df


def competitions() -> pd.DataFrame:
    """All competition-seasons available in the open data."""
    return _cached("competitions", sb.competitions)


def matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """All matches for one competition-season (scores, teams, dates)."""
    return _cached(
        f"matches_{competition_id}_{season_id}",
        lambda: sb.matches(competition_id=competition_id, season_id=season_id),
    )


def events(match_id: int) -> pd.DataFrame:
    """Full event stream for one match (~3,400 rows: passes, shots, xG...)."""
    return _cached(f"events_{match_id}", lambda: sb.events(match_id=match_id))


def match_shots(match_id: int) -> pd.DataFrame:
    """Shots only, with xG — the core input for outcome modeling."""
    ev = events(match_id)
    shots = ev[ev["type"] == "Shot"].copy()
    cols = [c for c in ("team", "player", "minute", "shot_statsbomb_xg", "shot_outcome") if c in shots]
    return shots[cols]


def season_team_xg(competition_id: int, season_id: int) -> pd.DataFrame:
    """Per-match xG for/against for every team in a competition-season.

    Returns one row per (match_id, team) with columns:
    match_date, team, opponent, xg_for, xg_against, goals_for, goals_against.
    """
    ms = matches(competition_id, season_id)
    rows = []
    for _, m in ms.iterrows():
        shots = match_shots(m["match_id"])
        if shots.empty:
            continue
        xg = shots.groupby("team")["shot_statsbomb_xg"].sum()
        home, away = m["home_team"], m["away_team"]
        for team, opp, gf, ga in (
            (home, away, m["home_score"], m["away_score"]),
            (away, home, m["away_score"], m["home_score"]),
        ):
            rows.append(
                {
                    "match_id": m["match_id"],
                    "match_date": m["match_date"],
                    "team": team,
                    "opponent": opp,
                    "is_home": team == home,
                    "xg_for": float(xg.get(team, 0.0)),
                    "xg_against": float(xg.get(opp, 0.0)),
                    "goals_for": gf,
                    "goals_against": ga,
                }
            )
    return pd.DataFrame(rows).sort_values("match_date").reset_index(drop=True)
