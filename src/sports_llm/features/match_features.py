"""Turn per-team match rows into model-ready features.

Input is the long-format frame from sports_llm.data.statsbomb.season_team_xg
(one row per match per team). Output is one row per match with pre-match
features for both sides: rolling xG form, goal form, and Elo ratings.

All rolling stats are shifted by one match so a row only ever contains
information available *before* kickoff — no leakage.
"""

from __future__ import annotations

import pandas as pd

ELO_START = 1500.0
ELO_K = 24.0
FORM_WINDOW = 5


def _elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def add_elo(team_rows: pd.DataFrame) -> pd.DataFrame:
    """Attach a pre-match Elo rating to every (match, team) row."""
    df = team_rows.sort_values("match_date").copy()
    ratings: dict[str, float] = {}
    pre_match = []
    for match_id, grp in df.groupby("match_id", sort=False):
        a, b = grp.iloc[0], grp.iloc[1]
        ra = ratings.get(a["team"], ELO_START)
        rb = ratings.get(b["team"], ELO_START)
        pre_match.append((match_id, a["team"], ra))
        pre_match.append((match_id, b["team"], rb))
        score_a = 0.5 if a["goals_for"] == a["goals_against"] else float(a["goals_for"] > a["goals_against"])
        exp_a = _elo_expected(ra, rb)
        ratings[a["team"]] = ra + ELO_K * (score_a - exp_a)
        ratings[b["team"]] = rb + ELO_K * ((1 - score_a) - (1 - exp_a))
    elo = pd.DataFrame(pre_match, columns=["match_id", "team", "elo"])
    return df.merge(elo, on=["match_id", "team"])


def add_rolling_form(team_rows: pd.DataFrame, window: int = FORM_WINDOW) -> pd.DataFrame:
    """Rolling means of xG and goals over the previous `window` matches."""
    df = team_rows.sort_values(["team", "match_date"]).copy()
    grouped = df.groupby("team")
    for col in ("xg_for", "xg_against", "goals_for", "goals_against"):
        df[f"{col}_form"] = (
            grouped[col].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
    return df


def build_match_features(team_rows: pd.DataFrame) -> pd.DataFrame:
    """One row per match: home_/away_ prefixed features + result label.

    Label: 0 = home win, 1 = draw, 2 = away win.
    """
    df = add_rolling_form(add_elo(team_rows))
    home = df[df["is_home"]].set_index("match_id")
    away = df[~df["is_home"]].set_index("match_id")

    feature_cols = ["elo", "xg_for_form", "xg_against_form", "goals_for_form", "goals_against_form"]
    out = pd.DataFrame(index=home.index)
    out["match_date"] = home["match_date"]
    out["home_team"] = home["team"]
    out["away_team"] = away["team"]
    for col in feature_cols:
        out[f"home_{col}"] = home[col]
        out[f"away_{col}"] = away[col]
    out["elo_diff"] = out["home_elo"] - out["away_elo"]

    out["result"] = 1  # draw
    out.loc[home["goals_for"] > home["goals_against"], "result"] = 0
    out.loc[home["goals_for"] < home["goals_against"], "result"] = 2
    return out.dropna().reset_index()
