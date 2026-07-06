"""Unit tests for the betting math — no network or data downloads needed."""

import pandas as pd
import pytest

from sports_llm.features.match_features import add_elo, add_rolling_form
from sports_llm.models.baseline import find_value_bets, kelly, remove_overround


def test_remove_overround_sums_to_one():
    fair = remove_overround({"home": 2.10, "draw": 3.40, "away": 3.60})
    assert sum(fair.values()) == pytest.approx(1.0)
    assert fair["home"] > fair["draw"]


def test_kelly_zero_without_edge():
    # 50% at even odds (2.0) is break-even — no stake
    assert kelly(0.5, 2.0) == 0.0
    # clear edge gives positive stake below 1
    assert 0 < kelly(0.6, 2.0) < 1


def test_find_value_bets_flags_edge():
    probs = {"home": 0.55, "draw": 0.25, "away": 0.20}
    odds = {"home": 2.20, "draw": 3.40, "away": 3.60}  # implied home ~43%
    bets = find_value_bets(probs, odds)
    assert bets and bets[0].outcome == "home"
    assert bets[0].edge > 0.05


def _sample_team_rows():
    rows = []
    for i, (h_goals, a_goals) in enumerate([(2, 0), (1, 1), (0, 3)]):
        date = f"2023-08-{10 + i:02d}"
        rows.append(dict(match_id=i, match_date=date, team="A", opponent="B",
                         is_home=True, xg_for=1.5, xg_against=0.8,
                         goals_for=h_goals, goals_against=a_goals))
        rows.append(dict(match_id=i, match_date=date, team="B", opponent="A",
                         is_home=False, xg_for=0.8, xg_against=1.5,
                         goals_for=a_goals, goals_against=h_goals))
    return pd.DataFrame(rows)


def test_elo_winner_gains_rating():
    df = add_elo(_sample_team_rows())
    # Team A won match 0, so its pre-match rating for match 1 must be higher
    a_match1 = df[(df["team"] == "A") & (df["match_id"] == 1)]["elo"].iloc[0]
    b_match1 = df[(df["team"] == "B") & (df["match_id"] == 1)]["elo"].iloc[0]
    assert a_match1 > 1500 > b_match1


def test_rolling_form_is_leak_free():
    df = add_rolling_form(_sample_team_rows())
    first_match_a = df[(df["team"] == "A") & (df["match_id"] == 0)]
    # No prior matches -> form must be NaN, not the current match's value
    assert first_match_a["goals_for_form"].isna().all()
