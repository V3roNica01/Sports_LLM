"""Baseline match-outcome model + value-bet math.

A gradient-boosted classifier over the engineered features gives calibrated-ish
probabilities for home/draw/away. Value detection compares model probabilities
against bookmaker implied probabilities (1/decimal odds, overround removed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

FEATURE_COLS = [
    "home_elo", "away_elo", "elo_diff",
    "home_xg_for_form", "home_xg_against_form",
    "away_xg_for_form", "away_xg_against_form",
    "home_goals_for_form", "home_goals_against_form",
    "away_goals_for_form", "away_goals_against_form",
]

OUTCOMES = ("home", "draw", "away")


class MatchOutcomeModel:
    def __init__(self) -> None:
        self.clf = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.08, max_iter=300, l2_regularization=1.0
        )

    def fit(self, features: pd.DataFrame) -> "MatchOutcomeModel":
        X = features[FEATURE_COLS]
        y = features["result"]
        self.clf.fit(X, y)
        return self

    def evaluate(self, features: pd.DataFrame, splits: int = 5) -> float:
        """Time-series cross-validated log loss (lower is better).

        Uses time-ordered splits — random CV would leak future form into
        the past and overstate accuracy.
        """
        df = features.sort_values("match_date")
        scores = cross_val_score(
            self.clf, df[FEATURE_COLS], df["result"],
            cv=TimeSeriesSplit(n_splits=splits), scoring="neg_log_loss",
        )
        return float(-scores.mean())

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        proba = self.clf.predict_proba(features[FEATURE_COLS])
        return pd.DataFrame(proba, columns=list(OUTCOMES), index=features.index)


@dataclass
class ValueBet:
    outcome: str
    model_prob: float
    implied_prob: float
    decimal_odds: float
    edge: float          # model_prob - implied_prob
    kelly_fraction: float  # suggested stake as fraction of bankroll (full Kelly)


def remove_overround(odds: dict[str, float]) -> dict[str, float]:
    """Convert decimal odds to fair implied probabilities (margin stripped)."""
    raw = {k: 1.0 / v for k, v in odds.items()}
    total = sum(raw.values())
    return {k: p / total for k, p in raw.items()}


def kelly(prob: float, decimal_odds: float) -> float:
    """Full Kelly criterion stake fraction; 0 if no edge."""
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    f = (prob * (b + 1.0) - 1.0) / b
    return max(0.0, f)


def find_value_bets(
    model_probs: dict[str, float],
    market_odds: dict[str, float],
    min_edge: float = 0.03,
) -> list[ValueBet]:
    """Compare model probabilities to market odds; return outcomes with edge.

    model_probs / market_odds keys: "home", "draw", "away".
    min_edge: minimum probability edge to flag (default 3 points).
    """
    implied = remove_overround(market_odds)
    bets = []
    for outcome in OUTCOMES:
        p, o = model_probs[outcome], market_odds[outcome]
        edge = p - implied[outcome]
        if edge >= min_edge:
            bets.append(
                ValueBet(
                    outcome=outcome,
                    model_prob=round(p, 4),
                    implied_prob=round(implied[outcome], 4),
                    decimal_odds=o,
                    edge=round(edge, 4),
                    kelly_fraction=round(kelly(p, o), 4),
                )
            )
    return sorted(bets, key=lambda b: b.edge, reverse=True)
