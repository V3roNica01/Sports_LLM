"""End-to-end quickstart on free StatsBomb data.

Trains the baseline model on a full season, predicts one held-out match,
checks the prediction against example odds, and (if ANTHROPIC_API_KEY is set)
asks Claude for a written analysis.

Run:  python examples/quickstart.py
First run downloads events for every match in the season (a few minutes);
everything is cached under data/ afterwards.
"""

import os

from sports_llm.data import statsbomb
from sports_llm.features import build_match_features
from sports_llm.models import MatchOutcomeModel, find_value_bets

# La Liga 2019/20 in the StatsBomb open data (competition_id=11, season_id=42)
COMPETITION_ID, SEASON_ID = 11, 42


def main() -> None:
    print("Loading StatsBomb open data (cached after first run)...")
    team_rows = statsbomb.season_team_xg(COMPETITION_ID, SEASON_ID)
    features = build_match_features(team_rows)
    print(f"Built features for {len(features)} matches")

    train, holdout = features.iloc[:-1], features.iloc[[-1]]
    model = MatchOutcomeModel().fit(train)
    print(f"Time-series CV log loss: {model.evaluate(train):.4f}")

    probs = model.predict_proba(holdout).iloc[0].to_dict()
    match = holdout.iloc[0]
    print(f"\n{match['home_team']} vs {match['away_team']}")
    print({k: round(v, 3) for k, v in probs.items()})

    # Example market odds — replace with a live odds feed.
    market_odds = {"home": 2.10, "draw": 3.40, "away": 3.60}
    value = find_value_bets(probs, market_odds)
    for bet in value:
        print(f"VALUE: {bet.outcome} @ {bet.decimal_odds} "
              f"(model {bet.model_prob:.1%} vs implied {bet.implied_prob:.1%}, "
              f"edge {bet.edge:.1%}, kelly {bet.kelly_fraction:.1%})")
    if not value:
        print("No value found at these odds.")

    if os.environ.get("ANTHROPIC_API_KEY"):
        from sports_llm.llm import analyze_match

        print("\n--- Claude analysis ---")
        print(analyze_match({
            "home_team": match["home_team"],
            "away_team": match["away_team"],
            "model_probs": probs,
            "features": match.drop(["match_id", "result"]).to_dict(),
            "market_odds": market_odds,
            "value_bets": [vars(b) for b in value],
        }))
    else:
        print("\nSet ANTHROPIC_API_KEY to enable the Claude analysis step.")


if __name__ == "__main__":
    main()
