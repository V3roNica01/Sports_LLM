"""Claude analysis layer.

Takes the statistical model's probabilities, the engineered features, and any
market odds, and produces a written pre-match analysis: where the model edge
comes from, what the numbers say, and what could invalidate the read.

Requires ANTHROPIC_API_KEY in the environment.
"""

from __future__ import annotations

import json

import anthropic

from sports_llm.config import ANTHROPIC_MODEL

SYSTEM_PROMPT = """You are a quantitative football (soccer) betting analyst.

You are given pre-match data for one fixture: model win/draw/loss
probabilities from a gradient-boosted model trained on xG and Elo features,
the underlying team form numbers, and (optionally) bookmaker odds with any
flagged value bets.

Write a concise pre-match analysis:
1. Lead with the model's view and how confident it is.
2. Explain WHERE the edge comes from in the underlying numbers (xG form,
   Elo gap, home advantage) — not just that it exists.
3. Flag what the model cannot see (injuries, motivation, lineup rotation,
   weather) that could invalidate the read.
4. If value bets are flagged, comment on whether the edge looks robust or
   like model noise, and note the Kelly stake is a full-Kelly maximum —
   sensible bankroll management uses a fraction of it.

Be direct and numeric. Never present the analysis as a guarantee; these are
probabilistic estimates. Do not encourage chasing losses or betting beyond
one's means."""


def analyze_match(context: dict) -> str:
    """Run the Claude analysis over a match-context dict and return the text.

    context should contain keys like: home_team, away_team, model_probs,
    features (form/Elo numbers), market_odds, value_bets.
    """
    client = anthropic.Anthropic()
    with client.messages.stream(
        model=ANTHROPIC_MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": (
                    "Analyze this fixture:\n\n"
                    + json.dumps(context, indent=2, default=str)
                ),
            }
        ],
    ) as stream:
        message = stream.get_final_message()

    return next(b.text for b in message.content if b.type == "text")
