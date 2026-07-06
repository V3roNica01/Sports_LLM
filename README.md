# Sports_LLM

A sports betting prediction engine that combines rich event-level football data with statistical modeling and Claude-powered analysis.

## How it works

```
StatsBomb open data ─┐
                     ├─> feature engineering ─> outcome model ─> value-bet detection ─> Claude analysis
FBref scraping ──────┘      (xG form, Elo)      (P(H/D/A))        (edge vs. odds)       (written read)
```

1. **Data** (`sports_llm.data`)
   - `statsbomb` — event-level data (every shot with xG) from the free [StatsBomb open data](https://github.com/statsbomb/open-data) via `statsbombpy`, cached to parquet.
   - `fbref` — current-season fixtures and league tables scraped from [FBref](https://fbref.com), rate-limited to respect their 10 requests/minute bot policy and cached to disk.
2. **Features** (`sports_llm.features`) — pre-match, leak-free features per fixture: rolling xG for/against, goal form, and Elo ratings.
3. **Model** (`sports_llm.models`) — gradient-boosted classifier producing home/draw/away probabilities, evaluated with time-series cross-validated log loss. Value-bet math strips the bookmaker overround and computes edge + Kelly stake.
4. **Analysis** (`sports_llm.llm`) — Claude (Opus 4.8) turns the numbers into a written pre-match read: where the edge comes from, and what the model can't see.

## Quickstart

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...   # optional, enables the Claude analysis step
python examples/quickstart.py
```

The first run downloads a full season of StatsBomb events (a few minutes); everything caches under `data/`.

Run the tests (no network needed):

```bash
pytest
```

## Data sources & attribution

- **StatsBomb**: free for research/non-commercial use. Published work must credit StatsBomb as the data source and use their logo.
- **FBref / Sports Reference**: scraped politely — one request per 6+ seconds, aggressive caching. Don't remove the rate limiter.

## Disclaimer

This project produces **probabilistic estimates for research purposes, not financial advice**. Sports outcomes are inherently uncertain; no model guarantees profit. If you bet, only stake what you can afford to lose — the Kelly fractions reported are full-Kelly *maximums*, and practitioners typically stake a quarter of that or less. Check the legality of sports betting in your jurisdiction.
