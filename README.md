# Market Opportunity Dashboard V3.0

A deployable research dashboard for Saudi stocks, US stocks, gold and silver.

## What changed in V3
V3 is no longer a momentum-first screener. It scans several opportunity types at the same time: momentum, breakout, pullback, recovery, quality/value and growth/value. It uses up to five years of price history, peer-relative fundamentals, explicit value-trap checks and entry-aware recommendations.

The visible output stays simple:
- **TRADE:** BUY / WAIT / AVOID / SELL + entry + target + stop + one reason.
- **INVEST:** BUY / HOLD / AVOID / SELL + target when defensible + one reason.

A stock can score well but still show **WAIT** if the current price is above the preferred entry. A deeply fallen stock does not become a BUY unless recovery evidence appears.

## Run
```bash
pip install -r requirements.txt
python dashboard.py
```
Open `docs/index.html`.

## GitHub Pages
The included workflow builds and deploys `docs/` directly with GitHub Actions. In **Settings → Pages**, set **Source = GitHub Actions**.

## Optional AI context
Add `ANTHROPIC_API_KEY` as a GitHub Actions repository secret. AI only summarizes recent catalysts/risks; it cannot change the deterministic recommendation.

## Important
This is a research tool, not a guarantee of returns. The free data layer is `yfinance`; Tadawul fundamental/analyst coverage can be incomplete. See `METHODOLOGY.md` for the model and source references.
