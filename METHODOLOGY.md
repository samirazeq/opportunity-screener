# Methodology v2.0

## TRADE (days to weeks)

Score = 30% trend + 25% medium-term momentum + 20% relative strength vs local benchmark + 10% volume confirmation + 10% entry/setup quality + 5% risk quality.

Recommendation bands:
- 7.5–10: BUY
- 6.2–7.4: WAIT
- 4.1–6.1: AVOID
- 0–4.0: SELL

## INVEST (12–36 months)

Score = 30% quality + 25% growth + 25% peer-relative value + 10% momentum confirmation + 10% risk quality.

Quality uses available profitability/cash-generation measures such as ROE, operating margin, profit margin, free-cash-flow yield and balance-sheet leverage. Financial companies are not penalized using the same debt/equity rule as industrial companies.

Value is ranked relative to peers using available forward P/E, price/book, EV/EBITDA and dividend yield. Peer groups use the same sector when enough names exist, otherwise the same local market.

Recommendation bands:
- 7.5–10: BUY
- 6.2–7.4: HOLD
- 4.1–6.1: AVOID
- 0–4.0: SELL

## Gold & Silver

Gold and silver use a separate model. Trend and momentum are combined with the direction of the US dollar and Treasury yields. Silver additionally includes copper direction as an industrial-demand proxy.

## Targets

Trade target/stop levels are mechanical ATR/trend levels and are intended as risk-map references.

Investment targets use Yahoo analyst consensus when available. If unavailable, the system only generates a peer-value estimate when there is positive forward EPS and enough comparable forward-P/E observations. If neither is defensible, it shows no target instead of fabricating one.

## Design principle

The recommendation and the short reason are deterministic. An LLM may add context from headlines, but cannot change BUY/WAIT/HOLD/AVOID/SELL.
