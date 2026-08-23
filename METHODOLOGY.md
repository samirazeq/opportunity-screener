# Market Opportunity Dashboard V3 — Methodology

## Goal
V3 is designed to answer one practical question: **is this asset worth buying now, waiting for, holding, avoiding, or selling — and why?** The page stays simple; the engine underneath evaluates several distinct opportunity types instead of forcing every stock through one momentum formula.

## What V3 learned from established systems
- **TradingView Technical Ratings:** technical confluence is more useful than a single indicator. TradingView combines moving averages and oscillators into a normalized rating. V3 uses the same principle (trend + momentum + relative strength + volume + setup), but does not copy TradingView's exact rating.
- **Seeking Alpha Quant / AAII A+ Grades:** long-term stock selection should combine multiple factor families such as Value, Growth, Profitability/Quality, Momentum and EPS/estimate information, with peer-relative comparisons rather than universal raw cutoffs.
- **MSCI / S&P factor methodologies:** value, quality and momentum are established factor families. Sector-relative normalization is particularly important for valuation and quality.
- **Morningstar:** a cheap price is not enough; uncertainty and margin of safety matter. V3 therefore does not award a BUY merely because a stock has fallen or looks cheap.
- **Danelfin:** the useful AI idea is to evaluate many feature families, compare against a benchmark, learn from historical outcomes, and explain positive/negative signals. V3 adopts the architecture idea, not Danelfin's proprietary model or claimed performance.

## Opportunity paths
Every equity is checked simultaneously for:
1. **Momentum / continuation** — strong trend, 6–12 month momentum and market-relative strength.
2. **Breakout** — strong trend/momentum with price near a 52-week high and volume confirmation.
3. **Pullback** — established trend but price has returned to a more attractive entry area.
4. **Recovery** — meaningful historical drawdown followed by stabilization/improving momentum, volume and fundamentals.
5. **Quality + Value** — strong profitability/financial quality with attractive peer-relative valuation.
6. **Growth + Value** — strong growth and quality without excessive peer-relative valuation.

The highest-quality path can surface an opportunity, but contradictory evidence can block a BUY.

## Falling stock vs recovery
A large decline by itself receives **no BUY credit**. A recovery score requires confirmation: improving 1–3 month price behavior, constructive volume and/or improving revenue/earnings. Continued earnings deterioration creates a **value-trap penalty**.

## TRADE recommendation
Trade scoring combines trend, risk-adjusted momentum, relative strength versus the local benchmark, volume, setup quality and risk. Breakout, pullback and recovery models can also surface candidates.

A high score is still **not automatically BUY**. BUY requires the live/last price to be inside the calculated preferred entry zone and the stop to be below that zone. A strong stock above its entry becomes **WAIT**, not BUY.

## INVEST recommendation
Investment scoring combines:
- Quality / profitability
- Growth
- Peer-relative value
- Momentum confirmation
- Risk
- Recovery confirmation when applicable

Likely value traps are penalized. Missing fundamental data lowers Data Quality; it is not silently treated as positive evidence.

## Historical context
The engine requests up to five years of price history and measures 1M, 3M, 6M, 12M and longer-run context where available, 52-week position, drawdowns, trend and momentum. Historical lows/highs are context — not automatic buy/sell rules.

## Gold and silver
Precious metals use a separate model. Trend and momentum are combined with the direction of the US dollar and US 10-year Treasury yield; silver also uses copper as an industrial-cycle proxy. Metals are never scored as if they were companies.

## Targets
- **Trade:** ATR/trend-based preferred entry, invalidation and objective. V3 enforces stop < entry for long BUY setups.
- **Investment:** analyst consensus target when available and plausible; otherwise a peer forward-P/E estimate only when sufficient comparable data exists. If neither is defensible, the dashboard shows no target.

## Data limitations
The free V3 build uses yfinance. US coverage is generally better than Tadawul fundamentals/analyst data. Missing Saudi data is explicitly reflected in Data Quality. For production-grade Saudi coverage, replace the provider layer with licensed Saudi Exchange/market-data feeds without changing the scoring architecture.

## Research references
- TradingView, Technical Ratings: https://www.tradingview.com/support/solutions/43000614331-technical-ratings/
- Seeking Alpha, Quant Ratings: https://help.seekingalpha.com/premium/what-are-quant-ratings-and-how-do-i-use-them
- AAII, A+ Stock Grades: https://www.aaii.com/plus/article/16561-aaii-stock-screen-grades
- MSCI Factor Indexes / Momentum: https://www.msci.com/documents/10199/242721/MSCI_Factor_Indices.pdf/74fe7772-583f-402a-8d91-49a2525d9f0c
- Morningstar stock valuation framework: https://www.morningstar.com/stocks/how-determine-whether-stock-is-cheap-expensive-or-fairly-valued
- Danelfin methodology: https://danelfin.com/how-it-works
