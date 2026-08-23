# Market Opportunity Dashboard v2.0

A clear research dashboard for **Saudi stocks, US stocks, gold and silver**.

## What changed in V2

- Two simple modes: **TRADE** and **INVEST**.
- Direct recommendations: **BUY / WAIT / HOLD / AVOID / SELL**.
- Every recommendation includes **one short reason**.
- Trade view shows **preferred entry, target, stop and risk/reward**.
- Investment view shows a **12-month target when defensible** and a 12–36 month horizon.
- Equity scoring uses established factor families instead of an arbitrary single score:
  - Trade: trend, medium-term momentum, relative strength, volume confirmation, setup quality and risk.
  - Invest: quality/profitability, growth, peer-relative valuation, momentum and risk.
- Gold and silver use a separate trend/macro model including the **US dollar and US Treasury-yield direction**; silver also includes copper as an industrial-demand proxy.
- AI is optional and **cannot change the recommendation**. It only adds one short news/catalyst sentence.
- GitHub Pages deployment now uses **GitHub Actions directly**. The workflow does not commit generated market-data files back to your repository, eliminating the merge conflicts from V1.

## Data

The project uses `yfinance` as a free prototype provider. It is generally useful for US securities and futures, but Saudi/Tadawul coverage can be incomplete or delayed. Always verify the live execution price with your broker before placing an order.

## Run locally

```bash
pip install -r requirements.txt
python dashboard.py
```

Open `docs/index.html`.

## GitHub Pages setup

After pushing V2 to GitHub:

1. Open **Settings → Pages**.
2. Under **Build and deployment → Source**, choose **GitHub Actions**.
3. Open **Actions → Build and deploy market dashboard → Run workflow** once.
4. The same Pages URL will serve the new dashboard.

## Optional AI context

Add `ANTHROPIC_API_KEY` under **Settings → Secrets and variables → Actions**. The core recommendations work without it.

## Research basis

The model intentionally uses widely established factor concepts rather than inventing a new investment theory. Quality is represented by profitability, growth, safety/balance-sheet and cash-generation measures; value is measured relative to peers; medium-term momentum and relative strength drive the trade model. Metals use their own macro/trend model because equity fundamentals do not apply to them.

No model can guarantee investment outcomes. The dashboard is designed to make the decision process transparent, consistent and easy to audit.
