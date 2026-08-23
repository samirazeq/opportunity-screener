# Market Opportunity Dashboard

A deployable research dashboard covering Saudi stocks, US stocks, gold, and silver.

## What it does
- Fetches 1 year of market history through `yfinance`.
- Calculates Wilder RSI, 20/50/200-day averages, ATR, volume ratio, momentum and 52-week structure.
- Produces separate **Short-Term**, **Swing**, and **Investment** scores from 0–10.
- Generates a ranked cross-market opportunity view.
- Shows mechanical ATR-based research entry/invalidation/objective levels.
- Pulls recent headlines for the top candidates.
- Optionally asks Claude to explain the top candidates, while keeping the scoring rule-based.
- Outputs a static webpage to `public/index.html` that works well with GitHub Pages.

## Run locally
```bash
pip install -r requirements.txt
python dashboard.py
```
Then open `public/index.html`.

## Optional Claude analysis
Add an environment variable named `ANTHROPIC_API_KEY` before running. On GitHub, add it under **Settings → Secrets and variables → Actions → New repository secret**.

## Put it online with GitHub Pages
1. Upload this project to a GitHub repository.
2. In **Settings → Pages**, choose **Deploy from a branch**.
3. Select your main branch and `/public` folder if GitHub offers it. If your Pages UI only supports `/root` or `/docs`, rename `public` to `docs` and change the two output paths in `dashboard.py` and the workflow accordingly.
4. Enable Actions. The included workflow refreshes the data hourly between 05:00 and 22:00 UTC and can also be run manually.

## Important prototype limitation
`yfinance` is convenient for US equities and metals but Saudi/Tadawul coverage can be incomplete or delayed. The code intentionally keeps data retrieval separate from scoring so a licensed Saudi-market data provider can replace it later without rebuilding the dashboard.

## Scoring philosophy
The score engine—not Claude—determines ranking. Claude only receives the highest-ranked assets and explains catalysts, context, and risks. This preserves repeatability and avoids AI-picked securities without quantitative support.
