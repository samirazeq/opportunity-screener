"""Market Opportunity Dashboard v1

Static research dashboard for US stocks, Saudi stocks, gold, and silver.
Data: yfinance (prototype provider). AI layer: optional Anthropic Claude.
The rule engine calculates the scores; the LLM only explains the highest-ranked setups.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

try:
    from anthropic import Anthropic
except Exception:  # optional at runtime
    Anthropic = None

# ----------------------------- Universe -----------------------------------
US_UNIVERSE = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon",
    "GOOGL": "Alphabet", "META": "Meta", "TSLA": "Tesla", "AMD": "AMD",
    "AVGO": "Broadcom", "NFLX": "Netflix", "JPM": "JPMorgan", "BAC": "Bank of America",
    "XOM": "Exxon Mobil", "CVX": "Chevron", "LLY": "Eli Lilly", "UNH": "UnitedHealth",
    "WMT": "Walmart", "COST": "Costco", "PLTR": "Palantir", "MRVL": "Marvell",
}

SAUDI_UNIVERSE = {
    "2222.SR": "Saudi Aramco", "1120.SR": "Al Rajhi Bank", "1180.SR": "Saudi National Bank",
    "2010.SR": "SABIC", "1211.SR": "Ma'aden", "7010.SR": "stc", "2020.SR": "SABIC Agri-Nutrients",
    "2380.SR": "Petro Rabigh", "2350.SR": "Saudi Kayan", "4200.SR": "Aldrees",
    "4003.SR": "Extra", "4004.SR": "Dallah Healthcare", "4013.SR": "Dr. Sulaiman Al Habib",
    "2280.SR": "Almarai", "2050.SR": "Savola", "4190.SR": "Jarir", "7203.SR": "Elm",
    "2082.SR": "ACWA Power", "4164.SR": "Nahdi Medical", "1321.SR": "East Pipes",
}

METALS = {"GC=F": "Gold", "SI=F": "Silver"}
BENCHMARKS = {"US": "SPY", "Saudi": "^TASI.SR"}

# ----------------------------- Indicators --------------------------------
def clip(v: float | None, lo: float = 0.0, hi: float = 10.0) -> float:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 5.0
    return max(lo, min(hi, float(v)))


def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def pct_change_safe(series: pd.Series, periods: int) -> float | None:
    if len(series) <= periods or float(series.iloc[-periods - 1]) == 0:
        return None
    return (float(series.iloc[-1]) / float(series.iloc[-periods - 1]) - 1) * 100


def sparkline_values(close: pd.Series, points: int = 45) -> list[float]:
    vals = close.dropna().tail(points).astype(float).tolist()
    return [round(v, 4) for v in vals]


def get_hist(ticker: str, period: str = "1y") -> pd.DataFrame | None:
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
        if df is None or df.empty or len(df) < 60:
            return None
        return df
    except Exception:
        return None


def get_info(ticker: str) -> dict[str, Any]:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


def technical_snapshot(hist: pd.DataFrame) -> dict[str, Any]:
    close = hist["Close"].astype(float)
    volume = hist["Volume"].astype(float) if "Volume" in hist else pd.Series(index=hist.index, dtype=float)
    rsi = wilder_rsi(close)
    atr14 = atr(hist)
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    high_52 = close.tail(252).max()
    low_52 = close.tail(252).min()
    price = float(close.iloc[-1])
    vol20 = float(volume.tail(20).mean()) if len(volume.dropna()) else 0.0
    vol_ratio = float(volume.iloc[-1] / vol20) if vol20 > 0 else None
    atr_pct = float(atr14.iloc[-1] / price * 100) if pd.notna(atr14.iloc[-1]) and price else None
    return {
        "price": price,
        "rsi": float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None,
        "sma20": float(sma20.iloc[-1]) if pd.notna(sma20.iloc[-1]) else None,
        "sma50": float(sma50.iloc[-1]) if pd.notna(sma50.iloc[-1]) else None,
        "sma200": float(sma200.iloc[-1]) if pd.notna(sma200.iloc[-1]) else None,
        "ret5": pct_change_safe(close, 5), "ret20": pct_change_safe(close, 20),
        "ret60": pct_change_safe(close, 60), "vol_ratio": vol_ratio, "atr_pct": atr_pct,
        "high52": float(high_52), "low52": float(low_52),
        "spark": sparkline_values(close),
    }


def trend_score(t: dict[str, Any]) -> float:
    p, s20, s50, s200 = t["price"], t["sma20"], t["sma50"], t["sma200"]
    score = 5.0
    if s20: score += 1.0 if p > s20 else -1.0
    if s50: score += 1.2 if p > s50 else -1.2
    if s200: score += 1.5 if p > s200 else -1.5
    if s20 and s50: score += 0.8 if s20 > s50 else -0.8
    if s50 and s200: score += 0.5 if s50 > s200 else -0.5
    return clip(score)


def momentum_score(t: dict[str, Any]) -> float:
    rsi = t.get("rsi")
    score = 5.0
    if rsi is not None:
        if 48 <= rsi <= 68: score += 2.0
        elif 40 <= rsi < 48: score += 0.8
        elif 30 <= rsi < 40: score += 0.2
        elif rsi < 30: score -= 0.6
        elif 68 < rsi <= 75: score += 0.4
        else: score -= 1.0
    for ret, weight in [(t.get("ret5"), 0.7), (t.get("ret20"), 1.1)]:
        if ret is not None:
            score += max(-weight, min(weight, ret / 8 * weight))
    return clip(score)


def volume_score(t: dict[str, Any]) -> float:
    vr = t.get("vol_ratio")
    if vr is None: return 5.0
    if vr >= 2.0: return 9.0
    if vr >= 1.5: return 8.0
    if vr >= 1.15: return 6.5
    if vr >= 0.8: return 5.0
    return 4.0


def structure_score(t: dict[str, Any]) -> float:
    p, hi, lo = t["price"], t["high52"], t["low52"]
    if hi <= lo: return 5.0
    pos = (p - lo) / (hi - lo)
    # Favor strong assets not extremely stretched at 52w highs.
    if 0.60 <= pos <= 0.90: return 8.5
    if 0.45 <= pos < 0.60: return 7.0
    if 0.90 < pos <= 0.98: return 7.0
    if 0.25 <= pos < 0.45: return 5.5
    if pos > 0.98: return 5.5
    return 4.0


def risk_score(t: dict[str, Any]) -> float:
    a = t.get("atr_pct")
    if a is None: return 5.0
    if a <= 1.5: return 9.0
    if a <= 2.5: return 7.5
    if a <= 4.0: return 6.0
    if a <= 6.0: return 4.5
    return 3.0


def fundamental_scores(info: dict[str, Any]) -> tuple[float, float, float]:
    # quality, valuation, growth (all 0-10). Missing data stays neutral.
    quality = 5.0
    valuation = 5.0
    growth = 5.0
    roe = info.get("returnOnEquity")
    margin = info.get("profitMargins")
    debt_eq = info.get("debtToEquity")
    pe = info.get("trailingPE")
    fpe = info.get("forwardPE")
    revg = info.get("revenueGrowth")
    earn_g = info.get("earningsGrowth")

    if isinstance(roe, (int, float)): quality += max(-1.5, min(1.5, (roe - 0.12) * 8))
    if isinstance(margin, (int, float)): quality += max(-1.0, min(1.0, (margin - 0.10) * 5))
    if isinstance(debt_eq, (int, float)): quality += 0.7 if debt_eq < 80 else (-0.8 if debt_eq > 200 else 0)

    valid_pe = fpe if isinstance(fpe, (int, float)) and fpe > 0 else pe
    if isinstance(valid_pe, (int, float)) and valid_pe > 0:
        if valid_pe < 12: valuation = 8.5
        elif valid_pe < 20: valuation = 7.0
        elif valid_pe < 30: valuation = 5.5
        elif valid_pe < 45: valuation = 4.0
        else: valuation = 3.0

    vals = [x for x in (revg, earn_g) if isinstance(x, (int, float))]
    if vals:
        avg = sum(vals) / len(vals)
        growth = clip(5 + avg * 12)
    return clip(quality), clip(valuation), clip(growth)


def scores_equity(t: dict[str, Any], info: dict[str, Any]) -> dict[str, float]:
    tr, mo, vo, st, rk = trend_score(t), momentum_score(t), volume_score(t), structure_score(t), risk_score(t)
    quality, valuation, growth = fundamental_scores(info)
    short = .28 * tr + .30 * mo + .18 * vo + .14 * st + .10 * rk
    swing = .34 * tr + .22 * mo + .12 * vo + .18 * st + .08 * valuation + .06 * rk
    invest = .18 * tr + .24 * quality + .22 * valuation + .24 * growth + .12 * rk
    return {"short": round(short, 1), "swing": round(swing, 1), "investment": round(invest, 1),
            "trend": round(tr, 1), "momentum": round(mo, 1), "volume": round(vo, 1),
            "structure": round(st, 1), "risk": round(rk, 1), "quality": round(quality, 1),
            "valuation": round(valuation, 1), "growth": round(growth, 1)}


def scores_metal(t: dict[str, Any]) -> dict[str, float]:
    tr, mo, st, rk = trend_score(t), momentum_score(t), structure_score(t), risk_score(t)
    short = .38 * tr + .34 * mo + .18 * st + .10 * rk
    swing = .46 * tr + .26 * mo + .20 * st + .08 * rk
    invest = .52 * tr + .18 * mo + .20 * st + .10 * rk
    return {"short": round(short, 1), "swing": round(swing, 1), "investment": round(invest, 1),
            "trend": round(tr, 1), "momentum": round(mo, 1), "volume": 5.0,
            "structure": round(st, 1), "risk": round(rk, 1), "quality": 5.0,
            "valuation": 5.0, "growth": 5.0}


def label_for(score: float) -> str:
    if score >= 8.5: return "High Priority"
    if score >= 7.5: return "Attractive"
    if score >= 6.5: return "Watch"
    if score >= 5.5: return "Neutral"
    return "Weak"


def levels(t: dict[str, Any]) -> dict[str, float | None]:
    p = t["price"]
    atr_abs = (t.get("atr_pct") or 0) / 100 * p
    s20 = t.get("sma20")
    support = min([x for x in [s20, p - atr_abs] if x and x > 0], default=None)
    entry_low = max(0, p - 0.45 * atr_abs) if atr_abs else p * 0.99
    entry_high = p + 0.15 * atr_abs if atr_abs else p * 1.005
    stop = (support - 0.55 * atr_abs) if support and atr_abs else p * 0.96
    target1 = p + 1.4 * atr_abs if atr_abs else p * 1.04
    target2 = p + 2.5 * atr_abs if atr_abs else p * 1.08
    return {k: round(v, 2) if v is not None else None for k, v in {
        "entry_low": entry_low, "entry_high": entry_high, "stop": stop,
        "target1": target1, "target2": target2}.items()}


def build_asset(ticker: str, name: str, market: str, asset_type: str) -> dict[str, Any] | None:
    hist = get_hist(ticker)
    if hist is None: return None
    tech = technical_snapshot(hist)
    info = get_info(ticker) if asset_type == "Equity" else {}
    scores = scores_equity(tech, info) if asset_type == "Equity" else scores_metal(tech)
    best_horizon = max(("short", "swing", "investment"), key=lambda k: scores[k])
    best_score = scores[best_horizon]
    currency = info.get("currency") if info else ("USD/oz" if ticker in METALS else "")
    if market == "Saudi": currency = "SAR"
    return {
        "ticker": ticker, "name": name, "market": market, "asset_type": asset_type,
        "currency": currency or "USD", "price": round(tech["price"], 2),
        "rsi": round(tech["rsi"], 1) if tech["rsi"] is not None else None,
        "ret5": round(tech["ret5"], 2) if tech["ret5"] is not None else None,
        "ret20": round(tech["ret20"], 2) if tech["ret20"] is not None else None,
        "ret60": round(tech["ret60"], 2) if tech["ret60"] is not None else None,
        "vol_ratio": round(tech["vol_ratio"], 2) if tech["vol_ratio"] is not None else None,
        "atr_pct": round(tech["atr_pct"], 2) if tech["atr_pct"] is not None else None,
        "sma20": round(tech["sma20"], 2) if tech["sma20"] else None,
        "sma50": round(tech["sma50"], 2) if tech["sma50"] else None,
        "sma200": round(tech["sma200"], 2) if tech["sma200"] else None,
        "pe": round(float(info["trailingPE"]), 2) if isinstance(info.get("trailingPE"), (int,float)) and info["trailingPE"] > 0 else None,
        "forward_pe": round(float(info["forwardPE"]), 2) if isinstance(info.get("forwardPE"), (int,float)) and info["forwardPE"] > 0 else None,
        "sector": info.get("sector") or ("Precious Metals" if asset_type == "Metal" else "—"),
        "scores": scores, "best_horizon": best_horizon, "opportunity_score": best_score,
        "signal": label_for(best_score), "levels": levels(tech), "spark": tech["spark"],
    }


def fetch_news(ticker: str, max_items: int = 4) -> list[dict[str, str]]:
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
    out = []
    for item in raw[:max_items]:
        c = item.get("content", item)
        title = c.get("title") or item.get("title")
        if not title: continue
        out.append({
            "title": title,
            "publisher": (c.get("provider") or {}).get("displayName") or item.get("publisher") or "Source",
            "link": (c.get("canonicalUrl") or {}).get("url") or item.get("link") or "#",
        })
    return out


def claude_commentary(top_assets: list[dict[str, Any]]) -> dict[str, str]:
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key or Anthropic is None or not top_assets:
        return {}
    payload = []
    for a in top_assets:
        payload.append({
            "ticker": a["ticker"], "name": a["name"], "market": a["market"],
            "price": a["price"], "scores": a["scores"], "rsi": a["rsi"],
            "ret20": a["ret20"], "atr_pct": a["atr_pct"], "pe": a["pe"],
            "news": a.get("news", []),
        })
    prompt = f"""You are the explanation layer for a quantitative market research dashboard.
The rule engine has already calculated every score. Do NOT change, override, or invent scores.
For each asset, write one compact paragraph (max 70 words) explaining why it ranks where it does,
what the key catalyst/setup is, and the main risk. Avoid certainty and hype. Return ONLY valid JSON
mapping ticker to paragraph.\n\nDATA:\n{json.dumps(payload, ensure_ascii=False)}"""
    try:
        client = Anthropic(api_key=key)
        r = client.messages.create(model="claude-sonnet-4-6", max_tokens=1400,
                                   messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        start, end = text.find("{"), text.rfind("}")
        return json.loads(text[start:end+1]) if start >= 0 and end > start else {}
    except Exception:
        return {}

# ----------------------------- HTML ---------------------------------------
CSS = r"""
:root{--bg:#080b10;--panel:#0f141b;--panel2:#131a23;--line:#202936;--txt:#edf2f7;--muted:#8b98a8;--good:#47d7a1;--warn:#f0bd5c;--bad:#ef6b73;--accent:#7aa2ff}
*{box-sizing:border-box} body{margin:0;background:radial-gradient(circle at 20% -10%,#152036 0,transparent 32%),var(--bg);color:var(--txt);font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}.wrap{max-width:1400px;margin:auto;padding:24px}.top{display:flex;align-items:center;justify-content:space-between;gap:18px;margin-bottom:20px}.brand h1{font-size:24px;margin:0}.brand p{margin:5px 0 0;color:var(--muted);font-size:13px}.stamp{text-align:right;color:var(--muted);font-size:12px}.hero{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}.kpi{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:14px;padding:15px}.kpi .l{font-size:11px;text-transform:uppercase;color:var(--muted)}.kpi .v{font-size:26px;font-weight:700;margin-top:6px}.toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:20px 0}.tabs{display:flex;gap:6px;flex-wrap:wrap}.tab,.pill,select,input{border:1px solid var(--line);background:var(--panel);color:var(--txt);border-radius:9px;padding:9px 12px}.tab{cursor:pointer}.tab.active{background:#1b2740;border-color:#3f5e98}.search{min-width:220px}.grow{flex:1}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.card{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:14px;padding:15px;cursor:pointer;transition:.18s}.card:hover{transform:translateY(-2px);border-color:#40506a}.row{display:flex;align-items:center;justify-content:space-between;gap:10px}.ticker{font-weight:800;font-size:17px}.name{font-size:12px;color:var(--muted);margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.price{font-size:18px;font-weight:700;margin-top:12px}.meta{font-size:11px;color:var(--muted)}.score{font-size:24px;font-weight:800}.signal{font-size:11px;border-radius:999px;padding:4px 8px;background:#1d2734}.high{color:var(--good)}.mid{color:var(--warn)}.low{color:var(--bad)}.scores{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:12px}.sbox{background:#0b1016;border:1px solid #1c2530;border-radius:8px;padding:7px;text-align:center}.sbox b{display:block;font-size:14px}.sbox span{font-size:9px;color:var(--muted);text-transform:uppercase}.spark{height:42px;margin:9px 0}.spark svg{width:100%;height:100%}.foot{margin:24px 0;color:var(--muted);font-size:11px;line-height:1.5}.empty{padding:60px;text-align:center;color:var(--muted);grid-column:1/-1}.overlay{display:none;position:fixed;inset:0;background:#0009;z-index:20;align-items:flex-start;justify-content:flex-end}.overlay.open{display:flex}.drawer{height:100%;width:min(620px,100%);background:#0a0f15;border-left:1px solid var(--line);padding:22px;overflow:auto}.close{cursor:pointer;background:transparent;border:0;color:var(--txt);font-size:25px}.detail-head{display:flex;justify-content:space-between;gap:20px}.detail-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:9px;margin:18px 0}.metric{background:var(--panel);border:1px solid var(--line);padding:10px;border-radius:10px}.metric .l{font-size:10px;color:var(--muted);text-transform:uppercase}.metric .v{font-size:16px;font-weight:700;margin-top:4px}.levels{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}.ai{background:#101827;border:1px solid #253653;border-radius:12px;padding:14px;line-height:1.55;font-size:13px;margin-top:16px}.news a{color:#9cb8ff;text-decoration:none}.news li{margin-bottom:8px}.badge{font-size:10px;color:var(--muted)}
@media(max-width:1100px){.grid{grid-template-columns:repeat(3,1fr)}}@media(max-width:800px){.grid{grid-template-columns:repeat(2,1fr)}.hero{grid-template-columns:repeat(2,1fr)}}@media(max-width:520px){.wrap{padding:14px}.grid{grid-template-columns:1fr}.hero{grid-template-columns:1fr 1fr}.detail-grid{grid-template-columns:repeat(2,1fr)}.top{align-items:flex-start}.stamp{display:none}}
"""

JS = r"""
const DATA = __DATA__;
let market='All'; let horizon='opportunity';
const grid=document.getElementById('grid');
function cls(s){return s>=7.5?'high':(s>=6.0?'mid':'low')}
function fmt(n,d=2){return (n===null||n===undefined)?'—':Number(n).toFixed(d)}
function spark(vals){if(!vals||vals.length<2)return '';let mn=Math.min(...vals),mx=Math.max(...vals),w=220,h=42;let pts=vals.map((v,i)=>`${i/(vals.length-1)*w},${h-(v-mn)/(mx-mn||1)*h}`).join(' ');return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><polyline fill="none" stroke="currentColor" stroke-width="2" points="${pts}"/></svg>`}
function scoreOf(a){return horizon==='opportunity'?a.opportunity_score:a.scores[horizon]}
function render(){let q=document.getElementById('q').value.toLowerCase();let min=Number(document.getElementById('minScore').value);let arr=DATA.filter(a=>(market==='All'||a.market===market||a.asset_type===market)&&(!q||a.ticker.toLowerCase().includes(q)||a.name.toLowerCase().includes(q))&&scoreOf(a)>=min).sort((a,b)=>scoreOf(b)-scoreOf(a)); grid.innerHTML=arr.length?arr.map(a=>`<div class="card" onclick='openAsset(${JSON.stringify(a.ticker)})'><div class="row"><div><div class="ticker">${a.ticker.replace('.SR','')}</div><div class="name">${a.name}</div></div><div class="score ${cls(scoreOf(a))}">${scoreOf(a).toFixed(1)}</div></div><div class="row"><div class="price">${a.currency} ${fmt(a.price)}</div><span class="signal ${cls(scoreOf(a))}">${a.signal}</span></div><div class="spark ${cls(scoreOf(a))}">${spark(a.spark)}</div><div class="scores"><div class="sbox"><b>${a.scores.short.toFixed(1)}</b><span>Short</span></div><div class="sbox"><b>${a.scores.swing.toFixed(1)}</b><span>Swing</span></div><div class="sbox"><b>${a.scores.investment.toFixed(1)}</b><span>Invest</span></div></div><div class="row" style="margin-top:10px"><span class="meta">RSI ${fmt(a.rsi,1)}</span><span class="meta">20D ${a.ret20===null?'—':(a.ret20>0?'+':'')+fmt(a.ret20)}%</span><span class="meta">ATR ${fmt(a.atr_pct)}%</span></div></div>`).join(''):'<div class="empty">No assets match these filters.</div>';}
function setMarket(m,el){market=m;document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));el.classList.add('active');render()}
function openAsset(t){let a=DATA.find(x=>x.ticker===t);if(!a)return;document.getElementById('dTitle').textContent=`${a.name} · ${a.ticker.replace('.SR','')}`;document.getElementById('dSub').textContent=`${a.market} · ${a.sector}`;document.getElementById('dPrice').textContent=`${a.currency} ${fmt(a.price)}`;document.getElementById('dScore').textContent=`${a.opportunity_score.toFixed(1)} / 10 · ${a.signal}`;let metrics=[['Short',a.scores.short],['Swing',a.scores.swing],['Investment',a.scores.investment],['Trend',a.scores.trend],['Momentum',a.scores.momentum],['Risk quality',a.scores.risk],['RSI',a.rsi],['20-day return',a.ret20===null?'—':`${fmt(a.ret20)}%`],['Volume ratio',a.vol_ratio],['P/E',a.pe],['SMA 50',a.sma50],['SMA 200',a.sma200]];document.getElementById('metrics').innerHTML=metrics.map(x=>`<div class="metric"><div class="l">${x[0]}</div><div class="v">${typeof x[1]==='number'?fmt(x[1],x[0]==='RSI'?1:2):x[1]??'—'}</div></div>`).join('');let l=a.levels;document.getElementById('levels').innerHTML=[['Research entry zone',`${fmt(l.entry_low)} – ${fmt(l.entry_high)}`],['Risk invalidation',fmt(l.stop)],['Objective 1',fmt(l.target1)],['Objective 2',fmt(l.target2)]].map(x=>`<div class="metric"><div class="l">${x[0]}</div><div class="v">${x[1]}</div></div>`).join('');document.getElementById('ai').textContent=a.ai||'Rule-based score only. Add ANTHROPIC_API_KEY to GitHub Secrets to generate an AI explanation for leading opportunities.';document.getElementById('news').innerHTML=(a.news||[]).map(n=>`<li><a target="_blank" rel="noopener" href="${n.link}">${n.title}</a> <span class="badge">${n.publisher}</span></li>`).join('')||'<li>No recent headlines returned.</li>';document.getElementById('overlay').classList.add('open')}
function closeDrawer(){document.getElementById('overlay').classList.remove('open')}
document.getElementById('q').addEventListener('input',render);document.getElementById('minScore').addEventListener('change',render);document.getElementById('horizon').addEventListener('change',e=>{horizon=e.target.value;render()});render();
"""


def render_html(assets: list[dict[str, Any]], generated: str) -> str:
    counts = {"US": sum(a["market"] == "US" for a in assets), "Saudi": sum(a["market"] == "Saudi" for a in assets),
              "Metals": sum(a["asset_type"] == "Metal" for a in assets)}
    top = max(assets, key=lambda a: a["opportunity_score"], default=None)
    data_json = json.dumps(assets, ensure_ascii=False).replace("</", "<\\/")
    js = JS.replace("__DATA__", data_json)
    top_text = f'{html.escape(top["ticker"].replace(".SR", ""))} · {top["opportunity_score"]:.1f}' if top else "—"
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Market Opportunity Dashboard</title><style>{CSS}</style></head><body><div class="wrap"><div class="top"><div class="brand"><h1>Market Opportunity Dashboard</h1><p>Saudi equities · US equities · Gold · Silver</p></div><div class="stamp">Research dashboard<br>Updated {html.escape(generated)} Riyadh time</div></div><div class="hero"><div class="kpi"><div class="l">Assets scanned</div><div class="v">{len(assets)}</div></div><div class="kpi"><div class="l">Saudi stocks</div><div class="v">{counts['Saudi']}</div></div><div class="kpi"><div class="l">US stocks</div><div class="v">{counts['US']}</div></div><div class="kpi"><div class="l">Top current score</div><div class="v">{top_text}</div></div><div class="toolbar" style="grid-column:1/-1"></div></div><div class="toolbar"><div class="tabs"><button class="tab active" onclick="setMarket('All',this)">All</button><button class="tab" onclick="setMarket('Saudi',this)">Saudi</button><button class="tab" onclick="setMarket('US',this)">US</button><button class="tab" onclick="setMarket('Metal',this)">Gold & Silver</button></div><span class="grow"></span><input id="q" class="search" placeholder="Search ticker or company"><select id="horizon"><option value="opportunity">Best opportunity</option><option value="short">Short-term</option><option value="swing">Swing</option><option value="investment">Investment</option></select><select id="minScore"><option value="0">All scores</option><option value="6.5">6.5+</option><option value="7.5">7.5+</option><option value="8.5">8.5+</option></select></div><div id="grid" class="grid"></div><div class="foot">Scores are rule-based research rankings, not guarantees or personalized financial advice. Yahoo Finance/yfinance is used as the prototype market-data provider; Saudi coverage can be incomplete or delayed. Entry, invalidation and objectives are mechanical ATR-based research levels and should be independently verified before any trade.</div></div><div id="overlay" class="overlay" onclick="if(event.target===this)closeDrawer()"><div class="drawer"><div class="detail-head"><div><h2 id="dTitle" style="margin:0"></h2><div id="dSub" class="meta"></div><div id="dPrice" class="price"></div><div id="dScore" style="margin-top:5px;font-weight:700"></div></div><button class="close" onclick="closeDrawer()">×</button></div><div id="metrics" class="detail-grid"></div><h3>Mechanical trade map</h3><div id="levels" class="levels"></div><div id="ai" class="ai"></div><h3>Recent headlines</h3><ul id="news" class="news"></ul></div></div><script>{js}</script></body></html>'''


def main() -> None:
    assets: list[dict[str, Any]] = []
    universe = [(t, n, "US", "Equity") for t, n in US_UNIVERSE.items()]
    universe += [(t, n, "Saudi", "Equity") for t, n in SAUDI_UNIVERSE.items()]
    universe += [(t, n, "Metals", "Metal") for t, n in METALS.items()]

    for ticker, name, market, asset_type in universe:
        try:
            a = build_asset(ticker, name, market, asset_type)
            if a: assets.append(a)
        except Exception as exc:
            print(f"Skip {ticker}: {exc}")

    assets.sort(key=lambda a: a["opportunity_score"], reverse=True)
    top = assets[:10]
    for a in top:
        a["news"] = fetch_news(a["ticker"])
    ai = claude_commentary(top)
    for a in assets:
        a["ai"] = ai.get(a["ticker"], "")
        a.setdefault("news", [])

    riyadh = dt.datetime.now(dt.timezone(dt.timedelta(hours=3))).strftime("%Y-%m-%d %H:%M")
    out = Path("docs")
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(render_html(assets, riyadh), encoding="utf-8")
    (out / "data.json").write_text(json.dumps({"generated": riyadh, "assets": assets}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated {len(assets)} assets -> docs/index.html")

if __name__ == "__main__":
    main()
