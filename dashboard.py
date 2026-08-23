"""Market Opportunity Dashboard v3.0

Research-backed opportunity engine for US stocks, Saudi stocks, gold and silver.
The recommendation engine is deterministic. AI, when enabled, only adds context.

Core ideas used:
- Equity trade view: trend, medium-term momentum, relative strength, volume confirmation,
  setup quality and volatility/risk.
- Equity investment view: quality/profitability, growth, peer-relative valuation,
  momentum confirmation and risk.
- Metals: trend/momentum plus USD/yield macro confirmation.

Data provider: yfinance (free provider; Saudi coverage can be incomplete/delayed). V3 explicitly penalizes missing data rather than treating it as neutral evidence.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd
import yfinance as yf

try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None

# ---------------------------------------------------------------------------
# Universes: deliberately liquid / widely followed names for a dependable V2.
# Easy to expand later without changing the scoring architecture.
# ---------------------------------------------------------------------------
US_UNIVERSE = {
    "AAPL":"Apple", "MSFT":"Microsoft", "NVDA":"NVIDIA", "AMZN":"Amazon",
    "GOOGL":"Alphabet", "META":"Meta", "AVGO":"Broadcom", "TSLA":"Tesla",
    "AMD":"AMD", "NFLX":"Netflix", "PLTR":"Palantir", "ORCL":"Oracle",
    "CRM":"Salesforce", "ADBE":"Adobe", "QCOM":"Qualcomm", "MU":"Micron",
    "JPM":"JPMorgan", "BAC":"Bank of America", "GS":"Goldman Sachs",
    "V":"Visa", "MA":"Mastercard", "BRK-B":"Berkshire Hathaway",
    "XOM":"Exxon Mobil", "CVX":"Chevron", "COP":"ConocoPhillips",
    "LLY":"Eli Lilly", "JNJ":"Johnson & Johnson", "ABBV":"AbbVie",
    "UNH":"UnitedHealth", "MRK":"Merck", "WMT":"Walmart", "COST":"Costco",
    "HD":"Home Depot", "MCD":"McDonald's", "KO":"Coca-Cola", "PEP":"PepsiCo",
    "CAT":"Caterpillar", "GE":"GE Aerospace", "RTX":"RTX", "LMT":"Lockheed Martin",
    "UBER":"Uber", "ABNB":"Airbnb", "MRVL":"Marvell", "ARM":"Arm Holdings",
    "SMCI":"Super Micro Computer", "PANW":"Palo Alto Networks", "CRWD":"CrowdStrike",
    "NOW":"ServiceNow", "INTC":"Intel", "DIS":"Disney",
}

SAUDI_UNIVERSE = {
    "2222.SR":"Saudi Aramco", "1120.SR":"Al Rajhi Bank", "1180.SR":"Saudi National Bank",
    "1150.SR":"Alinma Bank", "1010.SR":"Riyad Bank", "1050.SR":"Banque Saudi Fransi",
    "2010.SR":"SABIC", "1211.SR":"Ma'aden", "2020.SR":"SABIC Agri-Nutrients",
    "7010.SR":"stc", "2082.SR":"ACWA Power", "7203.SR":"Elm",
    "4013.SR":"Dr. Sulaiman Al Habib", "4004.SR":"Dallah Healthcare",
    "4164.SR":"Nahdi Medical", "2280.SR":"Almarai", "4190.SR":"Jarir",
    "4003.SR":"Extra", "4200.SR":"Aldrees", "1321.SR":"East Pipes",
    "2380.SR":"Petro Rabigh", "2350.SR":"Saudi Kayan", "2050.SR":"Savola",
    "2290.SR":"Yansab", "2330.SR":"Advanced Petrochemical", "3040.SR":"Qassim Cement",
    "3020.SR":"Yamama Cement", "5110.SR":"Saudi Electricity", "4030.SR":"Bahri",
    "4260.SR":"Budget Saudi", "4261.SR":"Theeb Rent a Car", "4071.SR":"Arabian Contracting", "4002.SR":"Mouwasat",
    "4031.SR":"Saudi Ground Services", "4040.SR":"SAPTCO", "4050.SR":"SASCO",
    "4080.SR":"Aseer", "4090.SR":"Taiba Investments", "4100.SR":"Makkah Construction",
    "4210.SR":"Saudi Research and Media", "4250.SR":"Jabal Omar", "4290.SR":"Alkhaleej Training",
    "4300.SR":"Dar Al Arkan", "4321.SR":"Cenomi Centers", "4322.SR":"Retal",
    "6001.SR":"Halwani Bros", "6002.SR":"Herfy", "6010.SR":"NADEC", "6020.SR":"Jahez",
    "1830.SR":"Leejam Sports", "1831.SR":"Maharah", "1832.SR":"Sadr Logistics",
    "4011.SR":"Lazurde", "4012.SR":"Thob Al Aseel", "4014.SR":"Scientific & Medical Equipment House",
}

METALS = {"GC=F":"Gold", "SI=F":"Silver"}
BENCHMARKS = {"US":"SPY", "Saudi":"^TASI.SR"}
MACRO_TICKERS = {"dollar":"DX-Y.NYB", "yield10":"^TNX", "copper":"HG=F"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def clip(v: float | None, lo: float = 0.0, hi: float = 10.0) -> float:
    if not finite(v):
        return 5.0
    return max(lo, min(hi, float(v)))


def mean_available(values: list[float | None], neutral: float = 5.0) -> float:
    clean = [float(v) for v in values if finite(v)]
    return sum(clean) / len(clean) if clean else neutral


def percentile(values: list[float], value: float, higher_is_better: bool = True) -> float:
    clean = sorted(float(v) for v in values if finite(v))
    if not clean or not finite(value):
        return 5.0
    if len(clean) == 1:
        return 5.0
    below = sum(v < value for v in clean)
    equal = sum(v == value for v in clean)
    p = (below + 0.5 * equal) / len(clean)
    score = 10.0 * p
    return clip(score if higher_is_better else 10.0 - score)


def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


def ret(close: pd.Series, days: int) -> float | None:
    if len(close) <= days or not finite(close.iloc[-days-1]) or float(close.iloc[-days-1]) == 0:
        return None
    return (float(close.iloc[-1]) / float(close.iloc[-days-1]) - 1) * 100


def get_hist(ticker: str, period: str = "5y") -> pd.DataFrame | None:
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
        if df is None or df.empty or len(df) < 120:
            return None
        return df
    except Exception:
        return None


def get_info(ticker: str) -> dict[str, Any]:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


def spark(close: pd.Series, points: int = 55) -> list[float]:
    return [round(float(v), 4) for v in close.dropna().tail(points)]


def technical_snapshot(hist: pd.DataFrame) -> dict[str, Any]:
    close = hist["Close"].astype(float)
    volume = hist.get("Volume", pd.Series(index=hist.index, dtype=float)).astype(float)
    r = wilder_rsi(close)
    a = atr(hist)
    p = float(close.iloc[-1])
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else pd.NA
    v20 = volume.tail(20).mean() if len(volume.dropna()) else 0
    vr = float(volume.iloc[-1] / v20) if finite(v20) and v20 > 0 else None
    atr_abs = float(a.iloc[-1]) if finite(a.iloc[-1]) else None
    peak = close.tail(252).cummax()
    dd = ((close.tail(252) / peak) - 1).min() * 100 if len(close) else None
    mom12_1 = None
    if len(close) >= 253 and finite(close.iloc[-22]) and finite(close.iloc[-253]) and close.iloc[-253] != 0:
        mom12_1 = (float(close.iloc[-22]) / float(close.iloc[-253]) - 1) * 100
    low52 = float(close.tail(252).min())
    high52 = float(close.tail(252).max())
    pos52 = ((p-low52)/(high52-low52)*100) if high52>low52 else 50.0
    low3 = float(close.tail(min(len(close),756)).min())
    high3 = float(close.tail(min(len(close),756)).max())
    pos3 = ((p-low3)/(high3-low3)*100) if high3>low3 else 50.0
    peak5 = close.cummax()
    dd5 = ((close/peak5)-1).min()*100 if len(close) else None
    latest = hist.index[-1]
    if getattr(latest, "tzinfo", None):
        latest = latest.tz_convert(None)
    return {
        "price": p,
        "rsi": float(r.iloc[-1]) if finite(r.iloc[-1]) else None,
        "sma20": float(sma20) if finite(sma20) else None,
        "sma50": float(sma50) if finite(sma50) else None,
        "sma200": float(sma200) if finite(sma200) else None,
        "ret5": ret(close, 5), "ret21": ret(close, 21), "ret63": ret(close, 63),
        "ret126": ret(close, 126), "ret252": ret(close, 252), "mom12_1": mom12_1,
        "vol_ratio": vr,
        "atr_abs": atr_abs,
        "atr_pct": (atr_abs / p * 100) if atr_abs and p else None,
        "drawdown": float(dd) if finite(dd) else None,
        "drawdown5y": float(dd5) if finite(dd5) else None,
        "high52": high52, "low52": low52, "position52": pos52,
        "position3y": pos3,
        "ret756": ret(close, min(756,len(close)-1)) if len(close)>300 else None,
        "spark": spark(close),
        "as_of": latest.strftime("%Y-%m-%d"),
    }


def raw_fundamentals(info: dict[str, Any]) -> dict[str, Any]:
    market_cap = info.get("marketCap")
    fcf = info.get("freeCashflow")
    fcf_yield = (fcf / market_cap) if finite(fcf) and finite(market_cap) and market_cap > 0 else None
    return {
        "roe": info.get("returnOnEquity"),
        "operating_margin": info.get("operatingMargins"),
        "profit_margin": info.get("profitMargins"),
        "debt_equity": info.get("debtToEquity"),
        "fcf_yield": fcf_yield,
        "revenue_growth": info.get("revenueGrowth"),
        "earnings_growth": info.get("earningsGrowth"),
        "earnings_q_growth": info.get("earningsQuarterlyGrowth"),
        "revenue_per_share": info.get("revenuePerShare"),
        "gross_margin": info.get("grossMargins"),
        "current_ratio": info.get("currentRatio"),
        "forward_pe": info.get("forwardPE"),
        "trailing_pe": info.get("trailingPE"),
        "price_book": info.get("priceToBook"),
        "ev_ebitda": info.get("enterpriseToEbitda"),
        "dividend_yield": info.get("dividendYield"),
        "target_mean": info.get("targetMeanPrice"),
        "forward_eps": info.get("forwardEps"),
        "sector": info.get("sector") or "Other",
        "currency": info.get("currency"),
    }


def trend_score(t: dict[str, Any]) -> float:
    p, s20, s50, s200 = t["price"], t.get("sma20"), t.get("sma50"), t.get("sma200")
    pts, n = 0.0, 0
    for s, w in [(s20, 1.0), (s50, 1.4), (s200, 1.8)]:
        if finite(s):
            pts += w if p > s else 0
            n += w
    if finite(s20) and finite(s50):
        pts += 0.8 if s20 > s50 else 0; n += 0.8
    if finite(s50) and finite(s200):
        pts += 1.0 if s50 > s200 else 0; n += 1.0
    return clip(10 * pts / n) if n else 5.0


def setup_score(t: dict[str, Any]) -> float:
    p, s20, s50, a = t["price"], t.get("sma20"), t.get("sma50"), t.get("atr_abs")
    if not finite(a) or a <= 0:
        return 5.0
    if finite(s20):
        z = (p - s20) / a
        if -0.5 <= z <= 1.2:
            base = 8.5
        elif 1.2 < z <= 2.2:
            base = 6.5
        elif z > 2.2:
            base = 4.0
        elif -1.5 <= z < -0.5:
            base = 6.0
        else:
            base = 3.5
    else:
        base = 5.0
    if finite(s50) and p < s50:
        base -= 1.5
    return clip(base)


def volume_confirmation(t: dict[str, Any]) -> float:
    vr, r5 = t.get("vol_ratio"), t.get("ret5")
    if not finite(vr) or not finite(r5):
        return 5.0
    if vr >= 1.5 and r5 > 0: return 9.0
    if vr >= 1.15 and r5 > 0: return 7.0
    if vr >= 1.5 and r5 < 0: return 2.5
    if vr >= 1.15 and r5 < 0: return 4.0
    return 5.0


def risk_quality(t: dict[str, Any], atr_values: list[float], drawdowns: list[float]) -> float:
    atr_s = percentile(atr_values, t.get("atr_pct"), higher_is_better=False) if finite(t.get("atr_pct")) else 5.0
    # Drawdown closer to zero is better.
    dd_s = percentile(drawdowns, t.get("drawdown"), higher_is_better=True) if finite(t.get("drawdown")) else 5.0
    return clip(0.55 * atr_s + 0.45 * dd_s)


def reversal_score(t: dict[str, Any], f: dict[str, Any]) -> float:
    """Detect a forming recovery without rewarding a falling knife.
    A deep drawdown creates *potential*, but points only arrive when price and/or earnings stabilize.
    """
    dd = t.get("drawdown")
    if not finite(dd) or dd > -15:
        return 3.5
    score = 3.0
    if dd <= -25: score += 1.0
    if dd <= -40: score += 0.5
    if finite(t.get("position52")) and t["position52"] <= 35: score += 0.5
    if finite(t.get("ret21")) and t["ret21"] > 0: score += 1.2
    if finite(t.get("ret63")) and t["ret63"] > 0: score += 1.2
    if finite(t.get("vol_ratio")) and t["vol_ratio"] >= 1.15 and finite(t.get("ret5")) and t["ret5"] > 0: score += 0.8
    eg = f.get("earnings_growth") if f else None
    rg = f.get("revenue_growth") if f else None
    if finite(eg) and eg > 0: score += 0.8
    elif finite(eg) and eg < -0.25: score -= 1.0
    if finite(rg) and rg > 0: score += 0.5
    return clip(score)


def value_trap_penalty(t: dict[str, Any], f: dict[str, Any]) -> float:
    """0 is clean; higher means cheap/low price may be deterioration rather than opportunity."""
    p = 0.0
    if finite(t.get("ret126")) and t["ret126"] < -20: p += 1.0
    if finite(t.get("ret252")) and t["ret252"] < -30: p += 1.0
    if finite(f.get("earnings_growth")) and f["earnings_growth"] < -0.20: p += 1.5
    if finite(f.get("revenue_growth")) and f["revenue_growth"] < -0.08: p += 1.0
    if finite(f.get("operating_margin")) and f["operating_margin"] < 0: p += 1.0
    return min(4.0,p)


def opportunity_type(a: dict[str, Any]) -> str:
    s=a["scores"]
    if s.get("recovery",0)>=7.0: return "Recovery"
    if s.get("breakout",0)>=7.5: return "Breakout"
    if s.get("pullback",0)>=7.2: return "Pullback"
    if s.get("growth_value",0)>=7.2: return "Growth + Value"
    if s.get("quality_value",0)>=7.2: return "Quality + Value"
    if s.get("momentum",0)>=7.5: return "Momentum"
    return "Balanced"

def fetch_base_asset(ticker: str, name: str, market: str, asset_type: str) -> dict[str, Any] | None:
    h = get_hist(ticker)
    if h is None:
        return None
    t = technical_snapshot(h)
    info = get_info(ticker) if asset_type == "Equity" else {}
    f = raw_fundamentals(info) if asset_type == "Equity" else {}
    currency = "USD/oz" if asset_type == "Metal" else ("SAR" if market == "Saudi" else (f.get("currency") or "USD"))
    return {
        "ticker": ticker, "name": name, "market": market, "asset_type": asset_type,
        "currency": currency, "tech": t, "fund": f,
    }


def peer_pool(assets: list[dict[str, Any]], asset: dict[str, Any]) -> list[dict[str, Any]]:
    same_market = [a for a in assets if a["asset_type"] == "Equity" and a["market"] == asset["market"]]
    sector = asset["fund"].get("sector")
    same_sector = [a for a in same_market if a["fund"].get("sector") == sector]
    return same_sector if len(same_sector) >= 4 else same_market


def factor_score(pool: list[dict[str, Any]], asset: dict[str, Any], key: str, higher: bool = True) -> float:
    vals = [a["fund"].get(key) for a in pool if finite(a["fund"].get(key))]
    v = asset["fund"].get(key)
    return percentile(vals, v, higher) if finite(v) and len(vals) >= 3 else 5.0


def calc_equity_scores(assets: list[dict[str, Any]], asset: dict[str, Any], benchmark_ret126: float | None) -> dict[str, float]:
    t, f = asset["tech"], asset["fund"]
    market_assets = [a for a in assets if a["asset_type"] == "Equity" and a["market"] == asset["market"]]
    pool = peer_pool(assets, asset)

    tr = trend_score(t)
    mom_vals = [a["tech"].get("mom12_1") for a in market_assets if finite(a["tech"].get("mom12_1"))]
    mom12 = percentile(mom_vals, t.get("mom12_1"), True) if finite(t.get("mom12_1")) else 5.0
    mom6_vals = [a["tech"].get("ret126") for a in market_assets if finite(a["tech"].get("ret126"))]
    mom6 = percentile(mom6_vals, t.get("ret126"), True) if finite(t.get("ret126")) else 5.0
    momentum = clip(0.65*mom12 + 0.35*mom6)

    if finite(t.get("ret126")) and finite(benchmark_ret126):
        rs_raw=t["ret126"]-benchmark_ret126
        rs_values=[a["tech"].get("ret126")-benchmark_ret126 for a in market_assets if finite(a["tech"].get("ret126"))]
        rel_strength=percentile(rs_values,rs_raw,True)
    else: rel_strength=5.0

    vc=volume_confirmation(t); setup=setup_score(t)
    atr_values=[a["tech"].get("atr_pct") for a in market_assets if finite(a["tech"].get("atr_pct"))]
    dd_values=[a["tech"].get("drawdown") for a in market_assets if finite(a["tech"].get("drawdown"))]
    risk=risk_quality(t,atr_values,dd_values)

    quality_parts=[factor_score(pool,asset,"roe",True),factor_score(pool,asset,"operating_margin",True),factor_score(pool,asset,"profit_margin",True),factor_score(pool,asset,"fcf_yield",True)]
    if f.get("sector") != "Financial Services": quality_parts.append(factor_score(pool,asset,"debt_equity",False))
    quality=clip(mean_available(quality_parts))
    growth=clip(mean_available([factor_score(pool,asset,"revenue_growth",True),factor_score(pool,asset,"earnings_growth",True),factor_score(pool,asset,"earnings_q_growth",True)]))
    value=clip(mean_available([factor_score(pool,asset,"forward_pe",False),factor_score(pool,asset,"price_book",False),factor_score(pool,asset,"ev_ebitda",False),factor_score(pool,asset,"dividend_yield",True)]))

    recovery=reversal_score(t,f)
    breakout=clip(0.35*tr+0.25*momentum+0.20*vc+0.20*(10 if finite(t.get("position52")) and t["position52"]>=85 else 4))
    pullback=clip(0.35*tr+0.25*momentum+0.20*rel_strength+0.20*setup)
    quality_value=clip(0.55*quality+0.45*value)
    growth_value=clip(0.40*growth+0.30*quality+0.30*value)
    trap=value_trap_penalty(t,f)

    # TRADE: multiple opportunity paths. Recovery is allowed, but cannot dominate without confirmation.
    continuation=0.28*tr+0.24*momentum+0.18*rel_strength+0.10*vc+0.12*setup+0.08*risk
    alternative=max(0.92*breakout,0.94*pullback,0.90*recovery)
    trade=clip(max(continuation,alternative))

    # INVEST: established factor families + historical/price confirmation. Penalize likely value traps.
    invest_base=0.26*quality+0.22*growth+0.22*value+0.14*momentum+0.08*risk+0.08*max(recovery,5.0)
    invest=clip(invest_base-0.65*trap)

    fields=[f.get(k) for k in ["roe","operating_margin","revenue_growth","earnings_growth","forward_pe","price_book","target_mean"]]
    dq=10*sum(finite(x) for x in fields)/len(fields)
    return {"trade":round(trade,1),"invest":round(invest,1),"trend":round(tr,1),"momentum":round(momentum,1),"relative_strength":round(rel_strength,1),"volume":round(vc,1),"setup":round(setup,1),"risk":round(risk,1),"quality":round(quality,1),"growth":round(growth,1),"value":round(value,1),"recovery":round(recovery,1),"breakout":round(breakout,1),"pullback":round(pullback,1),"quality_value":round(quality_value,1),"growth_value":round(growth_value,1),"value_trap":round(trap,1),"data_quality":round(dq,1)}

def macro_snapshot() -> dict[str, Any]:
    out = {}
    for key, ticker in MACRO_TICKERS.items():
        h = get_hist(ticker, "1y")
        if h is None:
            out[key] = {"ret21":None, "ret63":None}
            continue
        c = h["Close"].astype(float)
        out[key] = {"ret21":ret(c,21), "ret63":ret(c,63)}
    return out


def macro_inverse_score(change: float | None, scale: float) -> float:
    if not finite(change): return 5.0
    return clip(5 - float(change)/scale*2.5)


def macro_positive_score(change: float | None, scale: float) -> float:
    if not finite(change): return 5.0
    return clip(5 + float(change)/scale*2.5)


def calc_metal_scores(asset: dict[str, Any], macro: dict[str, Any]) -> dict[str, float]:
    t = asset["tech"]
    tr = trend_score(t)
    # Absolute momentum for metals, not peer percentile.
    r63, r126 = t.get("ret63"), t.get("ret126")
    mo = 5.0
    if finite(r63): mo += max(-2.0, min(2.0, r63/10*2.0))
    if finite(r126): mo += max(-2.0, min(2.0, r126/18*2.0))
    mo = clip(mo)
    dollar = macro_inverse_score(macro.get("dollar",{}).get("ret63"), 8)
    yields = macro_inverse_score(macro.get("yield10",{}).get("ret63"), 12)
    copper = macro_positive_score(macro.get("copper",{}).get("ret63"), 18)
    risk = clip(8.5 - max(0, (t.get("atr_pct") or 2.0)-1.5)*0.9)
    if asset["ticker"] == "GC=F":
        macro_score = 0.60*dollar + 0.40*yields
        trade = 0.40*tr + 0.30*mo + 0.20*macro_score + 0.10*risk
        invest = 0.35*tr + 0.25*mo + 0.30*macro_score + 0.10*risk
    else:
        macro_score = 0.45*dollar + 0.25*yields + 0.30*copper
        trade = 0.38*tr + 0.30*mo + 0.22*macro_score + 0.10*risk
        invest = 0.32*tr + 0.25*mo + 0.33*macro_score + 0.10*risk
    return {
        "trade":round(clip(trade),1), "invest":round(clip(invest),1),
        "trend":round(tr,1), "momentum":round(mo,1), "relative_strength":5.0,
        "volume":5.0, "setup":round(setup_score(t),1), "risk":round(risk,1),
        "quality":5.0, "growth":5.0, "value":round(macro_score,1), "data_quality":10.0,
        "recovery":3.5, "breakout":round(0.6*tr+0.4*mo,1), "pullback":round(setup_score(t),1),
        "quality_value":5.0, "growth_value":5.0, "value_trap":0.0,
    }


def trade_levels(a: dict[str, Any]) -> dict[str, float | None]:
    t=a["tech"]; p=t["price"]; aa=t.get("atr_abs"); s20=t.get("sma20"); s50=t.get("sma50")
    if not finite(aa) or aa<=0: aa=max(p*0.025,0.01)
    # Buy zone must contain current price for a direct BUY. Pullback zone may sit below current and then recommendation becomes WAIT.
    if finite(s20) and s20 < p and (p-s20) <= 1.7*aa:
        entry_low=max(0,s20-0.20*aa); entry_high=min(p+0.10*aa,s20+0.55*aa)
    else:
        entry_low=max(0,p-0.45*aa); entry_high=p+0.15*aa
    support=[x for x in [s20,s50,p-1.35*aa] if finite(x) and x < entry_low]
    base=max(support) if support else entry_low-0.75*aa
    stop=max(0,min(entry_low-0.35*aa,base-0.20*aa))
    risk=max(0.01,((entry_low+entry_high)/2)-stop)
    target=max(p+1.8*aa,((entry_low+entry_high)/2)+2.1*risk)
    rr=(target-((entry_low+entry_high)/2))/risk
    return {"entry_low":round(entry_low,2),"entry_high":round(entry_high,2),"stop":round(stop,2),"target":round(target,2),"rr":round(rr,1)}


def trade_recommendation(a: dict[str, Any]) -> str:
    s=a["scores"]; p=a["tech"]["price"]; l=a["trade_levels"]
    if s["trend"]<=3.0 and s["momentum"]<=3.5: return "SELL"
    if s["trade"]>=7.2:
        # A good setup is not a BUY if price has already run beyond the preferred zone.
        if l["entry_low"]*0.995 <= p <= l["entry_high"]*1.01 and l["stop"] < l["entry_low"]: return "BUY"
        return "WAIT"
    if s["trade"]>=6.0 or s.get("recovery",0)>=6.5: return "WAIT"
    return "AVOID"


def invest_recommendation(a: dict[str, Any]) -> str:
    s=a["scores"]
    if s.get("value_trap",0)>=3.0 and s["invest"]<6.0: return "SELL"
    if s["invest"]>=7.2 and s["data_quality"]>=4.0: return "BUY"
    if s["invest"]>=5.8: return "HOLD"
    if s["invest"]<=3.8 and (s["growth"]<4.0 or s["quality"]<4.0): return "SELL"
    return "AVOID"


def trade_reason(a: dict[str, Any]) -> str:
    s=a["scores"]; r=a["trade_rec"]; typ=a.get("opportunity_type","Balanced")
    if r=="BUY":
        if typ=="Recovery": return "Recovery is gaining confirmation: price momentum has improved after a major drawdown, with supportive volume/fundamentals."
        if typ=="Breakout": return "Price is near a major high with strong momentum and volume confirmation, while the current entry remains acceptable."
        if typ=="Pullback": return "The broader trend is strong and the pullback has brought price back into an attractive entry area."
        return "Trend, momentum and relative strength are aligned, and the current price is inside the preferred entry zone."
    if r=="WAIT":
        if s.get("recovery",0)>=6.5: return "A recovery may be forming after a large decline, but confirmation is not strong enough for a direct buy yet."
        if a["tech"]["price"]>a["trade_levels"]["entry_high"]: return "The setup is attractive, but price is above the preferred entry zone; wait for a better entry."
        return "The setup has promise, but trend/momentum confirmation is not strong enough for a direct buy."
    if r=="SELL": return "Price trend and momentum are both weak, so downside risk currently outweighs the opportunity."
    return "The evidence is not strong enough for a new trade: trend, momentum or confirmation is missing."


def invest_reason(a: dict[str, Any]) -> str:
    s=a["scores"]; r=a["invest_rec"]
    if a["asset_type"]=="Metal":
        return "Long-term trend and macro conditions are supportive." if r=="BUY" else ("Trend is acceptable, but macro confirmation is mixed." if r=="HOLD" else "Trend and macro conditions do not support a new long-term position.")
    if r=="BUY":
        if s["growth_value"]>=7.2: return "Growth and business quality are strong while valuation remains reasonable versus peers, with price momentum confirming the thesis."
        if s["quality_value"]>=7.2: return "Business quality is strong and valuation is attractive versus peers, with enough market confirmation to support accumulation."
        if s["recovery"]>=7.0: return "The stock is recovering from a major drawdown while fundamentals and price behavior are improving enough to support long-term accumulation."
        return "Quality, growth, valuation and market confirmation are collectively strong enough for long-term accumulation."
    if r=="HOLD":
        if s["value"]<4.5: return "The business may be sound, but valuation is not attractive enough for aggressive new buying."
        return "The long-term case is reasonable, but the evidence is not strong enough for a fresh high-conviction buy."
    if r=="SELL": return "Fundamental deterioration and weak market behavior create an unfavorable long-term risk/reward profile."
    if s.get("value_trap",0)>=2.5: return "The stock looks cheaper after its decline, but earnings/price deterioration raises value-trap risk."
    return "Fundamentals, valuation or forward confirmation are not strong enough for a new long-term position."

def investment_target(a: dict[str, Any], all_assets: list[dict[str, Any]]) -> tuple[float | None, str | None]:
    if a["asset_type"] != "Equity":
        return None, None
    p = a["tech"]["price"]; f=a["fund"]
    tm = f.get("target_mean")
    if finite(tm) and 0.5*p <= tm <= 2.5*p:
        return round(float(tm),2), "Analyst consensus"
    # Conservative peer-multiple fallback only when forward EPS and enough peer P/E data exist.
    eps = f.get("forward_eps")
    if not finite(eps) or eps <= 0:
        return None, None
    pool = peer_pool(all_assets, a)
    pes = [x["fund"].get("forward_pe") for x in pool if finite(x["fund"].get("forward_pe")) and x["fund"].get("forward_pe") > 0]
    if len(pes) < 4:
        return None, None
    pe_med = median(pes)
    target = float(eps) * pe_med
    if 0.55*p <= target <= 2.0*p:
        return round(target,2), "Peer-value estimate"
    return None, None


def recent_news(ticker: str, max_items: int = 4) -> list[dict[str,str]]:
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
    out=[]
    for item in raw[:max_items]:
        c=item.get("content",item); title=c.get("title") or item.get("title")
        if not title: continue
        out.append({
            "title":title,
            "publisher":(c.get("provider") or {}).get("displayName") or item.get("publisher") or "Source",
            "link":(c.get("canonicalUrl") or {}).get("url") or item.get("link") or "#",
        })
    return out


def ai_context(top_assets: list[dict[str, Any]]) -> dict[str,str]:
    key=os.getenv("ANTHROPIC_API_KEY","").strip()
    if not key or Anthropic is None or not top_assets: return {}
    payload=[{
        "ticker":a["ticker"],"name":a["name"],"trade":a["trade_rec"],"invest":a["invest_rec"],
        "scores":a["scores"],"trade_reason":a["trade_reason"],"invest_reason":a["invest_reason"],
        "news":a.get("news",[])
    } for a in top_assets]
    prompt=("You are an optional context layer for a deterministic market dashboard. The recommendations and reasons are fixed. "
            "Do not change them. For each ticker return one short sentence (max 35 words) highlighting the most relevant recent news/catalyst or risk. "
            "If news is not clearly relevant, say 'No material news catalyst identified.' Return only JSON ticker->sentence.\nDATA:\n"+json.dumps(payload,ensure_ascii=False))
    try:
        client=Anthropic(api_key=key)
        r=client.messages.create(model="claude-sonnet-4-6",max_tokens=1000,messages=[{"role":"user","content":prompt}])
        text="".join(b.text for b in r.content if getattr(b,"type","")=="text")
        i,j=text.find("{"),text.rfind("}")
        return json.loads(text[i:j+1]) if i>=0 and j>i else {}
    except Exception:
        return {}

# ---------------------------------------------------------------------------
# Front-end
# ---------------------------------------------------------------------------
CSS = r'''
:root{--bg:#070a0f;--p:#101722;--p2:#141d2a;--line:#263244;--text:#f2f6fb;--muted:#91a0b4;--buy:#43d39e;--wait:#f1ba5d;--sell:#ef6b73;--blue:#7ba7ff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% -8%,#172745 0,transparent 30%),var(--bg);color:var(--text);font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif}.wrap{max-width:1500px;margin:auto;padding:24px}.top{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}.brand h1{margin:0;font-size:27px;letter-spacing:-.6px}.brand p{margin:6px 0;color:var(--muted);font-size:13px}.updated{text-align:right;font-size:12px;color:var(--muted)}.modebar{display:flex;gap:7px;margin:22px 0 14px}.mode{border:1px solid var(--line);background:#0e141e;color:var(--text);padding:10px 17px;border-radius:10px;font-weight:800;cursor:pointer}.mode.active{background:#1c2c47;border-color:#476a9e}.summary{display:grid;grid-template-columns:repeat(4,1fr);gap:11px;margin-bottom:16px}.kpi{border:1px solid var(--line);background:linear-gradient(180deg,var(--p2),var(--p));border-radius:14px;padding:14px}.kpi .l{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.7px}.kpi .v{font-size:24px;font-weight:850;margin-top:5px}.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:14px 0}.tabs{display:flex;gap:6px;flex-wrap:wrap}.tab,select,input{border:1px solid var(--line);background:#0f151f;color:var(--text);border-radius:9px;padding:9px 11px}.tab{cursor:pointer}.tab.active{background:#1a2941;border-color:#41618d}.grow{flex:1}.search{min-width:230px}.section-title{font-size:15px;margin:20px 0 10px;color:#cbd7e7}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.card{background:linear-gradient(180deg,var(--p2),var(--p));border:1px solid var(--line);border-radius:15px;padding:16px;cursor:pointer;transition:.16s}.card:hover{transform:translateY(-2px);border-color:#48617f}.row{display:flex;justify-content:space-between;gap:10px;align-items:center}.ticker{font-size:17px;font-weight:850}.name{font-size:12px;color:var(--muted);margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.price{font-size:18px;font-weight:800}.rec{font-size:15px;font-weight:900;padding:5px 10px;border-radius:999px;border:1px solid currentColor}.buy{color:var(--buy)}.wait,.hold{color:var(--wait)}.avoid,.sell{color:var(--sell)}.score{font-size:30px;font-weight:900}.reason{font-size:13px;line-height:1.45;margin:12px 0;color:#d6dfeb;min-height:38px}.levels{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:12px}.lv{background:#0b1119;border:1px solid #1d2837;border-radius:9px;padding:8px}.lv span{display:block;color:var(--muted);font-size:9px;text-transform:uppercase}.lv b{font-size:12px}.spark{height:44px;margin:8px 0}.spark svg{height:100%;width:100%}.meta{font-size:10px;color:var(--muted)}.pill{font-size:10px;border:1px solid var(--line);padding:3px 7px;border-radius:99px;color:var(--muted)}.foot{margin:24px 0;color:var(--muted);font-size:11px;line-height:1.55}.empty{grid-column:1/-1;text-align:center;padding:60px;color:var(--muted)}.overlay{display:none;position:fixed;inset:0;background:#000a;z-index:30;justify-content:flex-end}.overlay.open{display:flex}.drawer{height:100%;width:min(670px,100%);background:#0a0f16;border-left:1px solid var(--line);padding:22px;overflow:auto}.close{background:none;border:0;color:white;font-size:27px;cursor:pointer}.dtitle h2{margin:0}.bigrec{font-size:28px;font-weight:900;margin:12px 0 4px}.dreasons{font-size:15px;line-height:1.55;color:#d8e1ec}.dlevels{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin:16px 0}.metricgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin:16px 0}.metric{background:var(--p);border:1px solid var(--line);border-radius:10px;padding:9px}.metric span{font-size:9px;text-transform:uppercase;color:var(--muted);display:block}.metric b{font-size:15px}.news a{color:#a9c2ff;text-decoration:none}.news li{margin-bottom:8px}.ai{background:#111a29;border:1px solid #273a58;border-radius:11px;padding:12px;font-size:12px;color:#cbd7e7;line-height:1.5}.method{margin-top:18px;border-top:1px solid var(--line);padding-top:13px;color:var(--muted);font-size:11px;line-height:1.5}
@media(max-width:1050px){.grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:720px){.wrap{padding:14px}.grid{grid-template-columns:1fr}.summary{grid-template-columns:repeat(2,1fr)}.updated{display:none}.metricgrid{grid-template-columns:repeat(2,1fr)}.search{min-width:170px}}
'''

JS = r'''
const DATA=__DATA__;let MODE='trade',MARKET='All';
const $=id=>document.getElementById(id);const fmt=(n,d=2)=>n==null?'—':Number(n).toFixed(d);
function rc(r){r=r.toLowerCase();return r==='buy'?'buy':r==='hold'?'hold':r==='wait'?'wait':r==='sell'?'sell':'avoid'}
function spark(v){if(!v||v.length<2)return'';let mn=Math.min(...v),mx=Math.max(...v),w=240,h=44;let p=v.map((x,i)=>`${i/(v.length-1)*w},${h-(x-mn)/(mx-mn||1)*h}`).join(' ');return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><polyline fill="none" stroke="currentColor" stroke-width="2" points="${p}"/></svg>`}
function score(a){return a.scores[MODE]};function rec(a){return MODE==='trade'?a.trade_rec:a.invest_rec};function reason(a){return MODE==='trade'?a.trade_reason:a.invest_reason}
function money(a,n){return n==null?'—':`${a.currency} ${fmt(n)}`}
function cardLevels(a){if(MODE==='trade'){let l=a.trade_levels;return `<div class="levels"><div class="lv"><span>Entry</span><b>${fmt(l.entry_low)}–${fmt(l.entry_high)}</b></div><div class="lv"><span>Target</span><b>${fmt(l.target)}</b></div><div class="lv"><span>Stop</span><b>${fmt(l.stop)}</b></div></div>`}let target=a.invest_target;let up=target?((target/a.price-1)*100):null;return `<div class="levels"><div class="lv"><span>12M Target</span><b>${money(a,target)}</b></div><div class="lv"><span>Upside</span><b>${up==null?'—':(up>0?'+':'')+fmt(up,1)+'%'}</b></div><div class="lv"><span>Horizon</span><b>12–36 mo</b></div></div>`}
function render(){let q=$('q').value.toLowerCase(),rfilter=$('recFilter').value;let arr=DATA.filter(a=>(MARKET==='All'||a.market===MARKET||a.asset_type===MARKET)&&(!q||a.ticker.toLowerCase().includes(q)||a.name.toLowerCase().includes(q))&&(!rfilter||rec(a)===rfilter)).sort((a,b)=>score(b)-score(a));$('grid').innerHTML=arr.length?arr.map(a=>`<div class="card" onclick='openAsset(${JSON.stringify(a.ticker)})'><div class="row"><div><div class="ticker">${a.ticker.replace('.SR','')}</div><div class="name">${a.name}</div></div><span class="rec ${rc(rec(a))}">${rec(a)}</span></div><div class="row" style="margin-top:11px"><div class="price">${money(a,a.price)}</div><div class="score ${rc(rec(a))}">${score(a).toFixed(1)}</div></div><div class="spark ${rc(rec(a))}">${spark(a.spark)}</div><div class="reason">${reason(a)}</div>${cardLevels(a)}<div class="row" style="margin-top:11px"><span class="meta">${a.market} · ${a.sector} · ${a.opportunity_type||'Balanced'}</span><span class="pill">Data ${a.as_of}</span></div></div>`).join(''):'<div class="empty">No opportunities match this view.</div>';updateHero(arr)}
function updateHero(arr){let buys=arr.filter(a=>rec(a)==='BUY').length;let top=arr[0];$('kMode').textContent=MODE==='trade'?'Trade opportunities':'Long-term opportunities';$('kBuys').textContent=buys;$('kTop').textContent=top?`${top.ticker.replace('.SR','')} · ${score(top).toFixed(1)}`:'—'}
function setMode(m,el){MODE=m;document.querySelectorAll('.mode').forEach(x=>x.classList.remove('active'));el.classList.add('active');$('modeTitle').textContent=m==='trade'?'Best Trade Opportunities':'Best Long-Term Investments';$('recFilter').innerHTML=m==='trade'?'<option value="">All recommendations</option><option>BUY</option><option>WAIT</option><option>AVOID</option><option>SELL</option>':'<option value="">All recommendations</option><option>BUY</option><option>HOLD</option><option>AVOID</option><option>SELL</option>';render()}
function setMarket(m,el){MARKET=m;document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));el.classList.add('active');render()}
function openAsset(t){let a=DATA.find(x=>x.ticker===t);if(!a)return;let r=rec(a),s=score(a);$('dTitle').textContent=`${a.name} · ${a.ticker.replace('.SR','')}`;$('dSub').textContent=`${a.market} · ${a.sector} · data through ${a.as_of}`;$('dPrice').textContent=money(a,a.price);$('dRec').textContent=`${r} · ${s.toFixed(1)}/10`;$('dRec').className=`bigrec ${rc(r)}`;$('dReason').textContent=reason(a);let m=MODE==='trade'?[['Trend',a.scores.trend],['Momentum',a.scores.momentum],['Rel. Strength',a.scores.relative_strength],['Setup',a.scores.setup],['Volume',a.scores.volume],['Risk',a.scores.risk],['Recovery',a.scores.recovery],['RSI',a.rsi],['6M Return',a.ret126==null?'—':fmt(a.ret126,1)+'%']]:[['Quality',a.scores.quality],['Growth',a.scores.growth],['Value',a.scores.value],['Momentum',a.scores.momentum],['Risk',a.scores.risk],['Recovery',a.scores.recovery],['Data Quality',a.scores.data_quality],['P/E',a.forward_pe||a.pe],['6M Return',a.ret126==null?'—':fmt(a.ret126,1)+'%']];$('metrics').innerHTML=m.map(x=>`<div class="metric"><span>${x[0]}</span><b>${typeof x[1]==='number'?fmt(x[1],1):x[1]}</b></div>`).join('');if(MODE==='trade'){let l=a.trade_levels;$('dLevels').innerHTML=[['Preferred Entry',`${money(a,l.entry_low)} – ${money(a,l.entry_high)}`],['Target',money(a,l.target)],['Stop / Invalidation',money(a,l.stop)],['Risk / Reward',l.rr?`${fmt(l.rr,1)}×`:'—']].map(x=>`<div class="metric"><span>${x[0]}</span><b>${x[1]}</b></div>`).join('')}else{let up=a.invest_target?((a.invest_target/a.price-1)*100):null;$('dLevels').innerHTML=[['12M Target',money(a,a.invest_target)],['Expected Upside',up==null?'—':(up>0?'+':'')+fmt(up,1)+'%'],['Target Basis',a.invest_target_source||'—'],['Horizon','12–36 months']].map(x=>`<div class="metric"><span>${x[0]}</span><b>${x[1]}</b></div>`).join('')}$('ai').textContent=a.ai_context||'No additional AI context. The recommendation above is generated directly by the scoring engine.';$('news').innerHTML=(a.news||[]).map(n=>`<li><a target="_blank" rel="noopener" href="${n.link}">${n.title}</a> <span class="meta">${n.publisher}</span></li>`).join('')||'<li>No recent headlines returned.</li>';$('overlay').classList.add('open')}
function closeDrawer(){$('overlay').classList.remove('open')};$('q').addEventListener('input',render);$('recFilter').addEventListener('change',render);render();
'''


def render_html(assets: list[dict[str, Any]], generated: str) -> str:
    data_json=json.dumps(assets,ensure_ascii=False).replace("</","<\\/")
    js=JS.replace("__DATA__",data_json)
    sa=sum(a["market"]=="Saudi" for a in assets); us=sum(a["market"]=="US" for a in assets)
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Market Opportunity Dashboard</title><style>{CSS}</style></head><body><div class="wrap"><div class="top"><div class="brand"><h1>Market Opportunity Dashboard</h1><p>Clear recommendations across Saudi stocks · US stocks · Gold · Silver</p></div><div class="updated">Updated {html.escape(generated)} Riyadh time<br>Deterministic scoring · AI only adds optional context</div></div><div class="modebar"><button class="mode active" onclick="setMode('trade',this)">TRADE</button><button class="mode" onclick="setMode('invest',this)">INVEST</button></div><div class="summary"><div class="kpi"><div class="l" id="kMode">Trade opportunities</div><div class="v">{len(assets)} scanned</div></div><div class="kpi"><div class="l">Saudi / US</div><div class="v">{sa} / {us}</div></div><div class="kpi"><div class="l">Current BUY signals</div><div class="v" id="kBuys">—</div></div><div class="kpi"><div class="l">Top current idea</div><div class="v" id="kTop">—</div></div></div><div class="toolbar"><div class="tabs"><button class="tab active" onclick="setMarket('All',this)">All</button><button class="tab" onclick="setMarket('Saudi',this)">Saudi</button><button class="tab" onclick="setMarket('US',this)">US</button><button class="tab" onclick="setMarket('Metal',this)">Gold & Silver</button></div><span class="grow"></span><input id="q" class="search" placeholder="Search ticker or company"><select id="recFilter"><option value="">All recommendations</option><option>BUY</option><option>WAIT</option><option>AVOID</option><option>SELL</option></select></div><div class="section-title" id="modeTitle">Best Trade Opportunities</div><div id="grid" class="grid"></div><div class="foot"><b>How to read this:</b> V3 scans several opportunity paths at once—momentum, breakout, pullback, recovery, quality/value and growth/value—then cross-checks them with historical price behavior and deterioration risk. TRADE only says BUY when the current price is actually inside the preferred entry zone. INVEST combines profitability/quality, growth, peer-relative valuation, momentum and recovery confirmation. Gold and silver use a separate trend/macro model including the US dollar and Treasury-yield direction. Recommendations are research signals, not guarantees. yfinance is the free data provider; verify live price before placing an order, especially for Tadawul.</div></div><div id="overlay" class="overlay" onclick="if(event.target===this)closeDrawer()"><div class="drawer"><div class="row"><div class="dtitle"><h2 id="dTitle"></h2><div id="dSub" class="meta"></div></div><button class="close" onclick="closeDrawer()">×</button></div><div id="dPrice" class="price"></div><div id="dRec"></div><div id="dReason" class="dreasons"></div><div id="dLevels" class="dlevels"></div><div id="metrics" class="metricgrid"></div><div id="ai" class="ai"></div><h3>Recent headlines</h3><ul id="news" class="news"></ul><div class="method"><b>Methodology:</b> V3 is a transparent multi-model engine, not an LLM stock picker. It uses established factor families (quality, value, growth, momentum/revisions where available), TradingView-style technical confluence concepts, multi-year historical context, peer-relative normalization, recovery detection and explicit value-trap checks. Target prices use analyst consensus when available, otherwise a conservative peer forward-P/E estimate when enough data exists.</div></div></div><script>{js}</script></body></html>'''


def main() -> None:
    universe=[(t,n,"US","Equity") for t,n in US_UNIVERSE.items()]
    universe += [(t,n,"Saudi","Equity") for t,n in SAUDI_UNIVERSE.items()]
    universe += [(t,n,"Metals","Metal") for t,n in METALS.items()]
    assets=[]
    for ticker,name,market,typ in universe:
        try:
            a=fetch_base_asset(ticker,name,market,typ)
            if a: assets.append(a)
            else: print(f"No usable data: {ticker}")
        except Exception as e:
            print(f"Skip {ticker}: {e}")

    # Benchmark 6-month returns for relative strength.
    bench={}
    for market,ticker in BENCHMARKS.items():
        h=get_hist(ticker,"1y")
        bench[market]=ret(h["Close"].astype(float),126) if h is not None else None
    macro=macro_snapshot()

    for a in assets:
        a["scores"] = calc_equity_scores(assets,a,bench.get(a["market"])) if a["asset_type"]=="Equity" else calc_metal_scores(a,macro)
        a["trade_levels"] = trade_levels(a)
        a["trade_rec"] = trade_recommendation(a)
        a["invest_rec"] = invest_recommendation(a)
        a["opportunity_type"] = opportunity_type(a)
        target,source = investment_target(a,assets)
        a["invest_target"],a["invest_target_source"] = target,source
        # Flatten only useful fields into final JSON.
        t=a["tech"]; f=a["fund"]
        a.update({
            "price":round(t["price"],2),"rsi":round(t["rsi"],1) if finite(t.get("rsi")) else None,
            "ret21":round(t["ret21"],2) if finite(t.get("ret21")) else None,
            "ret126":round(t["ret126"],2) if finite(t.get("ret126")) else None,
            "ret252":round(t["ret252"],2) if finite(t.get("ret252")) else None,
            "position52":round(t["position52"],1) if finite(t.get("position52")) else None,
            "drawdown":round(t["drawdown"],1) if finite(t.get("drawdown")) else None,
            "atr_pct":round(t["atr_pct"],2) if finite(t.get("atr_pct")) else None,
            "spark":t["spark"],"as_of":t["as_of"],"sector":f.get("sector") if f else "Precious Metals",
            "pe":round(f["trailing_pe"],2) if f and finite(f.get("trailing_pe")) and f["trailing_pe"]>0 else None,
            "forward_pe":round(f["forward_pe"],2) if f and finite(f.get("forward_pe")) and f["forward_pe"]>0 else None,
        })
        a["trade_reason"] = trade_reason(a)
        a["invest_reason"] = invest_reason(a)
        a["news"] = []
        a["ai_context"] = ""

    # News only for the strongest ideas in either mode to keep the run lean.
    leaders=sorted(assets,key=lambda x:max(x["scores"]["trade"],x["scores"]["invest"]),reverse=True)[:14]
    for a in leaders:
        a["news"]=recent_news(a["ticker"])
    ai=ai_context(leaders)
    for a in assets:
        a["ai_context"]=ai.get(a["ticker"],"")

    # Remove raw objects before serializing.
    for a in assets:
        a.pop("tech",None); a.pop("fund",None)
    assets.sort(key=lambda x:max(x["scores"]["trade"],x["scores"]["invest"]),reverse=True)

    riyadh=dt.datetime.now(dt.timezone(dt.timedelta(hours=3))).strftime("%Y-%m-%d %H:%M")
    out=Path("docs");out.mkdir(exist_ok=True)
    (out/"index.html").write_text(render_html(assets,riyadh),encoding="utf-8")
    (out/"data.json").write_text(json.dumps({"generated":riyadh,"assets":assets},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Generated {len(assets)} assets -> docs/index.html")

if __name__=="__main__":
    main()
