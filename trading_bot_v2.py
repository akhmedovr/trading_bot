"""
trading_bot_v2.py
Трендовый торговый бот для Bybit USDT-перпетуалов (PAPER режим).
Мультимонетный, 1x плечо, уведомления в Telegram.
"""

import time
import argparse
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import ccxt

from notify import send_telegram_message


# =========================================================================
# CONFIG
# =========================================================================
CONFIG = {
    "MODE": "paper",
    "SEND_TELEGRAM_IN_BACKTEST": False,
    "EXCHANGE_ID": "bybit",
    "MARKET_TYPE": "swap",
    "TIMEFRAME": "1h",
    "SYMBOLS": [
        "DOGE/USDT:USDT",
        "XRP/USDT:USDT",
        "1000PEPE/USDT:USDT",
        "SUI/USDT:USDT",
        "AVAX/USDT:USDT",
        "ADA/USDT:USDT",
        "JTO/USDT:USDT",
        "INJ/USDT:USDT",
    ],
    "VIRTUAL_BALANCE_START": 200.0,
    "LEVERAGE": 1,
    "RISK_PER_TRADE_PCT": 0.01,
    "ATR_SL_MULT": 2.0,
    "RR_RATIO": 2.0,
    "MAX_POSITION_PCT": 0.5,
    "EMA_FAST": 50,
    "EMA_SLOW": 200,
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,
    "RSI_PERIOD": 14,
    "ATR_PERIOD": 14,
    "BREAKOUT_LOOKBACK": 30,
    "SCORE_THRESHOLD": 3,
    "OHLCV_LIMIT": 300,
    "BACKTEST_LIMIT": 1000,
    "POLL_INTERVAL_SEC": 60,
    "VERBOSE": True,
}


# =========================================================================
# ЛОГИРОВАНИЕ
# =========================================================================
def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


# =========================================================================
# ИНДИКАТОРЫ
# =========================================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=CONFIG["EMA_FAST"], adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=CONFIG["EMA_SLOW"], adjust=False).mean()

    ema_macd_fast = df["close"].ewm(span=CONFIG["MACD_FAST"], adjust=False).mean()
    ema_macd_slow = df["close"].ewm(span=CONFIG["MACD_SLOW"], adjust=False).mean()
    df["macd"] = ema_macd_fast - ema_macd_slow
    df["macd_signal"] = df["macd"].ewm(span=CONFIG["MACD_SIGNAL"], adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    period = CONFIG["RSI_PERIOD"]
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/CONFIG["ATR_PERIOD"], adjust=False,
                        min_periods=CONFIG["ATR_PERIOD"]).mean()

    n = CONFIG["BREAKOUT_LOOKBACK"]
    df["res_30"] = df["high"].shift(1).rolling(n).max()
    df["sup_30"] = df["low"].shift(1).rolling(n).min()

    return df


# =========================================================================
# SCORE
# =========================================================================
def technical_score(df: pd.DataFrame) -> dict:
    row = df.iloc[-1]
    score_long = 0
    score_short = 0
    reasons_long = []
    reasons_short = []

    if row["ema_fast"] > row["ema_slow"]:
        score_long += 1
        reasons_long.append("EMA50 > EMA200")
    elif row["ema_fast"] < row["ema_slow"]:
        score_short += 1
        reasons_short.append("EMA50 < EMA200")

    if row["macd_hist"] > 0:
        score_long += 1
        reasons_long.append("MACD hist > 0")
    elif row["macd_hist"] < 0:
        score_short += 1
        reasons_short.append("MACD hist < 0")

    if row["rsi"] < 30:
        score_long += 1
        reasons_long.append(f"RSI={row['rsi']:.1f} < 30")
    elif row["rsi"] > 70:
        score_short += 1
        reasons_short.append(f"RSI={row['rsi']:.1f} > 70")

    if not np.isnan(row["res_30"]) and row["close"] > row["res_30"]:
        score_long += 1
        reasons_long.append("Пробой сопротивления")
    if not np.isnan(row["sup_30"]) and row["close"] < row["sup_30"]:
        score_short += 1
        reasons_short.append("Пробой поддержки")

    return {
        "score_long": score_long,
        "score_short": score_short,
        "reasons_long": reasons_long,
        "reasons_short": reasons_short,
        "row": row,
    }


def get_signal(df: pd.DataFrame):
    if len(df) < max(CONFIG["EMA_SLOW"], CONFIG["BREAKOUT_LOOKBACK"]) + 5:
        return None

    result = technical_score(df)
    row = result["row"]

    if pd.isna(row["atr"]) or row["atr"] <= 0:
        return None

    long_ok = result["score_long"] >= CONFIG["SCORE_THRESHOLD"]
    short_ok = result["score_short"] >= CONFIG["SCORE_THRESHOLD"]

    if long_ok and short_ok:
        if result["score_long"] >= result["score_short"]:
            side, score, reasons = "long", result["score_long"], result["reasons_long"]
        else:
            side, score, reasons = "short", result["score_short"], result["reasons_short"]
    elif long_ok:
        side, score, reasons = "long", result["score_long"], result["reasons_long"]
    elif short_ok:
        side, score, reasons = "short", result["score_short"], result["reasons_short"]
    else:
        return None

    return {
        "side": side,
        "score": score,
        "reasons": reasons,
        "price": float(row["close"]),
        "atr": float(row["atr"]),
    }
    github.com/akhmedovr/trading_bot/edit/main/trading_bot_v2.py
