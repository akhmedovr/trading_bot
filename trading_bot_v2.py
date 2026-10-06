"""
trading_bot_v2.py
Трендовый торговый бот для Bybit USDT-перпетуалов (PAPER режим).
Мультимонетный, 1x плечо, уведомления в Telegram.
"""

import argparse
import sys
import time
from datetime import datetime

import ccxt
import numpy as np
import pandas as pd

from notify import send_telegram_message


# =========================================================================
# CONFIG
# =========================================================================
CONFIG = {
    "MODE": "paper",
    "EXCHANGE_ID": "bybit",
    "MARKET_TYPE": "swap",
    "TIMEFRAME": "1h",
    "SYMBOLS": [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "SOL/USDT:USDT",
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
}


# =========================================================================
# LOG
# =========================================================================
def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


# =========================================================================
# ИНДИКАТОРЫ
# =========================================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]

    df["ema_fast"] = c.ewm(span=CONFIG["EMA_FAST"], adjust=False).mean()
    df["ema_slow"] = c.ewm(span=CONFIG["EMA_SLOW"], adjust=False).mean()

    macd_fast = c.ewm(span=CONFIG["MACD_FAST"], adjust=False).mean()
    macd_slow = c.ewm(span=CONFIG["MACD_SLOW"], adjust=False).mean()
    df["macd"] = macd_fast - macd_slow
    df["macd_signal"] = df["macd"].ewm(span=CONFIG["MACD_SIGNAL"], adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    period = CONFIG["RSI_PERIOD"]
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    prev_close = c.shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/CONFIG["ATR_PERIOD"], adjust=False,
                        min_periods=CONFIG["ATR_PERIOD"]).mean()

    n = CONFIG["BREAKOUT_LOOKBACK"]
    df["res_30"] = df["high"].shift(1).rolling(n).max()
    df["sup_30"] = df["low"].shift(1).rolling(n).min()

    return df


# =========================================================================
# СИГНАЛЫ
# =========================================================================
def technical_score(df: pd.DataFrame) -> dict:
    row = df.iloc[-1]
    sl, ss = 0, 0
    rl, rs = [], []

    if row["ema_fast"] > row["ema_slow"]:
        sl += 1
        rl.append("EMA50 > EMA200 (восходящий тренд)")
    elif row["ema_fast"] < row["ema_slow"]:
        ss += 1
        rs.append("EMA50 < EMA200 (нисходящий тренд)")

    if row["macd_hist"] > 0:
        sl += 1
        rl.append("MACD histogram > 0 (бычий импульс)")
    elif row["macd_hist"] < 0:
        ss += 1
        rs.append("MACD histogram < 0 (медвежий импульс)")

    if row["rsi"] < 30:
        sl += 1
        rl.append(f"RSI={row['rsi']:.1f} < 30 (перепроданность)")
    elif row["rsi"] > 70:
        ss += 1
        rs.append(f"RSI={row['rsi']:.1f} > 70 (перекупленность)")

    if not np.isnan(row["res_30"]) and row["close"] > row["res_30"]:
        sl += 1
        rl.append(f"Пробой сопротивления за {CONFIG['BREAKOUT_LOOKBACK']} свечей")
    if not np.isnan(row["sup_30"]) and row["close"] < row["sup_30"]:
        ss += 1
        rs.append(f"Пробой поддержки за {CONFIG['BREAKOUT_LOOKBACK']} свечей")

    return {
        "score_long": sl,
        "score_short": ss,
        "reasons_long": rl,
        "reasons_short": rs,
        "row": row,
    }


def analyze_symbol(ex, symbol: str) -> dict:
    """
    Анализирует ОДНУ монету. ВСЕГДА возвращает сторону (long/short),
    даже если score ниже порога.
    """
    try:
        df = fetch_df(ex, symbol, CONFIG["TIMEFRAME"], CONFIG["OHLCV_LIMIT"])
    except Exception as e:
        log(f"[{symbol}] Ошибка загрузки: {e}")
        return None

    if len(df) < max(CONFIG["EMA_SLOW"], CONFIG["BREAKOUT_LOOKBACK"]) + 5:
        return None

    result = technical_score(df)
    row = result["row"]

    if pd.isna(row["atr"]) or row["atr"] <= 0:
        return None

    sl = result["score_long"]
    ss = result["score_short"]

    if sl >= ss:
        side = "long"
        score = sl
        reasons = result["reasons_long"]
    else:
        side = "short"
        score = ss
        reasons = result["reasons_short"]

    return {
        "symbol": symbol,
        "side": side,
        "score": score,
        "score_long": sl,
        "score_short": ss,
        "reasons": reasons,
        "price": float(row["close"]),
        "atr": float(row["atr"]),
        "passed": score >= CONFIG["SCORE_THRESHOLD"],
    }


# =========================================================================
# РИСК-МЕНЕДЖМЕНТ
# =========================================================================
def build_trade(side: str, price: float, atr: float, balance: float, symbol: str) -> dict:
    stop_dist = atr * CONFIG["ATR_SL_MULT"]
    risk_amount = balance * CONFIG["RISK_PER_TRADE_PCT"]
    size = risk_amount / stop_dist
    max_size = balance * CONFIG["MAX_POSITION_PCT"] / price
    if size > max_size:
        size = max_size

    if side == "long":
        sl = price - stop_dist
        tp = price + stop_dist * CONFIG["RR_RATIO"]
    else:
        sl = price + stop_dist
        tp = price - stop_dist * CONFIG["RR_RATIO"]

    potential = abs(tp - price) * size
    return {
        "symbol": symbol,
        "side": side,
        "entry": price,
        "sl": sl,
        "tp": tp,
        "size": size,
        "risk_usd": risk_amount,
        "potential_usd": potential,
    }


# =========================================================================
# БИРЖА
# =========================================================================
def make_exchange():
    exchange_class = getattr(ccxt, CONFIG["EXCHANGE_ID"])
    ex = exchange_class({
        "enableRateLimit": True,
        "options": {"defaultType": CONFIG["MARKET_TYPE"]},
    })
    return ex


def fetch_df(ex, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts")
    return add_indicators(df)


# =========================================================================
# БУМАЖНАЯ ТОРГОВЛЯ
# =========================================================================
def check_exit(pos: dict, high: float, low: float):
    if pos["side"] == "long":
        if low <= pos["sl"]:
            return pos["sl"]
        if high >= pos["tp"]:
            return pos["tp"]
    else:
        if high >= pos["sl"]:
            return pos["sl"]
        if low <= pos["tp"]:
            return pos["tp"]
    return None


def pnl(pos: dict, exit_price: float) -> float:
    d = exit_price - pos["entry"]
    gross = (d if pos["side"] == "long" else -d) * pos["size"]
    fees = 0.001 * (pos["entry"] + exit_price) * pos["size"]
    return gross - fees


# =========================================================================
# УВЕДОМЛЕНИЯ
# =========================================================================
def notify_open(trade: dict, score: int):
    emoji = "🟢" if trade["side"] == "long" else "🔴"
    sym = trade["symbol"]
    entry = trade["entry"]
    sl = trade["sl"]
    tp = trade["tp"]
    size = trade["size"]
    risk = trade["risk_usd"]
    potential = trade["potential_usd"]
    sl_pct = abs(sl - entry) / entry * 100
    tp_pct = abs(tp - entry) / entry * 100
    rr = potential / risk if risk > 0 else 0

    msg = (
        f"{emoji} <b>СИГНАЛ {trade['side'].upper()} {sym}</b>\n\n"
        f"📊 Вход: ${entry:.6f}\n"
        f"🛑 Стоп: ${sl:.6f}  (-{sl_pct:.2f}%)\n"
        f"🎯 Тейк: ${tp:.6f}  (+{tp_pct:.2f}%)\n\n"
        f"⚖️ Размер: {size:.6f}\n"
        f"💵 Сумма: ${size * entry:.2f}\n\n"
        f"⚠️ Риск: <b>${risk:.2f}</b>\n"
        f"💰 Потенциал: <b>${potential:.2f}</b>\n"
        f"📊 R:R = 1:{rr:.2f}\n\n"
        f"🎯 Score: {score}"
    )
    send_telegram_message(msg)


def notify_close(symbol: str, pos: dict, exit_price: float, p: float, balance: float):
    emoji = "✅" if p >= 0 else "❌"
    pct = (exit_price - pos["entry"]) / pos["entry"] * 100
    if pos["side"] == "short":
        pct = -pct

    msg = (
        f"{emoji} <b>СДЕЛКА ЗАКРЫТА</b>\n\n"
        f"📊 {symbol} ({pos['side'].upper()})\n"
        f"💰 Вход: ${pos['entry']:.6f}\n"
        f"🎯 Выход: ${exit_price:.6f}\n\n"
        f"📈 PnL: <b>${p:+.2f}</b>\n"
        f"📊 Процент: <b>{pct:+.2f}%</b>\n"
        f"💼 Баланс: ${balance:.2f}"
    )
    send_telegram_message(msg)


# =========================================================================
# ГЛАВНЫЙ ЦИКЛ PAPER
# =========================================================================
def run_bot():
    log("Запуск бота в PAPER режиме")
    log(f"Монеты: {', '.join(CONFIG['SYMBOLS'])}")
    log(f"Стартовый баланс: ${CONFIG['VIRTUAL_BALANCE_START']:.2f}")

    send_telegram_message(
        f"🚀 <b>Торговый бот запущен (PAPER)</b>\n"
        f"Монет: {len(CONFIG['SYMBOLS'])}\n"
        f"Баланс: ${CONFIG['VIRTUAL_BALANCE_START']:.2f}"
    )

    ex = make_exchange()
    balance = CONFIG["VIRTUAL_BALANCE_START"]
    positions = {}
    last_candle_ts = {}

    while True:
        try:
            for symbol in CONFIG["SYMBOLS"]:
                try:
                    df = fetch_df(ex, symbol, CONFIG["TIMEFRAME"], CONFIG["OHLCV_LIMIT"])
                    if len(df) < 50:
                        continue

                    current_ts = df.index[-1]

                    if symbol in positions:
                        pos = positions[symbol]
                        last_closed = df.iloc[-2]
                        exit_price = check_exit(pos, last_closed["high"], last_closed["low"])
                        if exit_price is not None:
                            p = pnl(pos, exit_price)
                            balance += p
                            log(f"[{symbol}] Закрыта {pos['side']} по {exit_price:.6f}, PnL=${p:+.2f}")
                            notify_close(symbol, pos, exit_price, p, balance)
                            del positions[symbol]
                        continue

                    if last_candle_ts.get(symbol) == current_ts:
                        continue
                    last_candle_ts[symbol] = current_ts

                    analysis = analyze_symbol(ex, symbol)
                    if analysis is None or not analysis["passed"]:
                        continue

                    trade = build_trade(
                        analysis["side"],
                        analysis["price"],
                        analysis["atr"],
                        balance,
                        symbol,
                    )
                    positions[symbol] = trade
                    log(f"[{symbol}] Сигнал {analysis['side'].upper()} score={analysis['score']}")
                    notify_open(trade, analysis["score"])

                except Exception as e:
                    log(f"[{symbol}] Ошибка: {e}")

            time.sleep(CONFIG["POLL_INTERVAL_SEC"])

        except KeyboardInterrupt:
            log("Остановка (Ctrl+C)")
            break
        except Exception as e:
            log(f"Общая ошибка: {e}")
            time.sleep(30)


# =========================================================================
# БЭКТЕСТ
# =========================================================================
def run_backtest():
    log("Запуск бэктеста")
    ex = make_exchange()

    for symbol in CONFIG["SYMBOLS"]:
        try:
            log(f"\n=== {symbol} ===")
            df = fetch_df(ex, symbol, CONFIG["TIMEFRAME"], CONFIG["BACKTEST_LIMIT"])
            if len(df) < 250:
                continue

            balance = CONFIG["VIRTUAL_BALANCE_START"]
            pos = None
            trades = wins = losses = 0

            for i in range(250, len(df)):
                window = df.iloc[:i+1]
                row = window.iloc[-1]

                if pos is not None:
                    exit_price = check_exit(pos, row["high"], row["low"])
                    if exit_price is not None:
                        p = pnl(pos, exit_price)
                        balance += p
                        trades += 1
                        wins += 1 if p > 0 else 0
                        losses += 1 if p <= 0 else 0
                        pos = None
                    continue

                result = technical_score(window)
                if len(window) < max(CONFIG["EMA_SLOW"], CONFIG["BREAKOUT_LOOKBACK"]) + 5:
                    continue
                r = result["row"]
                if pd.isna(r["atr"]) or r["atr"] <= 0:
                    continue

                sl = result["score_long"]
                ss = result["score_short"]
                if sl >= ss and sl >= CONFIG["SCORE_THRESHOLD"]:
                    pos = build_trade("long", r["close"], r["atr"], balance, symbol)
                elif ss > sl and ss >= CONFIG["SCORE_THRESHOLD"]:
                    pos = build_trade("short", r["close"], r["atr"], balance, symbol)

            log(f"Сделок: {trades} | Прибыльных: {wins} | Убыточных: {losses}")
            log(f"Баланс: ${balance:.2f}")
            pnl_pct = (balance - CONFIG["VIRTUAL_BALANCE_START"]) / CONFIG["VIRTUAL_BALANCE_START"] * 100
            log(f"P&L: {pnl_pct:+.2f}%")
        except Exception as e:
            log(f"Ошибка бэктеста {symbol}: {e}")


# =========================================================================
# ТОЧКА ВХОДА
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Трендовый бот Bybit (paper)")
    parser.add_argument("--backtest", action="store_true", help="Запустить бэктест")
    args = parser.parse_args()

    if args.backtest:
        CONFIG["MODE"] = "backtest"
        run_backtest()
    else:
        run_bot()


if __name__ == "__main__":
    main()
