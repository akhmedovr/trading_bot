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
    # =========================================================================
# РИСК-МЕНЕДЖМЕНТ: построение сделки
# =========================================================================
def build_trade(side, price, atr, balance):
    """
    Строит сделку на основе стороны, цены, ATR и баланса.
    Возвращает dict с параметрами ордера и риск-менеджментом.
    """
    risk_amount = balance * CONFIG["RISK_PER_TRADE_PCT"]
    stop_dist = atr * CONFIG["ATR_SL_MULT"]
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
        "side": side,
        "entry": price,
        "sl": sl,
        "tp": tp,
        "size": size,
        "risk_usd": risk_amount,
        "potential_usd": potential,
    }


# =========================================================================
# УВЕДОМЛЕНИЯ В TELEGRAM
# =========================================================================
def notify_open(symbol, trade, score):
    """Красиво форматирует открытие сделки и отправляет в Telegram."""
    side_emoji = "🟢" if trade["side"] == "long" else "🔴"
    side_text = trade["side"].upper()
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
        f"{side_emoji} <b>СИГНАЛ {side_text} {symbol}</b>\n\n"
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


def notify_close(symbol, pos, exit_price, p, balance):
    """Красиво форматирует закрытие сделки и отправляет в Telegram."""
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
# ПОЛУЧЕНИЕ ДАННЫХ С БИРЖИ
# =========================================================================
def fetch_ohlcv(exchange, symbol, timeframe, limit=300):
    """Загружает свечи и добавляет индикаторы."""
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts")
    return add_indicators(df)


def make_exchange():
    """Создаёт объект биржи ccxt (публичный, без ключей)."""
    exchange_class = getattr(ccxt, CONFIG["EXCHANGE_ID"])
    ex = exchange_class({
        "enableRateLimit": True,
        "options": {"defaultType": CONFIG["MARKET_TYPE"]},
    })
    return ex


# =========================================================================
# ГЛАВНЫЙ ЦИКЛ PAPER-ТОРГОВЛИ
# =========================================================================
def run_bot():
    """Живой paper-трейдинг: опрос рынка, сигналы, виртуальные сделки."""
    log("Запуск бота в PAPER режиме")
    log(f"Монеты: {', '.join(CONFIG['SYMBOLS'])}")
    log(f"Стартовый баланс: ${CONFIG['VIRTUAL_BALANCE_START']:.2f}")
    log(f"Плечо: {CONFIG['LEVERAGE']}x")

    send_telegram_message(
        f"🚀 <b>Бот запущен (PAPER)</b>\n"
        f"Монет: {len(CONFIG['SYMBOLS'])}\n"
        f"Баланс: ${CONFIG['VIRTUAL_BALANCE_START']:.2f}"
    )

    ex = make_exchange()
    balance = CONFIG["VIRTUAL_BALANCE_START"]
    positions = {}  # {symbol: pos_dict}
    last_candle_ts = {}

    while True:
        try:
            for symbol in CONFIG["SYMBOLS"]:
                try:
                    df = fetch_ohlcv(ex, symbol, CONFIG["TIMEFRAME"], CONFIG["OHLCV_LIMIT"])
                    if len(df) < 50:
                        continue

                    current_ts = df.index[-1]

                    # Если уже есть открытая позиция — проверяем выход
                    if symbol in positions:
                        pos = positions[symbol]
                        last_closed = df.iloc[-2]
                        exit_price = check_exit(pos, last_closed["high"], last_closed["low"])

                        if exit_price is not None:
                            p = pnl(pos, exit_price)
                            balance += p
                            log(f"[{symbol}] Закрыта {pos['side']} по {exit_price:.6f}, PnL=${p:+.2f}, Баланс=${balance:.2f}")
                            notify_close(symbol, pos, exit_price, p, balance)
                            del positions[symbol]
                        continue

                    # Нет позиции — ищем сигнал
                    if last_candle_ts.get(symbol) == current_ts:
                        continue
                    last_candle_ts[symbol] = current_ts

                    signal = get_signal(df)
                    if signal is None:
                        continue

                    trade = build_trade(signal["side"], signal["price"], signal["atr"], balance)
                    positions[symbol] = trade
                    log(f"[{symbol}] Сигнал {signal['side'].upper()} score={signal['score']}, вход={signal['price']:.6f}")
                    notify_open(symbol, trade, signal["score"])

                except Exception as e:
                    log(f"[{symbol}] Ошибка: {e}")

            time.sleep(CONFIG["POLL_INTERVAL_SEC"])

        except KeyboardInterrupt:
            log("Остановка бота (Ctrl+C)")
            break
        except Exception as e:
            log(f"Общая ошибка: {e}")
            log(traceback.format_exc())
            time.sleep(30)


# =========================================================================
# БЭКТЕСТ
# =========================================================================
def run_backtest():
    """Простой бэктест на исторических данных по каждой монете."""
    log("Запуск бэктеста")
    ex = make_exchange()

    for symbol in CONFIG["SYMBOLS"]:
        try:
            log(f"\n=== {symbol} ===")
            df = fetch_ohlcv(ex, symbol, CONFIG["TIMEFRAME"], CONFIG["BACKTEST_LIMIT"])

            if len(df) < 250:
                log(f"Недостаточно данных ({len(df)} свечей)")
                continue

            balance = CONFIG["VIRTUAL_BALANCE_START"]
            pos = None
            trades = 0
            wins = 0
            losses = 0

            for i in range(250, len(df)):
                window = df.iloc[:i+1]
                if len(window) < 250:
                    continue

                row = window.iloc[-1]

                if pos is not None:
                    exit_price = check_exit(pos, row["high"], row["low"])
                    if exit_price is not None:
                        p = pnl(pos, exit_price)
                        balance += p
                        trades += 1
                        if p > 0:
                            wins += 1
                        else:
                            losses += 1
                        pos = None
                    continue

                signal = get_signal(window)
                if signal is not None:
                    pos = build_trade(signal["side"], signal["price"], signal["atr"], balance)

            log(f"Сделок: {trades} | Прибыльных: {wins} | Убыточных: {losses}")
            log(f"Финальный баланс: ${balance:.2f} (стартовый: ${CONFIG['VIRTUAL_BALANCE_START']:.2f})")
            pnl_pct = (balance - CONFIG["VIRTUAL_BALANCE_START"]) / CONFIG["VIRTUAL_BALANCE_START"] * 100
            log(f"P&L: {pnl_pct:+.2f}%")

        except Exception as e:
            log(f"Ошибка бэктеста для {symbol}: {e}")


# =========================================================================
# ТОЧКА ВХОДА
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Трендовый бот для Bybit (paper)")
    parser.add_argument("--backtest", action="store_true", help="Запустить бэктест")
    args = parser.parse_args()

    if args.backtest:
        CONFIG["MODE"] = "backtest"
        run_backtest()
    else:
        run_bot()


if __name__ == "__main__":
    main()
