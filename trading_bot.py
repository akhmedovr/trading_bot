import sys
import time
import ccxt
import numpy as np
import pandas as pd

# ------------------------- НАСТРОЙКИ -------------------------
CONFIG = {
    "exchange": "bybit",
    "symbol": "BTC/USDT:USDT",
    "timeframe": "1h",
    "live": False,
    "testnet": True,
    "api_key": "",
    "api_secret": "",
    "risk_per_trade": 0.01,
    "atr_stop_mult": 2.0,
    "rr": 2.0,
    "open_threshold": 3,
    "use_macro": True,
    "use_funding": True,
    "leverage": 3,
    "fee": 0.0005,
    "paper_balance": 10000.0,
    "loop_seconds": 30,
}

# ------------------------- ИНДИКАТОРЫ -------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - c.shift()).abs(),
        (df["low"] - c.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    return df

# ------------------------- ГРАФИЧЕСКИЙ АНАЛИЗ -------------------------
def chart_score(df: pd.DataFrame, lookback: int = 30) -> int:
    window = df.iloc[-lookback - 1:-1]
    resistance, support = window["high"].max(), window["low"].min()
    price = df["close"].iloc[-1]
    if price > resistance:
        return 2
    if price < support:
        return -2
    return 0

# ------------------------- ТЕХНИЧЕСКИЙ АНАЛИЗ -------------------------
def technical_score(df: pd.DataFrame) -> int:
    last = df.iloc[-1]
    score = 0
    score += 1 if last["ema50"] > last["ema200"] else -1
    score += 1 if last["macd_hist"] > 0 else -1
    if last["rsi"] < 30:
        score += 1
    elif last["rsi"] > 70:
        score -= 1
    return score

# ------------------------- МАКРО-АНАЛИЗ -------------------------
def macro_score() -> int:
    try:
        import yfinance as yf
        def hist(t):
            return yf.Ticker(t).history(period="3mo")["Close"].dropna()
        spx, vix, dxy, tnx = hist("^GSPC"), hist("^VIX"), hist("DX-Y.NYB"), hist("^TNX")
        s = 0
        s += 1 if spx.iloc[-1] > spx.rolling(50).mean().iloc[-1] else -1
        s += -1 if vix.iloc[-1] > 25 else 1
        s += -1 if dxy.iloc[-1] > dxy.iloc[-20] else 1
        s += -1 if tnx.iloc[-1] > tnx.iloc[-20] else 1
        return int(np.sign(s) * min(abs(s), 2))
    except Exception as e:
        print("Макро-данные недоступны:", e)
        return 0

def funding_score(ex) -> int:
    try:
        rate = ex.fetch_funding_rate(CONFIG["symbol"])["fundingRate"]
        if rate > 0.0005:
            return -1
        if rate < -0.0005:
            return 1
    except Exception as e:
        print("Funding недоступен:", e)
    return 0

# ------------------------- СИГНАЛ -------------------------
def get_signal(df: pd.DataFrame, macro: int) -> tuple:
    total = technical_score(df) + chart_score(df) + macro
    th = CONFIG["open_threshold"]
    if total >= th:
        return "long", total
    if total <= -th:
        return "short", total
    return "none", total

# ------------------------- РИСК-МЕНЕДЖМЕНТ -------------------------
def build_trade(side: str, price: float, atr: float, balance: float) -> dict:
    stop_dist = atr * CONFIG["atr_stop_mult"]
    risk_amount = balance * CONFIG["risk_per_trade"]
    size = risk_amount / stop_dist
    size = min(size, balance * CONFIG["leverage"] / price)
    if side == "long":
        sl, tp = price - stop_dist, price + stop_dist * CONFIG["rr"]
    else:
        sl, tp = price + stop_dist, price - stop_dist * CONFIG["rr"]
    return {"side": side, "entry": price, "sl": sl, "tp": tp, "size": size}

# ------------------------- БИРЖА -------------------------
def make_exchange():
    ex = getattr(ccxt, CONFIG["exchange"])({
        "apiKey": CONFIG["api_key"],
        "secret": CONFIG["api_secret"],
        "options": {"defaultType": "swap"},
    })
    if CONFIG["live"] and CONFIG["testnet"]:
        ex.set_sandbox_mode(True)
    ex.load_markets()
    return ex

def fetch_df(ex, limit=500) -> pd.DataFrame:
    raw = ex.fetch_ohlcv(CONFIG["symbol"], CONFIG["timeframe"], limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    return add_indicators(df)

def place_live_order(ex, t: dict):
    sym, side = CONFIG["symbol"], ("buy" if t["side"] == "long" else "sell")
    for fn, args in ((ex.set_margin_mode, ("isolated", sym)), (ex.set_leverage, (CONFIG["leverage"], sym))):
        try:
            fn(*args)
        except Exception as e:
            print("Плечо/маржа:", e)
    amount = float(ex.amount_to_precision(sym, t["size"]))
    min_amt = ex.market(sym)["limits"]["amount"]["min"]
    if min_amt and amount < min_amt:
        print(f"Размер {amount} меньше минимума {min_amt}, ордер пропущен")
        return
    order = ex.create_order(sym, "market", side, amount, None, {
        "stopLoss": {"triggerPrice": float(ex.price_to_precision(sym, t["sl"]))},
        "takeProfit": {"triggerPrice": float(ex.price_to_precision(sym, t["tp"]))},
    })
    print("Ордер отправлен:", order.get("id"))

# ------------------------- БУМАЖНАЯ ТОРГОВЛЯ -------------------------
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
    fees = CONFIG["fee"] * (pos["entry"] + exit_price) * pos["size"]
    return gross - fees

def live_position_open(ex) -> bool:
    for p in ex.fetch_positions([CONFIG["symbol"]]):
        if abs(float(p.get("contracts") or 0)) > 0:
            return True
    return False

def run_bot():
    ex = make_exchange()
    balance, pos = CONFIG["paper_balance"], None
    last_signal_ts = None
    print(f"Старт. Режим: {'LIVE' if CONFIG['live'] else 'PAPER'}")
    while True:
        try:
            df = fetch_df(ex)
            closed = df.iloc[:-1]
            closed_ts = closed["ts"].iloc[-1]
            if last_signal_ts == closed_ts:
                time.sleep(CONFIG["loop_seconds"])
                continue
            last_signal_ts = closed_ts
            price = closed["close"].iloc[-1]
            atr = closed["atr"].iloc[-1]
            high = closed["high"].iloc[-1]
            low = closed["low"].iloc[-1]
            macro = macro_score() if CONFIG["use_macro"] else 0
            fund = funding_score(ex) if CONFIG["use_funding"] else 0
            signal, score = get_signal(closed, macro + fund)
            if pos:
                exit_price = check_exit(pos, high, low)
                if exit_price:
                    p = pnl(pos, exit_price)
                    balance += p
                    print(f"Закрыта {pos['side']} по {exit_price:.2f}, PnL={p:.2f}, Баланс={balance:.2f}")
                    pos = None
            elif signal != "none":
                trade = build_trade(signal, price, atr, balance)
                pos = trade
                print(f"Сигнал {signal.upper()} (score={score}), вход={price:.2f}, SL={trade['sl']:.2f}, TP={trade['tp']:.2f}")
                if CONFIG["live"]:
                    place_live_order(ex, trade)
            else:
                print(f"Сигнал: none (score={score}), цена={price:.2f}, баланс={balance:.2f}")
        except Exception as e:
            print("Ошибка:", e)
        time.sleep(CONFIG["loop_seconds"])

if __name__ == "__main__":
    run_bot()