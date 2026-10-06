"""
bot_handler.py
Telegram-бот: слушает команды и кнопки, анализирует монеты по запросу.
Работает отдельным процессом от trading_bot_v2.
"""

import sys
import time
from datetime import datetime

import ccxt

from notify import (
    answer_callback_query,
    edit_message_text,
    get_me,
    get_updates,
    send_telegram_message,
)
from trading_bot_v2 import CONFIG, analyze_symbol, build_trade


# =========================================================================
# WHITELIST — кому разрешено пользоваться ботом
# =========================================================================
# Формат: chat_id: "имя"
# Добавь друга сюда, когда он напишет тебе свой chat_id.
ALLOWED_USERS = {
    2144158162: "Ахмед",
    # 1234567890: "Друг",
}


# =========================================================================
# LOG
# =========================================================================
def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [bot_handler] {msg}")


# =========================================================================
# МЕНЮ
# =========================================================================
def build_main_menu() -> list:
    """Кнопки главного меню — по одной монете в ряду."""
    buttons = []
    for sym in CONFIG["SYMBOLS"]:
        short = sym.split("/")[0]  # BTC/USDT:USDT → BTC
        buttons.append([{
            "text": f"📊 {short}",
            "callback_data": f"analyze:{sym}",
        }])
    return buttons


def get_main_menu_text(user_name: str = "") -> str:
    greeting = f"Привет, {user_name}!" if user_name else "Привет!"
    return (
        f"{greeting}\n\n"
        f"Я — бот-аналитик Bybit. Выбери монету — и я проанализирую её прямо сейчас.\n\n"
        f"<b>Что я показываю:</b>\n"
        f"• Направление (LONG или SHORT)\n"
        f"• Точку входа, стоп, тейк\n"
        f"• Размер позиции, риск, потенциал\n"
        f"• R:R и score (из 4)\n\n"
        f"⚠️ Это PAPER-режим. Реальные ордера не отправляются."
    )


def get_analyzing_text(symbol: str) -> str:
    short = symbol.split("/")[0]
    return f"🔄 Анализирую <b>{short}</b>...\n\nПодожди 3-5 секунд."


# =========================================================================
# ФОРМАТИРОВАНИЕ РЕЗУЛЬТАТА
# =========================================================================
def format_analysis(analysis: dict, balance: float) -> tuple:
    """
    Возвращает (текст_сигнала, текст_предупреждения_или_None).
    """
    symbol = analysis["symbol"]
    short = symbol.split("/")[0]
    side = analysis["side"]
    score = analysis["score"]
    passed = analysis["passed"]
    reasons = analysis["reasons"]
    entry = analysis["price"]
    atr = analysis["atr"]

    trade = build_trade(side, entry, atr, balance, symbol)

    emoji = "🟢" if side == "long" else "🔴"
    side_text = side.upper()
    sl = trade["sl"]
    tp = trade["tp"]
    size = trade["size"]
    risk = trade["risk_usd"]
    potential = trade["potential_usd"]
    sl_pct = abs(sl - entry) / entry * 100
    tp_pct = abs(tp - entry) / entry * 100
    rr = potential / risk if risk > 0 else 0

    if passed:
        header = f"{emoji} <b>{side_text} {short}</b>"
    else:
        header = f"{emoji} <b>Склоняюсь к {side_text} {short}</b>  (score {score}/4)"

    reasons_text = "\n".join(f"• {r}" for r in reasons[:4]) if reasons else "• (нет активных условий)"

    signal_msg = (
        f"{header}\n\n"
        f"📊 Цена: ${entry:.6f}\n"
        f"🛑 Стоп: ${sl:.6f}  (-{sl_pct:.2f}%)\n"
        f"🎯 Тейк: ${tp:.6f}  (+{tp_pct:.2f}%)\n\n"
        f"⚖️ Размер: {size:.6f}\n"
        f"💵 Сумма: ${size * entry:.2f}\n\n"
        f"⚠️ Риск: <b>${risk:.2f}</b>\n"
        f"💰 Потенциал: <b>${potential:.2f}</b>\n"
        f"📊 R:R = 1:{rr:.2f}\n\n"
        f"<b>Причины:</b>\n{reasons_text}"
    )

    warning = None
    if not passed:
        warning = (
            f"⚠️ <b>Но я бы подождал</b>\n\n"
            f"Score <b>{score}/4</b> — ниже порога 3. "
            f"Значит, условий для уверенного входа пока недостаточно.\n\n"
            f"<b>Что делать:</b>\n"
            f"• Дождись следующей свечи (1 час)\n"
            f"• Проверь позже ещё раз\n"
            f"• Или посмотри другую монету"
        )

    return signal_msg, warning


# =========================================================================
# ОБРАБОТКА СООБЩЕНИЙ
# =========================================================================
def handle_message(update: dict):
    msg = update.get("message", {})
    chat = msg.get("chat", {})
    chat_id = chat.get("id")
    text = (msg.get("text") or "").strip()
    first_name = chat.get("first_name", "")

    if chat_id not in ALLOWED_USERS:
        send_telegram_message(
            "⛔ Доступ запрещён.",
            chat_id=str(chat_id),
        )
        log(f"Отказано: chat_id={chat_id}")
        return

    if text in ("/start", "/menu", "/help"):
        send_telegram_message(
            get_main_menu_text(first_name),
            chat_id=str(chat_id),
            buttons=build_main_menu(),
        )
        log(f"Меню → {chat_id}")
        return

    send_telegram_message(
        "Выбери монету из меню:\n/menu",
        chat_id=str(chat_id),
    )


# =========================================================================
# ОБРАБОТКА НАЖАТИЙ КНОПОК
# =========================================================================
def handle_callback(update: dict):
    cb = update.get("callback_query", {})
    cb_id = cb.get("id")
    data = cb.get("data", "")
    msg = cb.get("message", {})
    chat = msg.get("chat", {})
    chat_id = chat.get("id")
    message_id = msg.get("message_id")

    if chat_id not in ALLOWED_USERS:
        answer_callback_query(cb_id, "Доступ запрещён")
        return

    if not data.startswith("analyze:"):
        answer_callback_query(cb_id, "Неизвестная команда")
        return

    symbol = data.split(":", 1)[1]

    answer_callback_query(cb_id, "Анализирую...")
    edit_message_text(
        chat_id=str(chat_id),
        message_id=message_id,
        text=get_analyzing_text(symbol),
    )

    try:
        ex = ccxt.bybit({
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        analysis = analyze_symbol(ex, symbol)
    except Exception as e:
        log(f"Ошибка анализа {symbol}: {e}")
        edit_message_text(
            chat_id=str(chat_id),
            message_id=message_id,
            text=f"❌ Ошибка анализа {symbol}: {e}",
            buttons=build_main_menu(),
        )
        return

    if analysis is None:
        edit_message_text(
            chat_id=str(chat_id),
            message_id=message_id,
            text=f"❌ Не удалось проанализировать {symbol}. Недостаточно данных.",
            buttons=build_main_menu(),
        )
        return

    signal_msg, warning = format_analysis(analysis, CONFIG["VIRTUAL_BALANCE_START"])

    edit_message_text(
        chat_id=str(chat_id),
        message_id=message_id,
        text=signal_msg,
        buttons=build_main_menu(),
    )

    if warning:
        send_telegram_message(warning, chat_id=str(chat_id))

    log(f"Анализ {symbol} → {chat_id}: {analysis['side']} score={analysis['score']}")


# =========================================================================
# ГЛАВНЫЙ ЦИКЛ
# =========================================================================
def run_handler():
    me = get_me()
    if not me:
        log("❌ Не удалось получить информацию о боте. Проверь токен.")
        sys.exit(1)

    log(f"✅ Бот @{me.get('username')} слушает команды")
    log(f"Разрешённые: {list(ALLOWED_USERS.values())}")
    log("Ожидание...")

    offset = 0
    while True:
        try:
            updates = get_updates(offset=offset, timeout=25)
            for upd in updates:
                offset = upd["update_id"] + 1

                if "message" in upd:
                    try:
                        handle_message(upd)
                    except Exception as e:
                        log(f"Ошибка message: {e}")

                elif "callback_query" in upd:
                    try:
                        handle_callback(upd)
                    except Exception as e:
                        log(f"Ошибка callback: {e}")

        except KeyboardInterrupt:
            log("Остановка (Ctrl+C)")
            break
        except Exception as e:
            log(f"Главная ошибка: {e}")
            time.sleep(5)


if __name__ == "__main__":
    run_handler()
