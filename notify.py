"""
notify.py
Отправка уведомлений в Telegram.
"""

import requests
from datetime import datetime

TELEGRAM_TOKEN = "8858133882:AAE7-Zr45sAsusuAaA9lbQJnxSZxgH_UY94"
TELEGRAM_CHAT_ID = "2144158162"

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"


def _log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [notify] {msg}")


def send_telegram_message(text: str, parse_mode: str = "HTML") -> bool:
    """Отправляет сообщение в Telegram. Возвращает True при успехе."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        _log("TELEGRAM_TOKEN/CHAT_ID не заданы. Вывод в консоль:")
        print(text)
        return False

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }

    try:
        resp = requests.post(TELEGRAM_API_URL, data=payload, timeout=10)
        if resp.status_code == 200:
            return True
        _log(f"Ошибка Telegram: HTTP {resp.status_code} | {resp.text}")
        return False
    except Exception as e:
        _log(f"Исключение при отправке: {e}")
        return False
