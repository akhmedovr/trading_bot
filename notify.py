"""
notify.py
Telegram-модуль: отправка сообщений, кнопки, polling.
Поддерживает отправку нескольким пользователям (chat_id передаётся явно).
"""

import json
import time
from datetime import datetime
from typing import Optional

import requests


# =========================================================================
# НАСТРОЙКИ
# =========================================================================
TELEGRAM_TOKEN = "8800264793:AAE3GRVALr1I3XmQAkaqsladdRqtxlDDAZQ"

# Твой chat_id по умолчанию (для уведомлений от торгового бота)
DEFAULT_CHAT_ID = "2144158162"

API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


def _log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [notify] {msg}")


# =========================================================================
# ОТПРАВКА СООБЩЕНИЙ
# =========================================================================
def send_telegram_message(
    text: str,
    chat_id: Optional[str] = None,
    parse_mode: str = "HTML",
    buttons: Optional[list] = None,
) -> Optional[dict]:
    """
    Отправляет сообщение в Telegram.
    chat_id — если None, используется DEFAULT_CHAT_ID.
    buttons — список рядов кнопок:
        [
            [{"text": "BTC", "callback_data": "analyze:BTC/USDT:USDT"}],
            [{"text": "ETH", "callback_data": "analyze:ETH/USDT:USDT"}],
        ]
    Возвращает result-объект (содержит message_id).
    """
    if not TELEGRAM_TOKEN:
        _log("Токен не задан, вывод в консоль:")
        print(text)
        return None

    target_chat_id = chat_id or DEFAULT_CHAT_ID

    payload = {
        "chat_id": target_chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }

    if buttons:
        payload["reply_markup"] = json.dumps({"inline_keyboard": buttons})

    try:
        resp = requests.post(f"{API_URL}/sendMessage", data=payload, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("result", {})
        _log(f"sendMessage: HTTP {resp.status_code} | {resp.text}")
        return None
    except Exception as e:
        _log(f"sendMessage исключение: {e}")
        return None


def edit_message_text(
    chat_id: str,
    message_id: int,
    text: str,
    parse_mode: str = "HTML",
    buttons: Optional[list] = None,
) -> bool:
    """Редактирует существующее сообщение."""
    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }

    if buttons is not None:
        payload["reply_markup"] = json.dumps({"inline_keyboard": buttons})
    else:
        payload["reply_markup"] = json.dumps({"inline_keyboard": []})

    try:
        resp = requests.post(f"{API_URL}/editMessageText", data=payload, timeout=10)
        if resp.status_code == 200:
            return True
        _log(f"editMessageText: HTTP {resp.status_code} | {resp.text}")
        return False
    except Exception as e:
        _log(f"editMessageText исключение: {e}")
        return False


def answer_callback_query(callback_id: str, text: Optional[str] = None) -> bool:
    """Убирает 'часики' с кнопки, опционально показывает всплывающий текст."""
    payload = {"callback_query_id": callback_id}
    if text:
        payload["text"] = text

    try:
        resp = requests.post(f"{API_URL}/answerCallbackQuery", data=payload, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        _log(f"answerCallbackQuery исключение: {e}")
        return False


# =========================================================================
# ПОЛУЧЕНИЕ ОБНОВЛЕНИЙ (POLLING)
# =========================================================================
def get_updates(offset: int = 0, timeout: int = 25) -> list:
    """
    Long polling: если новых сообщений нет — Telegram держит соединение
    до 25 секунд и возвращает пустой список.
    """
    params = {
        "offset": offset,
        "timeout": timeout,
        "allowed_updates": json.dumps(["message", "callback_query"]),
    }
    try:
        resp = requests.get(
            f"{API_URL}/getUpdates",
            params=params,
            timeout=timeout + 10,
        )
        if resp.status_code == 200:
            return resp.json().get("result", [])
        _log(f"getUpdates: HTTP {resp.status_code} | {resp.text}")
        return []
    except requests.exceptions.ReadTimeout:
        return []
    except Exception as e:
        _log(f"getUpdates исключение: {e}")
        time.sleep(5)
        return []


def get_me() -> Optional[dict]:
    """Проверка токена — возвращает инфу о боте."""
    try:
        resp = requests.get(f"{API_URL}/getMe", timeout=10)
        if resp.status_code == 200:
            return resp.json().get("result")
        return None
    except Exception:
        return None


__all__ = [
    "send_telegram_message",
    "edit_message_text",
    "answer_callback_query",
    "get_updates",
    "get_me",
    "TELEGRAM_TOKEN",
    "DEFAULT_CHAT_ID",
]
