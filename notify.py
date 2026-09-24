import requests

TOKEN = "8858133882:AAE7-Zr45sAsusuAaA9lbQJnxSZxgH_UY94"
CHAT_ID = "2144158162"

def notify(text: str):
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        requests.post(url, data={
            "chat_id": CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
        })
    except Exception as e:
        print(f"Telegram ошибка: {e}")
