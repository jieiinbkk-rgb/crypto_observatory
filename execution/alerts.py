"""
execution/alerts.py
Telegram アラート送信
"""
import streamlit as st
import requests
from datetime import datetime, timedelta

COOLDOWN_MINUTES = 30   # 同一アラートの再送クールダウン


@st.cache_resource
def get_alert_store() -> dict:
    return {
        "last_sent": {},   # {alert_key: datetime}
    }


def is_telegram_configured() -> bool:
    try:
        token = st.secrets.get("TELEGRAM_TOKEN", "")
        chat  = st.secrets.get("TELEGRAM_CHAT_ID", "")
        return bool(token and chat)
    except Exception:
        return False


def _send(msg: str):
    try:
        token   = st.secrets["TELEGRAM_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
            timeout=8,
        )
    except Exception:
        pass


def _cooldown_ok(store: dict, key: str) -> bool:
    last = store["last_sent"].get(key)
    if last is None:
        return True
    return datetime.utcnow() - last > timedelta(minutes=COOLDOWN_MINUTES)


def fire_alerts(
    state_key:  str,
    btc_z:      float,
    eth_z:      float,
    opp_score:  int,
    confidence: float,
    btc_iv:     float,
    eth_iv:     float,
    store:      dict,
):
    """条件に応じて Telegram アラートを送信"""
    if not is_telegram_configured():
        return

    now = datetime.utcnow()

    # パニック警報
    if state_key == "panic" and _cooldown_ok(store, "panic"):
        _send(
            f"🚨 *PANIC ALERT*\n"
            f"BTC IV: {btc_iv:.1f} | ETH IV: {eth_iv:.1f}\n"
            f"BTC Z: {btc_z:.2f} | ETH Z: {eth_z:.2f}\n"
            f"Opp Score: {opp_score}/100 | Conf: {confidence*100:.0f}%\n"
            f"_{now.strftime('%Y-%m-%d %H:%M UTC')}_"
        )
        store["last_sent"]["panic"] = now

    # 高 Opportunity Score
    if opp_score >= 70 and _cooldown_ok(store, "high_opp"):
        _send(
            f"⚡ *HIGH OPPORTUNITY* Score={opp_score}/100\n"
            f"State: {state_key} | Conf: {confidence*100:.0f}%\n"
            f"_{now.strftime('%Y-%m-%d %H:%M UTC')}_"
        )
        store["last_sent"]["high_opp"] = now

    # 異常 Z スコア
    if abs(btc_z) > 3.0 and _cooldown_ok(store, "zscore"):
        _send(
            f"⚠️ *BTC Z-Score Spike* = {btc_z:.2f}\n"
            f"BTC IV: {btc_iv:.1f}\n"
            f"_{now.strftime('%Y-%m-%d %H:%M UTC')}_"
        )
        store["last_sent"]["zscore"] = now
