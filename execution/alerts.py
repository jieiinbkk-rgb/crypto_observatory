"""
execution/alerts.py  v2
- アラート条件を改善
- ポートフォリオリスクアラートを追加
"""
import streamlit as st
import requests
from datetime import datetime, timedelta

COOLDOWN_MINUTES = 30

@st.cache_resource
def get_alert_store():
    return {"last_sent": {}}

def is_telegram_configured():
    try:
        token = st.secrets.get("TELEGRAM_TOKEN", "")
        chat  = st.secrets.get("TELEGRAM_CHAT_ID", "")
        return bool(token and chat)
    except Exception:
        return False

def _send(msg):
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

def _cooldown_ok(store, key):
    last = store["last_sent"].get(key)
    if last is None: return True
    return datetime.utcnow() - last > timedelta(minutes=COOLDOWN_MINUTES)

def fire_alerts(state_key, btc_z, eth_z, opp_score, confidence, btc_iv, eth_iv, store):
    if not is_telegram_configured(): return
    now = datetime.utcnow()

    if state_key == "panic" and _cooldown_ok(store, "panic"):
        _send(f"🚨 *PANIC ALERT*
BTC IV: {btc_iv:.1f} | ETH IV: {eth_iv:.1f}
BTC Z: {btc_z:.2f} | ETH Z: {eth_z:.2f}
Opp Score: {opp_score}/100 | Conf: {confidence*100:.0f}%
_{now.strftime('%Y-%m-%d %H:%M UTC')}_")
        store["last_sent"]["panic"] = now

    if state_key == "squeeze" and _cooldown_ok(store, "squeeze"):
        _send(f"🔵 *VOL SQUEEZE ALERT*
BTC IV: {btc_iv:.1f} | Conf: {confidence*100:.0f}%
Opp Score: {opp_score}/100
_{now.strftime('%Y-%m-%d %H:%M UTC')}_")
        store["last_sent"]["squeeze"] = now

    if opp_score >= 70 and _cooldown_ok(store, "high_opp"):
        _send(f"⚡ *HIGH OPPORTUNITY* Score={opp_score}/100
State: {state_key} | Conf: {confidence*100:.0f}%
_{now.strftime('%Y-%m-%d %H:%M UTC')}_")
        store["last_sent"]["high_opp"] = now

    if abs(btc_z) > 3.0 and _cooldown_ok(store, "zscore"):
        _send(f"⚠️ *BTC Z-Score Spike* = {btc_z:.2f}
BTC IV: {btc_iv:.1f}
_{now.strftime('%Y-%m-%d %H:%M UTC')}_")
        store["last_sent"]["zscore"] = now
