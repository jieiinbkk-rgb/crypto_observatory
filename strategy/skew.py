"""
strategy/skew.py
IVスキュー分析
- Put/Call IVの差からスキューを計算
- 市場の方向感・ヘッジ需要を把握
"""
import requests
import numpy as np
import pandas as pd


def get_skew_data(symbol: str = "BTC") -> dict:
    """
    Deribitから直近限月のPut/Call IVスキューを取得
    Returns: {"atm_iv", "put_skew", "call_skew", "risk_reversal", "butterfly"}
    """
    url = f"https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency={symbol}&kind=option"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        if r.status_code != 200:
            return {}

        results = r.json().get("result", [])
        if not results:
            return {}

        # スポット価格取得
        spot = None
        for item in results:
            if item.get("underlying_price"):
                spot = float(item["underlying_price"])
                break
        if not spot:
            return {}

        # 直近限月のオプションだけ抽出
        from datetime import datetime
        expiries = set()
        for item in results:
            parts = item.get("instrument_name", "").split("-")
            if len(parts) >= 3:
                expiries.add(parts[1])
        if not expiries:
            return {}

        # 最も近い限月
        nearest = sorted(expiries)[0]

        calls = [x for x in results if f"-{nearest}-" in x.get("instrument_name","") and x["instrument_name"].endswith("-C") and x.get("mark_iv")]
        puts  = [x for x in results if f"-{nearest}-" in x.get("instrument_name","") and x["instrument_name"].endswith("-P") and x.get("mark_iv")]

        def get_strike(name):
            try: return float(name.split("-")[2])
            except: return 0

        # ATM（スポットに最も近い）
        atm_strike = min([get_strike(x["instrument_name"]) for x in calls], key=lambda s: abs(s - spot))

        # 25デルタ相当のOTM（ATMから±10%）
        otm_call_strike = atm_strike * 1.10
        otm_put_strike  = atm_strike * 0.90

        def nearest_iv(options, target_strike):
            if not options: return None
            nearest = min(options, key=lambda x: abs(get_strike(x["instrument_name"]) - target_strike))
            return float(nearest.get("mark_iv", 0))

        atm_call_iv = nearest_iv(calls, atm_strike)
        atm_put_iv  = nearest_iv(puts,  atm_strike)
        otm_call_iv = nearest_iv(calls, otm_call_strike)
        otm_put_iv  = nearest_iv(puts,  otm_put_strike)

        if not all([atm_call_iv, atm_put_iv, otm_call_iv, otm_put_iv]):
            return {}

        atm_iv       = (atm_call_iv + atm_put_iv) / 2
        risk_reversal = otm_call_iv - otm_put_iv   # 正=コール高い=上昇期待
        butterfly     = (otm_call_iv + otm_put_iv) / 2 - atm_iv  # 正=テール需要高い

        return {
            "atm_iv":        round(atm_iv, 2),
            "otm_call_iv":   round(otm_call_iv, 2),
            "otm_put_iv":    round(otm_put_iv, 2),
            "risk_reversal": round(risk_reversal, 2),
            "butterfly":     round(butterfly, 2),
            "spot":          round(spot, 2),
            "expiry":        nearest,
        }
    except Exception:
        return {}
