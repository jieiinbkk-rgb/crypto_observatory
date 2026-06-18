import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from config.settings import INITIAL_CAPITAL, STRATEGIES, MAX_POSITIONS

COMMISSION_RATE = 0.0005
POSITION_SIZE   = 0.05
MAX_DRAWDOWN_LIMIT = 10.0
MAX_DAILY_LOSS     = 5.0

IV_SENSITIVITY = {
    "vol_sell":     -1.2,
    "ratio_spread": -0.6,
    "gamma_long":    1.5,
    "tail_hedge":    2.0,
    "calendar_spread": -0.4,
    "butterfly":       -0.8,
}

THETA_DECAY = {
    "vol_sell":     +0.0002,
    "ratio_spread":  0.0001,
    "gamma_long":   -0.0003,
    "tail_hedge":   -0.0004,
    "calendar_spread": +0.0003,
    "butterfly":       +0.0001,
}

@st.cache_resource
def get_portfolio_store():
    return {"positions":[],"closed_trades":[],"capital":INITIAL_CAPITAL,"peak_capital":INITIAL_CAPITAL,"max_drawdown":0.0}

def check_risk_limits(store):
    dd = store.get("max_drawdown", 0)
    if dd >= MAX_DRAWDOWN_LIMIT:
        return False, f"Max DD {dd:.1f}% 超過"
    cap = store["capital"]
    if cap < store["peak_capital"] * (1 - MAX_DAILY_LOSS / 100):
        return False, f"日次損失上限超過"
    return True, "OK"

def kelly_position_size(confidence, opp_score, win_rate=0.55, avg_win=0.15, avg_loss=0.10):
    """
    Kelly基準でポジションサイズを動的計算
    f = (p*b - q) / b
    p = 勝率, q = 負け率, b = 平均勝ち/平均負け比
    """
    p = min(0.8, max(0.3, win_rate * confidence))
    q = 1 - p
    b = avg_win / max(avg_loss, 0.01)
    kelly = (p * b - q) / b
    # ハーフケリー（過剰リスク防止）
    half_kelly = max(0.01, min(0.10, kelly * 0.5))
    # Opportunity Scoreで調整
    score_factor = 0.5 + (opp_score / 200)
    return round(half_kelly * score_factor, 4)

def open_trade(signal, store):
    # Kelly基準でポジションサイズを動的計算
    closed = store["closed_trades"]
    if len(closed) >= 5:
        wins    = [t for t in closed if t["current_pnl_usd"] > 0]
        wr      = len(wins) / len(closed)
        avg_win = float(np.mean([t["current_pnl_pct"] for t in wins])) if wins else 0.15
        losses  = [t for t in closed if t["current_pnl_usd"] <= 0]
        avg_loss= abs(float(np.mean([t["current_pnl_pct"] for t in losses]))) if losses else 0.10
        kelly_size = kelly_position_size(signal.get("confidence",0.6), signal.get("opp_score",50), wr, avg_win, avg_loss)
    else:
        kelly_size = POSITION_SIZE
    size_usd = store["capital"] * kelly_size
    cost     = size_usd * COMMISSION_RATE
    pos = {**signal,"entry_time":signal["timestamp"],"size_usd":round(size_usd,2),"cost_usd":round(cost,2),
           "entry_btc_iv":signal["btc_iv"],"entry_eth_iv":signal["eth_iv"],
           "entry_ratio":signal["btc_iv"]/max(signal["eth_iv"],1),
           "current_pnl_pct":0.0,"current_pnl_usd":-cost,"status":"open","bars_held":0}
    store["positions"].append(pos)
    store["capital"] -= cost

def _calc_pnl(pos, latest):
    strat_key = pos.get("strategy", "vol_sell")
    btc_now = float(latest.get("BTC_IV") or pos["entry_btc_iv"])
    eth_now = float(latest.get("ETH_IV") or pos["entry_eth_iv"])
    vega  = float(latest.get("BTC_Vega")  or 0)
    theta = float(latest.get("BTC_Theta") or 0)
    gamma = float(latest.get("BTC_Gamma") or 0)

    if vega != 0:
        d_iv_btc = btc_now - pos["entry_btc_iv"]
        d_iv_eth = eth_now - pos["entry_eth_iv"]
        if strat_key == "vol_sell":
            vega_pnl  = -vega * d_iv_btc * pos["size_usd"] / 1000
            theta_pnl = abs(theta) * pos["bars_held"] * pos["size_usd"] / 10000
            pnl_usd   = vega_pnl + theta_pnl - pos["cost_usd"]
        elif strat_key == "gamma_long":
            vega_pnl  = vega * abs(d_iv_btc) * pos["size_usd"] / 1000
            gamma_pnl = gamma * (d_iv_btc**2) * pos["size_usd"] / 100
            theta_pnl = theta * pos["bars_held"] * pos["size_usd"] / 10000
            pnl_usd   = vega_pnl + gamma_pnl + theta_pnl - pos["cost_usd"]
        elif strat_key == "ratio_spread":
            ratio_now   = btc_now / max(eth_now, 1)
            ratio_entry = pos.get("entry_ratio", 1.0)
            ratio_chg   = (ratio_now - ratio_entry) / max(ratio_entry, 0.01)
            pnl_usd     = -ratio_chg * pos["size_usd"] * 0.6 - pos["cost_usd"]
        elif strat_key == "tail_hedge":
            vega_pnl  = vega * d_iv_btc * 2.0 * pos["size_usd"] / 1000
            theta_pnl = theta * pos["bars_held"] * pos["size_usd"] / 10000
            pnl_usd   = vega_pnl + theta_pnl - pos["cost_usd"]
        else:
            pnl_usd = 0.0
        pnl_pct = pnl_usd / pos["size_usd"]
        return round(pnl_pct, 6), round(pnl_usd, 2)

    sens      = IV_SENSITIVITY.get(strat_key, 0.5)
    theta_dec = THETA_DECAY.get(strat_key, 0.0)
    btc_chg = (btc_now - pos["entry_btc_iv"]) / max(pos["entry_btc_iv"], 1)
    eth_chg = (eth_now - pos["entry_eth_iv"]) / max(pos["entry_eth_iv"], 1)
    if strat_key == "ratio_spread":
        ratio_now   = btc_now / max(eth_now, 1)
        ratio_entry = pos.get("entry_ratio", 1.0)
        ratio_chg   = (ratio_now - ratio_entry) / max(ratio_entry, 0.01)
        iv_pnl      = -ratio_chg * abs(sens)
    elif strat_key == "gamma_long":
        avg_chg = (abs(btc_chg) + abs(eth_chg)) / 2
        iv_pnl  = avg_chg * sens * (1 + avg_chg * 2)
    else:
        avg_chg = (btc_chg + eth_chg) / 2
        iv_pnl  = avg_chg * sens
    theta_pnl = theta_dec * pos["bars_held"]
    pnl_pct   = iv_pnl + theta_pnl
    pnl_usd   = pos["size_usd"] * pnl_pct - pos["cost_usd"]
    return round(pnl_pct, 6), round(pnl_usd, 2)

def update_positions(store, df):
    if not store["positions"] or df.empty: return
    latest   = df.iloc[-1]
    to_close = []
    for pos in store["positions"]:
        pos["bars_held"] += 1
        pnl_pct, pnl_usd = _calc_pnl(pos, latest)
        pos["current_pnl_pct"] = pnl_pct
        pos["current_pnl_usd"] = pnl_usd
        tp = pos.get("target_pct", 0.01)
        sl = pos.get("stop_pct", -0.015)
        if pnl_pct >= tp: to_close.append((pos, "TP"))
        elif pnl_pct <= sl: to_close.append((pos, "SL"))
        elif pos["bars_held"] > 1440: to_close.append((pos, "TIMEOUT"))
    for pos, reason in to_close: _close(pos, store, reason)

def close_manually(pos, store, df):
    if not df.empty:
        pnl_pct, pnl_usd = _calc_pnl(pos, df.iloc[-1])
        pos["current_pnl_pct"] = pnl_pct
        pos["current_pnl_usd"] = pnl_usd
    _close(pos, store, "MANUAL")

def _notify_close(pos, reason):
    """決済時にTelegram通知"""
    try:
        import streamlit as st
        import requests
        token   = st.secrets.get("TELEGRAM_TOKEN", "")
        chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            return
        emoji = "✅" if reason == "TP" else ("🛑" if reason == "SL" else "⏰")
        pnl   = pos.get("current_pnl_usd", 0)
        pct   = pos.get("current_pnl_pct", 0) * 100
        sign  = "+" if pnl >= 0 else ""
        msg   = (f"{emoji} {reason} 決済\n"
                 f"戦略: {pos.get('strategy_name','?')}\n"
                 f"P&L: {sign}${pnl:.2f} ({sign}{pct:.2f}%)\n"
                 f"状態: {pos.get('state','?')}")
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg},
            timeout=5,
        )
    except Exception:
        pass

def _close(pos, store, reason):
    pos["status"]    = reason
    pos["exit_time"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    store["capital"] += pos["size_usd"] + pos["current_pnl_usd"]
    store["positions"]     = [p for p in store["positions"] if p["id"] != pos["id"]]
    store["closed_trades"].append(pos)
    _notify_close(pos, reason)
    if store["capital"] > store["peak_capital"]: store["peak_capital"] = store["capital"]
    dd = (store["peak_capital"] - store["capital"]) / store["peak_capital"] * 100
    if dd > store["max_drawdown"]: store["max_drawdown"] = round(dd, 2)

def portfolio_stats(store):
    closed = store["closed_trades"]
    if not closed: return {"total_trades":0,"win_rate":0,"total_pnl_usd":0.0,"avg_pnl_pct":0.0,"sharpe":0.0,"max_drawdown":store["max_drawdown"],"profit_factor":0.0}
    pnls     = [t["current_pnl_usd"] for t in closed]
    pct_pnls = [t["current_pnl_pct"] for t in closed]
    wins     = [p for p in pnls if p > 0]
    losses   = [p for p in pnls if p <= 0]
    pf       = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else 99.0
    arr      = np.array(pct_pnls)
    sharpe   = float(arr.mean()/arr.std()*np.sqrt(252)) if arr.std()>0 else 0.0
    return {"total_trades":len(closed),"win_rate":round(len(wins)/len(closed)*100,1),
            "total_pnl_usd":round(sum(pnls),2),"avg_pnl_pct":round(float(np.mean(pct_pnls))*100,3),
            "sharpe":round(sharpe,2),"max_drawdown":store["max_drawdown"],"profit_factor":round(pf,2)}

def equity_curve(store):
    cap=INITIAL_CAPITAL; curve=[cap]
    for t in store["closed_trades"]:
        cap+=t["current_pnl_usd"]; curve.append(round(cap,2))
    return curve
