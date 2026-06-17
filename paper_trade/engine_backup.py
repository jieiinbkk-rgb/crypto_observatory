"""
paper_trade/engine.py
ペーパートレード エンジン
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from config.settings import INITIAL_CAPITAL, STRATEGIES

COMMISSION_RATE = 0.0005   # 0.05% / 片道
POSITION_SIZE   = 0.10     # 元本の 10% / ポジション


@st.cache_resource
def get_portfolio_store() -> dict:
    return {
        "positions":     [],
        "closed_trades": [],
        "capital":       INITIAL_CAPITAL,
        "peak_capital":  INITIAL_CAPITAL,
        "max_drawdown":  0.0,
    }


def open_trade(signal: dict, store: dict):
    """シグナルからポジションを開く"""
    size_usd  = store["capital"] * POSITION_SIZE
    cost      = size_usd * COMMISSION_RATE

    pos = {
        **signal,
        "entry_time":      signal["timestamp"],
        "size_usd":        round(size_usd, 2),
        "cost_usd":        round(cost, 2),
        "entry_btc_iv":    signal["btc_iv"],
        "entry_eth_iv":    signal["eth_iv"],
        "current_pnl_pct": 0.0,
        "current_pnl_usd": -cost,   # 初期コスト
        "status":          "open",
        "bars_held":       0,
    }
    store["positions"].append(pos)
    store["capital"] -= cost


def _calc_pnl(pos: dict, latest: pd.Series) -> tuple[float, float]:
    """ポジションの現在 PnL を計算（IV 変化ベースの簡易モデル）"""
    strat_key = pos.get("strategy", "vol_sell")
    strat     = STRATEGIES.get(strat_key, {})
    action    = strat.get("action", "")

    btc_now = float(latest.get("BTC_IV") or pos["entry_btc_iv"])
    eth_now = float(latest.get("ETH_IV") or pos["entry_eth_iv"])

    btc_chg = (btc_now - pos["entry_btc_iv"]) / max(pos["entry_btc_iv"], 1)
    eth_chg = (eth_now - pos["entry_eth_iv"]) / max(pos["entry_eth_iv"], 1)
    avg_chg = (btc_chg + eth_chg) / 2

    # 戦略方向による損益
    if "SHORT" in action or "SELL" in action:
        pnl_pct = -avg_chg * 0.5   # ボル低下で利益
    elif "LONG" in action or "BUY" in action:
        pnl_pct = avg_chg * 0.5
    else:
        pnl_pct = avg_chg * 0.3

    pnl_usd = pos["size_usd"] * pnl_pct - pos["cost_usd"]
    return round(pnl_pct, 6), round(pnl_usd, 2)


def update_positions(store: dict, df: pd.DataFrame):
    """全オープンポジションを更新し、TP/SL チェック"""
    if not store["positions"] or df.empty:
        return

    latest    = df.iloc[-1]
    to_close  = []

    for pos in store["positions"]:
        pos["bars_held"] += 1
        pnl_pct, pnl_usd  = _calc_pnl(pos, latest)
        pos["current_pnl_pct"] = pnl_pct
        pos["current_pnl_usd"] = pnl_usd

        strat = STRATEGIES.get(pos.get("strategy", ""), {})
        tp    = strat.get("target_pnl_pct", 0.15)
        sl    = strat.get("stop_pnl_pct",   -0.25)

        if pnl_pct >= tp:
            to_close.append((pos, "TP"))
        elif pnl_pct <= sl:
            to_close.append((pos, "SL"))
        elif pos["bars_held"] > 1440:   # 24h タイムアウト
            to_close.append((pos, "TIMEOUT"))

    for pos, reason in to_close:
        _close(pos, store, reason)


def close_manually(pos: dict, store: dict, df: pd.DataFrame):
    """手動決済"""
    if not df.empty:
        pnl_pct, pnl_usd = _calc_pnl(pos, df.iloc[-1])
        pos["current_pnl_pct"] = pnl_pct
        pos["current_pnl_usd"] = pnl_usd
    _close(pos, store, "MANUAL")


def _close(pos: dict, store: dict, reason: str):
    """ポジションをクローズして PnL 確定"""
    pos["status"]    = reason
    pos["exit_time"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    store["capital"] += pos["size_usd"] + pos["current_pnl_usd"]
    store["positions"]     = [p for p in store["positions"] if p["id"] != pos["id"]]
    store["closed_trades"].append(pos)

    # ドローダウン更新
    if store["capital"] > store["peak_capital"]:
        store["peak_capital"] = store["capital"]
    dd = (store["peak_capital"] - store["capital"]) / store["peak_capital"] * 100
    if dd > store["max_drawdown"]:
        store["max_drawdown"] = round(dd, 2)


def portfolio_stats(store: dict) -> dict:
    """統計サマリーを返す"""
    closed = store["closed_trades"]
    if not closed:
        return {
            "total_trades": 0, "win_rate": 0,
            "total_pnl_usd": 0.0, "avg_pnl_pct": 0.0,
            "sharpe": 0.0, "max_drawdown": store["max_drawdown"],
            "profit_factor": 0.0,
        }

    pnls    = [t["current_pnl_usd"] for t in closed]
    pct_pnls= [t["current_pnl_pct"] for t in closed]
    wins    = [p for p in pnls if p > 0]
    losses  = [p for p in pnls if p <= 0]

    pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")

    arr = np.array(pct_pnls)
    sharpe = float(arr.mean() / arr.std()) * np.sqrt(252) if arr.std() > 0 else 0.0

    return {
        "total_trades":  len(closed),
        "win_rate":      round(len(wins) / len(closed) * 100, 1),
        "total_pnl_usd": round(sum(pnls), 2),
        "avg_pnl_pct":   round(np.mean(pct_pnls) * 100, 3),
        "sharpe":        round(sharpe, 2),
        "max_drawdown":  store["max_drawdown"],
        "profit_factor": round(pf, 2) if pf != float("inf") else 99.0,
    }


def equity_curve(store: dict) -> list[float]:
    cap = INITIAL_CAPITAL
    curve = [cap]
    for t in store["closed_trades"]:
        cap += t["current_pnl_usd"]
        curve.append(round(cap, 2))
    return curve
