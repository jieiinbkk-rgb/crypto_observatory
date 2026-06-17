"""
strategy/signals.py
市場状態・スコアに基づくシグナル生成
"""
import uuid
from datetime import datetime
import pandas as pd

from config.settings import MARKET_STATES, STRATEGIES, MAX_POSITIONS

# No-trade ゾーン（スコアが低すぎる場合は発火しない）
MIN_OPP_SCORE  = 30
MIN_CONFIDENCE = 0.50


def generate_signal(
    state_key:   str,
    confidence:  float,
    df:          pd.DataFrame,
    positions:   list,
    opp_score:   int,
) -> dict | None:
    """
    条件を満たす場合にシグナル辞書を返す。
    返り値 None = ノーシグナル
    """
    # No-trade フィルター
    if state_key == "unknown":
        return None
    if confidence < MIN_CONFIDENCE:
        return None
    if opp_score < MIN_OPP_SCORE:
        return None
    if len(positions) >= MAX_POSITIONS:
        return None

    # 同一状態のポジションが既にある場合はスキップ
    existing_states = {p["state"] for p in positions}
    if state_key in existing_states:
        return None

    state_info = MARKET_STATES[state_key]
    strat_key  = state_info.get("strategy")
    if not strat_key or strat_key not in STRATEGIES:
        return None

    strat   = STRATEGIES[strat_key]
    latest  = df.iloc[-1]

    return {
        "id":         str(uuid.uuid4())[:8].upper(),
        "timestamp":  datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "state":      state_key,
        "confidence": round(confidence, 4),
        "strategy":   strat_key,
        "strategy_name": strat["name"],
        "action":     strat["action"],
        "legs":       strat["legs"],
        "target_pct": strat["target_pnl_pct"],
        "stop_pct":   strat["stop_pnl_pct"],
        "btc_iv":     float(latest.get("BTC_IV") or 0),
        "eth_iv":     float(latest.get("ETH_IV") or 0),
        "opp_score":  opp_score,
    }
