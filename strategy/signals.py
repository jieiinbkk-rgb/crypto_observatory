"""
strategy/signals.py  v2
- 遷移確率をシグナル条件に活用
- Opportunity Scoreをトレード条件に統合
- 信頼度閾値を動的に調整
"""
import uuid
from datetime import datetime
import pandas as pd
from config.settings import MARKET_STATES, STRATEGIES, MAX_POSITIONS

MIN_OPP_SCORE  = 35
MIN_CONFIDENCE = 0.55

def generate_signal(state_key, confidence, df, positions, opp_score, clf_store=None):
    if state_key == "unknown": return None
    if confidence < MIN_CONFIDENCE: return None
    if opp_score < MIN_OPP_SCORE: return None
    if len(positions) >= MAX_POSITIONS: return None

    existing_strategies = {p["strategy"] for p in positions}
    state_info = MARKET_STATES[state_key]
    strat_key  = state_info.get("strategy")
    if not strat_key or strat_key not in STRATEGIES: return None
    if strat_key in existing_strategies: return None

    # 遷移確率ボーナス（次の状態が同じ状態である確率が高いほど強いシグナル）
    transition_bonus = 0.0
    if clf_store:
        try:
            from strategy.classifier import get_next_state_probability
            next_probs = get_next_state_probability(clf_store, state_key)
            stay_prob  = next_probs.get(state_key, 0) / 100
            transition_bonus = stay_prob * 0.1
        except Exception:
            pass

    adjusted_confidence = min(1.0, confidence + transition_bonus)

    strat  = STRATEGIES[strat_key]
    latest = df.iloc[-1]

    return {
        "id":              str(uuid.uuid4())[:8].upper(),
        "timestamp":       datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "state":           state_key,
        "confidence":      round(adjusted_confidence, 4),
        "strategy":        strat_key,
        "strategy_name":   strat["name"],
        "action":          strat["action"],
        "legs":            strat["legs"],
        "target_pct":      strat["target_pnl_pct"],
        "stop_pct":        strat["stop_pnl_pct"],
        "btc_iv":          float(latest.get("BTC_IV") or 0),
        "eth_iv":          float(latest.get("ETH_IV") or 0),
        "opp_score":       opp_score,
        "transition_bonus": round(transition_bonus, 4),
    }
