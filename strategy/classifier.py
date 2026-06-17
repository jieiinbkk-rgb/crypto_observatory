"""
strategy/classifier.py
GMM による市場状態分類・Vol Regime・遷移行列・Opportunity Score
"""
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from config.settings import MARKET_STATES, VOL_REGIMES


# ── Classifier Store（セッションキャッシュ） ──────────────────
@st.cache_resource
def get_classifier_store() -> dict:
    return {
        "gmm_model":     None,
        "gmm_scaler":    None,
        "gmm_label_map": {},
        "state_history": [],      # [{"Timestamp","State","Confidence","Method"}]
    }


def fit_gmm(store: dict, df: pd.DataFrame):
    """Z スコアの直近 window 行で GMM を学習"""
    feats = ["BTC_Z", "ETH_Z", "Ratio_Z"]
    sub   = df[feats].dropna()
    if len(sub) < 20:
        return

    X      = sub.values
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=4, covariance_type="full",
                          random_state=42, max_iter=200)
    gmm.fit(Xs)

    # クラスタ → 状態ラベルのマッピング（means の特徴で決定）
    means   = scaler.inverse_transform(gmm.means_)
    label_map = {}
    for i, m in enumerate(means):
        btc_z, eth_z, ratio_z = m[0], m[1], m[2]
        avg_z = (abs(btc_z) + abs(eth_z)) / 2
        if avg_z > 2.0:
            label_map[i] = "panic"
        elif btc_z < -0.5 and eth_z < -0.5:
            label_map[i] = "risk_on"
        elif ratio_z > 0.5:
            label_map[i] = "hedging"
        elif avg_z < 0.3:
            label_map[i] = "squeeze"
        else:
            label_map[i] = "risk_on"

    store["gmm_model"]     = gmm
    store["gmm_scaler"]    = scaler
    store["gmm_label_map"] = label_map


def classify_state(store: dict, df: pd.DataFrame) -> tuple[str, float, str]:
    """最新行を GMM / ルールベースで分類。(state_key, confidence, method) を返す"""
    feats = ["BTC_Z", "ETH_Z", "Ratio_Z"]
    latest = df[feats].dropna().tail(1)
    if latest.empty:
        return "unknown", 0.0, "no_data"

    row = latest.iloc[0]
    btc_z, eth_z, ratio_z = float(row["BTC_Z"]), float(row["ETH_Z"]), float(row["Ratio_Z"])

    # GMM 分類
    if store.get("gmm_model"):
        try:
            scaler = store["gmm_scaler"]
            gmm    = store["gmm_model"]
            lmap   = store["gmm_label_map"]
            Xs     = scaler.transform([[btc_z, eth_z, ratio_z]])
            proba  = gmm.predict_proba(Xs)[0]
            cluster   = int(np.argmax(proba))
            confidence = float(proba[cluster])
            state_key  = lmap.get(cluster, "unknown")
            return state_key, confidence, "GMM"
        except Exception:
            pass

    # ルールベース フォールバック
    avg_z = (abs(btc_z) + abs(eth_z)) / 2
    if avg_z > 2.0:
        return "panic",   0.8, "rule"
    if btc_z < -0.5 and eth_z < -0.5:
        return "risk_on", 0.7, "rule"
    if ratio_z > 0.8:
        return "hedging", 0.65, "rule"
    if avg_z < 0.3:
        return "squeeze", 0.6,  "rule"
    return "risk_on", 0.5, "rule"


def record_state(store: dict, state_key: str, confidence: float, method: str):
    """状態履歴に追記（最大 500 件）"""
    store["state_history"].append({
        "Timestamp":  datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        "State":      state_key,
        "Confidence": round(confidence, 4),
        "Method":     method,
    })
    if len(store["state_history"]) > 500:
        store["state_history"] = store["state_history"][-500:]


def get_state_history_df(store: dict) -> pd.DataFrame:
    if not store["state_history"]:
        return pd.DataFrame()
    df = pd.DataFrame(store["state_history"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


# ── Vol Regime ─────────────────────────────────────────────────
def classify_vol_regime(df: pd.DataFrame) -> tuple[str, str, float]:
    """
    過去データにおける BTC IV のパーセンタイル順位で Regime を判定
    戻り値: (regime_key, label, percentile)
    """
    iv_series = df["BTC_IV"].dropna()
    if len(iv_series) < 10:
        return "normal", VOL_REGIMES["normal"]["label"], 50.0

    current = float(iv_series.iloc[-1])
    pct     = float((iv_series < current).mean() * 100)

    for key, info in VOL_REGIMES.items():
        lo, hi = info["pct_range"]
        if lo <= pct < hi:
            return key, info["label"], pct

    return "very_high", VOL_REGIMES["very_high"]["label"], pct


# ── 遷移行列 ──────────────────────────────────────────────────
def compute_transition_matrix(store: dict) -> pd.DataFrame | None:
    hist = store["state_history"]
    if len(hist) < 10:
        return None

    states  = [h["State"] for h in hist]
    labels  = [s for s in MARKET_STATES if s != "unknown"]
    matrix  = pd.DataFrame(0, index=labels, columns=labels)

    for i in range(len(states) - 1):
        s_from = states[i]
        s_to   = states[i + 1]
        if s_from in matrix.index and s_to in matrix.columns:
            matrix.loc[s_from, s_to] += 1

    # 行を確率に変換
    row_sums = matrix.sum(axis=1).replace(0, np.nan)
    return (matrix.div(row_sums, axis=0) * 100).round(1).fillna(0)


# ── Opportunity Score ─────────────────────────────────────────
def compute_opportunity_score(
    df: pd.DataFrame,
    state_key: str,
    confidence: float,
) -> tuple[int, list[str]]:
    """
    0-100 のスコアと理由リストを返す
    """
    score   = 0
    reasons = []

    # 1. 信頼度 (最大 30 点)
    conf_pts = int(confidence * 30)
    score   += conf_pts
    reasons.append(f"GMM信頼度 {confidence*100:.0f}% → +{conf_pts}pt")

    # 2. 状態ボーナス (最大 25 点)
    state_bonus = {"panic": 25, "squeeze": 20, "hedging": 15, "risk_on": 10, "unknown": 0}
    sb = state_bonus.get(state_key, 0)
    score   += sb
    reasons.append(f"状態 [{state_key}] → +{sb}pt")

    # 3. IV-RV スプレッド (最大 25 点)
    sp20 = df["BTC_Spread20"].dropna()
    if not sp20.empty:
        sp = abs(float(sp20.iloc[-1]))
        pts = min(25, int(sp / 2))
        score += pts
        reasons.append(f"IV-RVスプレッド {sp:.1f} → +{pts}pt")

    # 4. Z スコア強度 (最大 20 点)
    btc_z = df["BTC_Z"].dropna()
    if not btc_z.empty:
        z   = abs(float(btc_z.iloc[-1]))
        pts = min(20, int(z * 5))
        score += pts
        reasons.append(f"|BTC Z| {z:.2f} → +{pts}pt")

    return min(100, score), reasons
