"""
data/features.py
特徴量エンジニアリング・Z スコア・RV 計算・異常検知
"""
import pandas as pd
import numpy as np


def _rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    mu  = series.rolling(window, min_periods=10).mean()
    std = series.rolling(window, min_periods=10).std().replace(0, np.nan)
    return (series - mu) / std


def _realized_vol(price: pd.Series, window: int) -> pd.Series:
    """単純なローリング標準偏差ベースの RV（年率換算 ×√(525600/window)）"""
    ret = price.pct_change()
    rv  = ret.rolling(window, min_periods=max(5, window // 2)).std()
    return (rv * np.sqrt(525600 / window) * 100).round(2)   # 年率 %


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    入力: Timestamp, BTC_IV, ETH_IV, BTC_Spot（オプション）, ETH_Spot（オプション）
    出力: 各種特徴量を追加した DataFrame
    """
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # 数値変換
    for col in ["BTC_IV", "ETH_IV", "BTC_Spot", "ETH_Spot"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── IV 比率 ──────────────────────────────────────────────
    df["BTC_ETH_Ratio"] = (df["BTC_IV"] / df["ETH_IV"].replace(0, np.nan)).round(4)

    # ── Z スコア ─────────────────────────────────────────────
    df["BTC_Z"]   = _rolling_zscore(df["BTC_IV"],        60).round(3)
    df["ETH_Z"]   = _rolling_zscore(df["ETH_IV"],        60).round(3)
    df["Ratio_Z"] = _rolling_zscore(df["BTC_ETH_Ratio"], 60).round(3)

    # ── Realized Vol ─────────────────────────────────────────
    for sym in ["BTC", "ETH"]:
        spot_col = f"{sym}_Spot"
        if spot_col in df.columns:
            for w in [10, 20, 60]:
                df[f"{sym}_RV{w}"] = _realized_vol(df[spot_col], w)
        else:
            # スポット価格がなければ RV を IV の割引で近似
            for w in [10, 20, 60]:
                df[f"{sym}_RV{w}"] = (df[f"{sym}_IV"] * 0.75).round(2)

    # ── IV-RV スプレッド ─────────────────────────────────────
    df["BTC_Spread20"] = (df["BTC_IV"] - df["BTC_RV20"]).round(2)
    df["BTC_Spread60"] = (df["BTC_IV"] - df["BTC_RV60"]).round(2)
    df["ETH_Spread20"] = (df["ETH_IV"] - df["ETH_RV20"]).round(2)
    df["ETH_Spread60"] = (df["ETH_IV"] - df["ETH_RV60"]).round(2)

    # ── 異常検知（Z スコア閾値 ± 2.5） ───────────────────────
    THRESH = 2.5
    reasons = []
    for i, row in df.iterrows():
        r = []
        if abs(row.get("BTC_Z", 0) or 0) > THRESH:
            r.append(f"BTC_Z={row['BTC_Z']:.2f}")
        if abs(row.get("ETH_Z", 0) or 0) > THRESH:
            r.append(f"ETH_Z={row['ETH_Z']:.2f}")
        if abs(row.get("Ratio_Z", 0) or 0) > THRESH:
            r.append(f"Ratio_Z={row['Ratio_Z']:.2f}")
        reasons.append(", ".join(r) if r else "")

    df["Anomaly_Reason"] = reasons
    df["Anomaly"]        = df["Anomaly_Reason"].str.len() > 0

    return df
