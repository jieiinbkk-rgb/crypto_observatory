"""
strategy/labeler.py
過去シグナルに正解ラベルを自動付与
教師あり学習の準備
"""
import pandas as pd
import numpy as np

def label_signals(df: pd.DataFrame, signal_df: pd.DataFrame) -> pd.DataFrame:
    """
    シグナル発生後1h・6hのIV変化から正解ラベルを付与
    label=1: 戦略が正しかった（利益方向）
    label=0: 戦略が間違っていた（損失方向）
    """
    if signal_df.empty or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    signal_df = signal_df.copy()
    signal_df["Timestamp"] = pd.to_datetime(signal_df["Timestamp"])

    rows = []
    for _, sig in signal_df.iterrows():
        ts    = sig["Timestamp"]
        strat = str(sig.get("Strategy", ""))
        btc_iv0 = float(sig.get("BTC_IV", 0) or 0)
        if btc_iv0 == 0:
            continue

        # 1時間後のIV
        future_1h = df[df["Timestamp"] >= ts + pd.Timedelta(hours=1)]
        if future_1h.empty:
            continue
        btc_iv_1h = float(future_1h.iloc[0].get("BTC_IV", 0) or 0)
        iv_chg_1h = (btc_iv_1h - btc_iv0) / max(btc_iv0, 1)

        # 戦略方向と正解判定
        if strat == "vol_sell":
            label_1h = 1 if iv_chg_1h < 0 else 0
        elif strat == "gamma_long":
            label_1h = 1 if abs(iv_chg_1h) > 0.02 else 0
        elif strat == "ratio_spread":
            label_1h = 1 if abs(iv_chg_1h) < 0.01 else 0
        elif strat == "tail_hedge":
            label_1h = 1 if iv_chg_1h > 0.03 else 0
        else:
            label_1h = 0

        rows.append({
            "Timestamp":  ts,
            "Strategy":   strat,
            "BTC_IV_0":   btc_iv0,
            "BTC_IV_1h":  btc_iv_1h,
            "IV_Chg_1h":  round(iv_chg_1h * 100, 2),
            "Label_1h":   label_1h,
            "Confidence": sig.get("Confidence", 0),
            "OppScore":   sig.get("OppScore", 0),
        })

    return pd.DataFrame(rows)


def train_simple_classifier(labeled_df: pd.DataFrame):
    """
    ラベル付きデータで簡易分類器を学習
    特徴量: Confidence・OppScore
    目的変数: Label_1h
    """
    if len(labeled_df) < 20:
        return None, "データ不足（最低20件必要）"

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    import numpy as np

    features = ["Confidence", "OppScore"]
    X = labeled_df[features].fillna(0).values
    y = labeled_df["Label_1h"].values

    clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=3)
    
    if len(labeled_df) >= 30:
        scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
        cv_score = round(float(np.mean(scores)) * 100, 1)
    else:
        cv_score = None

    clf.fit(X, y)
    return clf, f"CV精度: {cv_score}%" if cv_score else "学習完了"
