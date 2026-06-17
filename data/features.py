"""
data/features.py  v2
"""
import pandas as pd
import numpy as np

def _rolling_zscore(series, window=60):
    mu  = series.rolling(window, min_periods=10).mean()
    std = series.rolling(window, min_periods=10).std().replace(0, float("nan"))
    return (series - mu) / std

def _realized_vol_dvol_scale(iv_series, window):
    log_chg = np.log(iv_series / iv_series.shift(1))
    rv = log_chg.rolling(window, min_periods=max(5, window // 3)).std()
    return (rv * np.sqrt(525600) * iv_series).round(2)

def compute_features(df):
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    for col in ["BTC_IV","ETH_IV","BTC_Spot","ETH_Spot"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["BTC_ETH_Ratio"] = (df["BTC_IV"] / df["ETH_IV"].replace(0, float("nan"))).round(4)
    df["BTC_Z"]   = _rolling_zscore(df["BTC_IV"], 60).round(3)
    df["ETH_Z"]   = _rolling_zscore(df["ETH_IV"], 60).round(3)
    df["Ratio_Z"] = _rolling_zscore(df["BTC_ETH_Ratio"], 60).round(3)
    df["BTC_Z20"] = _rolling_zscore(df["BTC_IV"], 20).round(3)
    df["ETH_Z20"] = _rolling_zscore(df["ETH_IV"], 20).round(3)
    df["BTC_Mom5"]  = df["BTC_IV"].pct_change(5).round(4)
    df["ETH_Mom5"]  = df["ETH_IV"].pct_change(5).round(4)
    df["BTC_Mom20"] = df["BTC_IV"].pct_change(20).round(4)
    df["ETH_Mom20"] = df["ETH_IV"].pct_change(20).round(4)
    for sym in ["BTC","ETH"]:
        for w in [10,20,60]:
            df[f"{sym}_RV{w}"] = _realized_vol_dvol_scale(df[f"{sym}_IV"], w)
    df["BTC_Spread20"] = (df["BTC_IV"] - df["BTC_RV20"]).round(2)
    df["BTC_Spread60"] = (df["BTC_IV"] - df["BTC_RV60"]).round(2)
    df["ETH_Spread20"] = (df["ETH_IV"] - df["ETH_RV20"]).round(2)
    df["ETH_Spread60"] = (df["ETH_IV"] - df["ETH_RV60"]).round(2)
    df["BTC_Accel"] = df["BTC_Mom5"].diff(5).round(4)
    df["ETH_Accel"] = df["ETH_Mom5"].diff(5).round(4)
    df["IV_Divergence"] = ((df["BTC_IV"]-df["ETH_IV"])/((df["BTC_IV"]+df["ETH_IV"])/2).replace(0,float("nan"))).round(4)
    df["IV_Div_Z"] = _rolling_zscore(df["IV_Divergence"], 60).round(3)
    r20 = df["BTC_IV"].rolling(20).max()-df["BTC_IV"].rolling(20).min()
    r60 = df["BTC_IV"].rolling(60).max()-df["BTC_IV"].rolling(60).min()
    df["Vol_Compression"] = (r20/r60.replace(0,float("nan"))).round(3)
    THRESH=2.5
    reasons=[]
    for i,row in df.iterrows():
        r=[]
        if abs(row.get("BTC_Z",0) or 0)>THRESH: r.append(f"BTC_Z={row['BTC_Z']:.2f}")
        if abs(row.get("ETH_Z",0) or 0)>THRESH: r.append(f"ETH_Z={row['ETH_Z']:.2f}")
        if abs(row.get("Ratio_Z",0) or 0)>THRESH: r.append(f"Ratio_Z={row['Ratio_Z']:.2f}")
        if abs(row.get("BTC_Mom5",0) or 0)>0.05: r.append("BTC_Spike")
        if abs(row.get("ETH_Mom5",0) or 0)>0.05: r.append("ETH_Spike")
        reasons.append(", ".join(r) if r else "")
    df["Anomaly_Reason"]=reasons
    df["Anomaly"]=df["Anomaly_Reason"].str.len()>0
    return df
