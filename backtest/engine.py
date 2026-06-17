"""
backtest/engine.py  v2
- Sharpe計算を修正（時間足に合わせた年率換算）
- 戦略別の正確なPnL推計
- 勝率・最大DDの改善
"""
import pandas as pd
import numpy as np
from config.settings import STRATEGIES

HORIZONS = {"1h": 60, "6h": 360, "24h": 1440}

IV_SENSITIVITY = {
    "vol_sell":     -1.2,
    "ratio_spread": -0.6,
    "gamma_long":    1.5,
    "tail_hedge":    2.0,
}

def _calc_ret(strat, btc_chg, eth_chg):
    sens = IV_SENSITIVITY.get(strat, 0.5)
    action = STRATEGIES.get(strat, {}).get("action", "")
    if strat == "gamma_long":
        avg = (abs(btc_chg) + abs(eth_chg)) / 2
        return avg * sens * (1 + avg * 2)
    elif strat == "ratio_spread":
        return -(btc_chg - eth_chg) * abs(sens)
    else:
        avg = (btc_chg + eth_chg) / 2
        return avg * sens

def run_backtest(df, signal_df):
    if signal_df.empty or df.empty: return {}
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    signal_df = signal_df.copy()
    signal_df["Timestamp"] = pd.to_datetime(signal_df["Timestamp"])

    results = {}
    for _, sig in signal_df.iterrows():
        ts = sig["Timestamp"]
        strat = str(sig.get("Strategy", "unknown"))
        btc_iv0 = float(sig.get("BTC_IV", np.nan) or np.nan)
        eth_iv0 = float(sig.get("ETH_IV", np.nan) or np.nan)
        if np.isnan(btc_iv0): continue

        for horizon_label, horizon_min in HORIZONS.items():
            target_ts  = ts + pd.Timedelta(minutes=horizon_min)
            future     = df[df["Timestamp"] >= target_ts]
            if future.empty: continue

            btc_iv1 = float(future.iloc[0].get("BTC_IV", np.nan) or np.nan)
            eth_iv1 = float(future.iloc[0].get("ETH_IV", np.nan) or np.nan)
            if np.isnan(btc_iv1): continue

            btc_chg = (btc_iv1 - btc_iv0) / max(btc_iv0, 1)
            eth_chg = (eth_iv1 - (eth_iv0 or btc_iv0)) / max(eth_iv0 or btc_iv0, 1)
            ret     = _calc_ret(strat, btc_chg, eth_chg)

            results.setdefault(strat, []).append({
                "Horizon":   horizon_label,
                "Signal_TS": ts,
                "BTC_IV_0":  btc_iv0,
                "BTC_IV_1":  btc_iv1,
                "IV_Chg%":   round(btc_chg * 100, 2),
                "Ret%":      round(ret * 100, 2),
                "Win":       int(ret > 0),
            })

    summary = {}
    for strat, rows in results.items():
        raw = pd.DataFrame(rows)
        agg = []
        for h, hdf in raw.groupby("Horizon"):
            rets = hdf["Ret%"].values
            n    = len(hdf)
            wins = hdf["Win"].sum()
            std  = np.std(rets) if len(rets) > 1 else 0

            # ホライズンに合わせた年率換算
            horizon_map = {"1h": 8760, "6h": 1460, "24h": 365}
            annual_factor = np.sqrt(horizon_map.get(h, 365))
            sharpe = float(np.mean(rets) / std * annual_factor) if std > 0 else 0.0

            # 最大ドローダウン
            cum  = (1 + np.array(rets) / 100).cumprod()
            peak = np.maximum.accumulate(cum)
            dd   = ((cum - peak) / peak * 100).min() if len(cum) > 0 else 0

            agg.append({
                "Horizon":   h,
                "N":         n,
                "WinRate%":  round(wins / n * 100, 1),
                "AvgRet%":   round(float(np.mean(rets)), 2),
                "MaxDD%":    round(float(dd), 2),
                "Sharpe":    round(sharpe, 2),
            })
        summary[strat] = pd.DataFrame(agg).set_index("Horizon")
    return summary
