"""
backtest/engine.py
シグナル発生後の IV 変化から PnL を推計するヒストリカルバックテスト
"""
import pandas as pd
import numpy as np
from config.settings import STRATEGIES

HORIZONS = {"1h": 60, "6h": 360, "24h": 1440}


def run_backtest(
    df: pd.DataFrame,
    signal_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:

    if signal_df.empty or df.empty:
        return {}

    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    signal_df = signal_df.copy()
    signal_df["Timestamp"] = pd.to_datetime(signal_df["Timestamp"])

    results: dict[str, list] = {}

    # ★ここが関数内である必要あり
    for _, sig in signal_df.iterrows():
        ts = sig["Timestamp"]
        strat = str(sig.get("Strategy", "unknown"))
        btc_iv0 = float(sig.get("BTC_IV", np.nan) or np.nan)

        if np.isnan(btc_iv0):
            continue

        action = STRATEGIES.get(strat, {}).get("action", "LONG VOL")

        for horizon_label, horizon_min in HORIZONS.items():
            target_ts = ts + pd.Timedelta(minutes=horizon_min)
            future = df[df["Timestamp"] >= target_ts]

            if future.empty:
                continue

            btc_iv1 = float(future.iloc[0].get("BTC_IV", np.nan) or np.nan)
            if np.isnan(btc_iv1):
                continue

            chg = (btc_iv1 - btc_iv0) / max(btc_iv0, 1)
            ret = (-chg * 0.5) if ("SHORT" in action or "SELL" in action) else (chg * 0.5)

            results.setdefault(strat, []).append({
                "Horizon": horizon_label,
                "Signal_TS": ts,
                "BTC_IV_0": btc_iv0,
                "BTC_IV_1": btc_iv1,
                "IV_Chg%": round(chg * 100, 2),
                "Ret%": round(ret * 100, 2),
                "Win": int(ret > 0),
            })

    summary: dict[str, pd.DataFrame] = {}

    for strat, rows in results.items():
        raw = pd.DataFrame(rows)
        agg = []

        for h, hdf in raw.groupby("Horizon"):
            rets = hdf["Ret%"].values
            wins = hdf["Win"].sum()
            n = len(hdf)
            std = np.std(rets) if len(rets) > 1 else 0
            sharpe = float(np.mean(rets) / std * np.sqrt(252)) if std > 0 else 0.0

            agg.append({
                "Horizon": h,
                "N": n,
                "WinRate%": round(wins / n * 100, 1),
                "AvgRet%": round(float(np.mean(rets)), 2),
                "MaxDD%": round(float(np.min(rets)), 2),
                "Sharpe": round(sharpe, 2),
            })

        summary[strat] = pd.DataFrame(agg).set_index("Horizon")

    return summary