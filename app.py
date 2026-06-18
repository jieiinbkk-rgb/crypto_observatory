"""
╔══════════════════════════════════════════════════════════════════════╗
║   Crypto Volatility Research Platform  v3.0                         ║
║   モジュール分離設計 · 研究品質最優先                                  ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time

# ── Config ────────────────────────────────────────────────────
from config.settings import (
    APP_VERSION, INITIAL_CAPITAL, MAX_POSITIONS,
    MARKET_STATES, STRATEGIES, VOL_REGIMES, IV_RV_ALERT,
    SHEET_ID,
)

# ── Data Layer ────────────────────────────────────────────────
from data.collector import (
    get_gsheet_client, get_dq_store, dq_score,
    load_raw_data, load_signals, load_trades,
    launch_collector, sheet_append,
)
from data.features import compute_features

# ── Strategy Layer ────────────────────────────────────────────
from strategy.classifier import (
    get_classifier_store, fit_gmm, classify_state, record_state,
    get_state_history_df, classify_vol_regime,
    compute_transition_matrix, compute_opportunity_score,
)
from strategy.signals import generate_signal

# ── Paper Trade Layer ─────────────────────────────────────────
from paper_trade.engine import (
    get_portfolio_store, open_trade, update_positions,
    close_manually, portfolio_stats, equity_curve,
)

# ── Backtest Layer ────────────────────────────────────────────
from backtest.engine import run_backtest

# ── Execution Layer ───────────────────────────────────────────
from execution.alerts import get_alert_store, fire_alerts, is_telegram_configured
from execution.trading_interface import TradingInterface

# ── UI Layer ──────────────────────────────────────────────────
from ui.styles import CSS, opp_score_card, state_card, position_card

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title=f"Crypto Vol Research Platform v{APP_VERSION}",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  INIT (once per session, cached)
# ══════════════════════════════════════════════════════════════
launch_collector()

ws_iv, ws_sig, ws_trd = get_gsheet_client()
sheet_ok   = ws_iv  is not None
signals_ok = ws_sig is not None

clf_store   = get_classifier_store()
port_store  = get_portfolio_store()
dq_store    = get_dq_store()
alert_store = get_alert_store()
trading_if  = TradingInterface()

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"## ⚙️ 設定  `v{APP_VERSION}`")

    auto_trade = st.toggle("🤖 自動シグナル → ペーパートレード", value=True)
    gmm_window = st.slider("GMM 学習データ数",  30, 300, 100, 10)
    display_n  = st.slider("チャート表示件数",  50, 500, 300, 50)

    st.markdown("---")
    st.markdown("### 📡 接続状態")
    st.success("✅ Google Sheets") if sheet_ok else st.warning("⚠️ CSV モード")
    if is_telegram_configured():
        st.success("✅ Telegram 接続中")
    else:
        st.info("💬 Telegram 未設定\n\nsecrets.toml に\nTELEGRAM_TOKEN と\nTELEGRAM_CHAT_ID を追加")

    # Data Quality
    dq_s, dq_st = dq_score(dq_store)
    st.markdown("---")
    st.markdown(f"### 🔬 System Health  {dq_st}")
    st.progress(dq_s / 100, text=f"{dq_s}/100")
    st.caption(
        f"API calls: {dq_store['api_calls']} | "
        f"Failures: {dq_store['api_failures']} | "
        f"Last: {dq_store['last_success'].strftime('%H:%M:%S') if dq_store['last_success'] else 'N/A'}"
    )

    st.markdown("---")
    st.markdown("### 🗺️ ステップ進捗")
    steps = [
        ("✅ Step 1  Data Collection",    "done"),
        ("✅ Step 2  Anomaly Detection",   "done"),
        ("✅ Step 3  Market State (GMM)",  "done"),
        ("✅ Step 4  Strategy Engine",     "done"),
        ("✅ Step 5  Paper Trading",       "done"),
        ("✅ Step 6  Google Sheets",       "done"),
        ("✅ Step 7  Opportunity Score",   "done"),
        ("✅ Step 8  IV-RV Spread",        "done"),
        ("✅ Step 9  Vol Regime",          "done"),
        ("✅ Step 10 Transition Matrix",   "done"),
        ("✅ Step 11 Signal DB",           "done"),
        ("✅ Step 12 Backtest Engine",     "done"),
        ("✅ Step 13 Research Dashboard",  "done"),
        ("✅ Step 14 Telegram Alerts",     "done"),
        ("✅ Step 15 Data Quality",        "done"),
        ("✅ Step 16 Modular Architecture","done"),
        ("⚪ Step 17 Live Trading",        ""),
    ]
    for label, cls in steps:
        st.markdown(f'<div class="roadmap {cls}">{label}</div>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ ペーパートレードリセット", type="secondary"):
        port_store["positions"]     = []
        port_store["closed_trades"] = []
        port_store["capital"]       = INITIAL_CAPITAL
        port_store["peak_capital"]  = INITIAL_CAPITAL
        port_store["max_drawdown"]  = 0.0
        st.success("リセット完了")

# ══════════════════════════════════════════════════════════════
#  DATA PIPELINE
# ══════════════════════════════════════════════════════════════
raw = load_raw_data(ws_iv)

if raw.empty:
    st.info("🔄 データ取得中... 初回は約1分かかります。しばらくお待ちください。")
    time.sleep(10)
    st.rerun()

df = compute_features(raw.copy())

# Data quality update
dq_store["total_rows"]    = len(df)
dq_store["missing_count"] = int(df[["BTC_IV","ETH_IV"]].isna().sum().sum())

# GMM fit + classify（30分に1回だけ再学習）
import datetime as _dt
_now = _dt.datetime.utcnow()
_last = clf_store.get("last_fit")
if _last is None or (_now - _last).seconds >= 1800:
    fit_gmm(clf_store, df.tail(gmm_window))
    clf_store["last_fit"] = _now
state_key, confidence, method = classify_state(clf_store, df)
record_state(clf_store, state_key, confidence, method)

# Derived metrics
opp_score, opp_reasons             = compute_opportunity_score(df, state_key, confidence)
vol_regime, regime_label, iv_pct   = classify_vol_regime(df)
transition_mx                       = compute_transition_matrix(clf_store)

latest       = df.iloc[-1]
btc_z_now    = float(latest.get("BTC_Z")  or 0)
eth_z_now    = float(latest.get("ETH_Z")  or 0)
btc_iv_now   = float(latest.get("BTC_IV") or 0)
eth_iv_now   = float(latest.get("ETH_IV") or 0)
state_info   = MARKET_STATES[state_key]

# Signal generation
signal = generate_signal(state_key, confidence, df, port_store["positions"], opp_score, clf_store)
if auto_trade and signal:
    from paper_trade.engine import check_risk_limits
    risk_ok, risk_msg = check_risk_limits(port_store)
    if risk_ok:
        open_trade(signal, port_store)
    else:
        pass  # リスク上限超過のためスキップ
    sheet_append(ws_sig, [
        signal["timestamp"], signal["state"], round(signal["confidence"],4),
        signal["id"], signal["strategy"], round(signal["btc_iv"],4),
        round(signal["eth_iv"],4), opp_score, method,
    ])

# Update paper trades
update_positions(port_store, df)

# Telegram alerts
fire_alerts(state_key, btc_z_now, eth_z_now, opp_score,
            confidence, btc_iv_now, eth_iv_now, alert_store)

# Load historical data
signal_history_df = load_signals(ws_sig)

# Chart data
df_disp   = df.tail(display_n)
chart_idx = df_disp.set_index("Timestamp")
stats     = portfolio_stats(port_store)
anomaly_count = int(df["Anomaly"].fillna(False).sum())

# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
h1, h2, h3 = st.columns([2, 1, 1])
with h1:
    st.markdown(f"# 🛰️ Crypto Volatility Research Platform `v{APP_VERSION}`")
    src = "Google Sheets" if sheet_ok else "CSV Local"
    st.caption(f"Deribit DVOL · 1-min · {len(df)} rows · {src} · Last: {latest['Timestamp']}")

with h2:
    st.markdown(state_card(state_info, confidence, method), unsafe_allow_html=True)

with h3:
    st.markdown(opp_score_card(opp_score, opp_reasons), unsafe_allow_html=True)

st.divider()

# ── KPI Row ───────────────────────────────────────────────────
def _delta(col):
    s = df[col]
    return round(float(s.iloc[-1] - s.iloc[-2]), 4) if len(s) >= 2 else None

k = st.columns(8)
k[0].metric("BTC ATM IV",    f"{btc_iv_now:.2f}",                  delta=_delta("BTC_IV"))
k[1].metric("ETH ATM IV",    f"{eth_iv_now:.2f}",                  delta=_delta("ETH_IV"))
k[2].metric("BTC/ETH Ratio", f"{latest['BTC_ETH_Ratio']:.4f}",     delta=_delta("BTC_ETH_Ratio"))
k[3].metric("BTC Z-score",   f"{btc_z_now:.2f}")
k[4].metric("Vol Regime",    " ".join(regime_label.split()[:2]))
k[5].metric("Anomalies",     f"{anomaly_count}")
k[6].metric("Opp Score",     f"{opp_score}/100")
pnl_str = f"+${stats['total_pnl_usd']:.0f}" if stats['total_pnl_usd'] >= 0 else f"-${abs(stats['total_pnl_usd']):.0f}"
k[7].metric("Paper P&L", pnl_str, delta=f"{stats['win_rate']}% WR" if stats["total_trades"] else None)

# 2行目KPI
k2 = st.columns(4)
fg_now = float(latest.get("Fear_Greed") or 0)
fr_now = float(latest.get("Funding_Rate") or 0)
fg_label = "Extreme Greed" if fg_now >= 75 else ("Greed" if fg_now >= 55 else ("Neutral" if fg_now >= 45 else ("Fear" if fg_now >= 25 else "Extreme Fear")))
k2[0].metric("Fear & Greed", f"{int(fg_now)}" if fg_now else "N/A", delta=fg_label if fg_now else None)
k2[1].metric("Funding Rate", f"{fr_now:.6f}" if fr_now else "N/A", delta="強気 🟢" if fr_now > 0 else ("弱気 🔴" if fr_now < 0 else None))
k2[2].metric("Vol Compression", f"{float(latest.get('Vol_Compression') or 0):.2f}")
k2[3].metric("IV Divergence", f"{float(latest.get('IV_Divergence') or 0):.4f}")

st.divider()

# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📈 IV Monitor",
    "🚨 Anomaly",
    "🧠 Market State",
    "📊 IV-RV Spread",
    "⚡ Strategy",
    "💼 Paper Trading",
    "🔬 Research",
    "⏮️ Backtest",
    "🗄️ Data & System",
])
(tab_iv, tab_anom, tab_state, tab_spread,
 tab_strat, tab_paper, tab_res, tab_bt, tab_data) = tabs

# ────────────────────────────────────────────────────────────
# TAB 1: IV Monitor
# ────────────────────────────────────────────────────────────
with tab_iv:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### BTC / ETH ATM IV")
        st.line_chart(chart_idx[["BTC_IV","ETH_IV"]], height=260)
    with c2:
        st.markdown("#### BTC/ETH IV Ratio")
        st.line_chart(chart_idx[["BTC_ETH_Ratio"]], color="#ffd700", height=260)

    c3, c4 = st.columns(2)
    with c3:
        spot = chart_idx[["BTC_Spot"]].dropna()
        if not spot.empty:
            st.markdown("#### BTC Spot Price")
            st.line_chart(spot, color="#f7931a", height=200)
    with c4:
        spot = chart_idx[["ETH_Spot"]].dropna()
        if not spot.empty:
            st.markdown("#### ETH Spot Price")
            st.line_chart(spot, color="#627eea", height=200)

    rv_cols = [c for c in ["BTC_RV10","ETH_RV10"] if c in chart_idx.columns]
    rv_data = chart_idx[rv_cols].dropna()
    if not rv_data.empty:
        st.markdown("#### Realized Vol Proxy (10-period)")
        st.line_chart(rv_data, height=180)

    # Fear & Greed / Funding Rate
    c5, c6 = st.columns(2)
    with c5:
        fg_data = chart_idx[["Fear_Greed"]].dropna() if "Fear_Greed" in chart_idx.columns else pd.DataFrame()
        if not fg_data.empty:
            st.markdown("#### Fear & Greed Index")
            latest_fg = int(fg_data.iloc[-1]["Fear_Greed"])
            fg_label = "Extreme Greed 🤑" if latest_fg >= 75 else ("Greed 😊" if latest_fg >= 55 else ("Neutral 😐" if latest_fg >= 45 else ("Fear 😨" if latest_fg >= 25 else "Extreme Fear 😱")))
            st.metric("現在値", f"{latest_fg}", delta=fg_label)
            st.line_chart(fg_data, color="#ffd700", height=160)
    with c6:
        fr_data = chart_idx[["Funding_Rate"]].dropna() if "Funding_Rate" in chart_idx.columns else pd.DataFrame()
        if not fr_data.empty:
            st.markdown("#### Funding Rate")
            latest_fr = float(fr_data.iloc[-1].get("Funding_Rate") or 0)
            fr_label = "強気 🟢" if latest_fr > 0 else "弱気 🔴"
            st.metric("現在値", f"{latest_fr:.6f}", delta=fr_label)
            st.line_chart(fr_data, color="#00c896" if latest_fr >= 0 else "#ff4b4b", height=160)

# ────────────────────────────────────────────────────────────
# TAB 2: Anomaly
# ────────────────────────────────────────────────────────────
with tab_anom:
    z_data = chart_idx[[c for c in ["BTC_Z","ETH_Z","Ratio_Z"] if c in chart_idx.columns]].dropna()
    if not z_data.empty:
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("#### IV Z-scores")
            st.line_chart(z_data[["BTC_Z","ETH_Z"]], height=230)
        with a2:
            st.markdown("#### Ratio Z-score")
            st.line_chart(z_data[["Ratio_Z"]], color="#ffd700", height=230)

    anomalies = df[df["Anomaly"].fillna(False)].copy()
    if anomalies.empty:
        st.success("✅ 現在、異常は検出されていません。")
    else:
        st.error(f"⚠️ {len(anomalies)} 件の異常を検出")
        cols = ["Timestamp","BTC_IV","ETH_IV","BTC_ETH_Ratio","BTC_Z","ETH_Z","Ratio_Z","Anomaly_Reason"]
        st.dataframe(
            anomalies[[c for c in cols if c in anomalies.columns]].tail(60)[::-1],
            use_container_width=True, height=280,
        )

# ────────────────────────────────────────────────────────────
# TAB 3: Market State
# ────────────────────────────────────────────────────────────
with tab_state:
    s1, s2 = st.columns([1, 2])

    with s1:
        st.markdown("#### 状態定義")
        for key, info in MARKET_STATES.items():
            if key == "unknown": continue
            is_cur = (key == state_key)
            border = f"border:1px solid {info['color']}88;" if is_cur else ""
            badge  = '<br><b style="color:#4b9eff;font-size:.7rem">← 現在</b>' if is_cur else ""
            st.markdown(
                f'<div class="card" style="{border}margin-bottom:.4rem;">'
                f'<span class="state-pill" style="background:{info["color"]}22;color:{info["color"]};'
                f'border:1px solid {info["color"]}44;font-size:.85rem;">{info["label"]}</span>'
                f'{badge}<div style="font-size:.75rem;color:#777;margin-top:.3rem;">'
                f'{info["description"]}</div></div>', unsafe_allow_html=True)

        rc = VOL_REGIMES[vol_regime]["color"]
        st.markdown(
            f'<div class="card" style="border-color:{rc}55;">'
            f'<div class="label">VOLATILITY REGIME</div>'
            f'<span class="regime-badge" style="background:{rc}22;color:{rc};border:1px solid {rc}44;">'
            f'{VOL_REGIMES[vol_regime]["label"]}</span>'
            f'<div style="font-size:.75rem;color:#777;margin-top:.3rem;">'
            f'IV Percentile: <b style="color:{rc}">{iv_pct:.0f}th</b></div>'
            f'</div>', unsafe_allow_html=True)

        gmm_ok = "✅ GMM稼働中" if clf_store.get("gmm_model") else "⏳ データ蓄積中..."
        st.info(f"分類器: **{method}** · {gmm_ok}")

    with s2:
        st.markdown("#### 状態遷移履歴")
        hist_df = get_state_history_df(clf_store)
        if not hist_df.empty:
            state_int = {"risk_on":1,"hedging":2,"squeeze":3,"panic":4,"unknown":0}
            hist_df["State_Int"] = hist_df["State"].map(state_int).fillna(0)
            st.line_chart(hist_df.set_index("Timestamp")[["State_Int","Confidence"]], height=200)
            hist_df["Prev"] = hist_df["State"].shift(1)
            for _, row in hist_df[hist_df["State"] != hist_df["Prev"]].tail(15)[::-1].iterrows():
                info  = MARKET_STATES.get(row["State"], MARKET_STATES["unknown"])
                c     = info["color"]
                st.markdown(
                    f'<div class="trade-row">'
                    f'<span style="color:#555;font-size:.75rem;min-width:130px">{row["Timestamp"]}</span>'
                    f'<span class="state-pill" style="background:{c}22;color:{c};'
                    f'border:1px solid {c}44;font-size:.75rem;padding:.15em .6em">{info["label"]}</span>'
                    f'<span style="color:#777;font-size:.75rem">'
                    f'conf: {row["Confidence"]*100:.0f}% · {row["Method"]}</span>'
                    f'</div>', unsafe_allow_html=True)
        else:
            st.info("状態履歴を蓄積中...")

    st.markdown("#### 状態遷移確率 Matrix (%)")
    if transition_mx is not None:
        st.dataframe(
            transition_mx.style.background_gradient(cmap="YlOrRd", vmin=0, vmax=100)
                               .format("{:.1f}%"),
            use_container_width=True,
        )
        st.markdown("##### 最多遷移先")
        tcols = st.columns(4)
        for i, s_from in enumerate([s for s in MARKET_STATES if s != "unknown"]):
            if s_from not in transition_mx.index: continue
            row      = transition_mx.loc[s_from]
            best_to  = row.idxmax(); best_pct = row.max()
            info_f   = MARKET_STATES[s_from]
            info_t   = MARKET_STATES.get(best_to, MARKET_STATES["unknown"])
            with tcols[i % 4]:
                st.markdown(
                    f'<div class="card" style="padding:.6rem;">'
                    f'<span style="color:{info_f["color"]};font-size:.8rem;">{info_f["label"]}</span>'
                    f'<div style="color:#555;font-size:.7rem;margin:.2rem 0;">↓ {best_pct:.1f}%</div>'
                    f'<span style="color:{info_t["color"]};font-size:.8rem;">{info_t["label"]}</span>'
                    f'</div>', unsafe_allow_html=True)
    else:
        st.info("遷移行列の計算には十分な状態履歴が必要です（最低10状態変化）")

    if clf_store.get("gmm_model"):
        with st.expander("🔬 GMM クラスター詳細"):
            gmm   = clf_store["gmm_model"]
            sclr  = clf_store["gmm_scaler"]
            lmap  = clf_store["gmm_label_map"]
            means = sclr.inverse_transform(gmm.means_)
            feat_names = ["BTC_Z","ETH_Z","Ratio_Z","BTC_Mom5","ETH_Mom5","IV_Div_Z"]
            rows  = [{
                "Cluster": i, "→ State": MARKET_STATES[lmap.get(i,"unknown")]["label"],
                "Weight%": f"{w*100:.1f}%",
                **{feat_names[j]: f"{m[j]:.2f}" for j in range(min(len(m),len(feat_names)))}
            } for i,(m,w) in enumerate(zip(means, gmm.weights_))]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ────────────────────────────────────────────────────────────
# TAB 4: IV-RV Spread Monitor
# ────────────────────────────────────────────────────────────
with tab_spread:
    st.markdown("#### 📊 IV-RV Spread Monitor")
    st.caption("正 = オプション割高（IVがRVより高い） · 負 = 割安")

    def _s(v): return round(float(v), 2) if pd.notna(v) else 0.0
    biv = _s(latest.get("BTC_IV")); eiv = _s(latest.get("ETH_IV"))
    brv20 = _s(latest.get("BTC_RV20")); brv60 = _s(latest.get("BTC_RV60"))
    erv20 = _s(latest.get("ETH_RV20")); erv60 = _s(latest.get("ETH_RV60"))
    bsp20 = _s(latest.get("BTC_Spread20")); bsp60 = _s(latest.get("BTC_Spread60"))
    esp20 = _s(latest.get("ETH_Spread20")); esp60 = _s(latest.get("ETH_Spread60"))

    sc = st.columns(6)
    sc[0].metric("BTC IV",  f"{biv:.1f}")
    sc[1].metric("BTC RV20",f"{brv20:.1f}")
    sc[2].metric("BTC RV60",f"{brv60:.1f}")
    sc[3].metric("ETH IV",  f"{eiv:.1f}")
    sc[4].metric("ETH RV20",f"{erv20:.1f}")
    sc[5].metric("ETH RV60",f"{erv60:.1f}")

    st.divider()
    for sym, sp20, sp60 in [("BTC", bsp20, bsp60), ("ETH", esp20, esp60)]:
        for period, sp in [("20", sp20), ("60", sp60)]:
            if abs(sp) > IV_RV_ALERT:
                d   = "RICH" if sp > 0 else "CHEAP"
                clr = "#ff9900" if d == "RICH" else "#4b9eff"
                st.markdown(
                    f'<div class="spread-alert">⚠️ <b style="color:{clr}">'
                    f'{sym} IV-RV{period} Spread = {sp:+.1f}</b> — {d}</div>',
                    unsafe_allow_html=True)

    sp1, sp2 = st.columns(2)
    with sp1:
        st.markdown("#### BTC IV vs RV20/RV60")
        btc_rv = chart_idx[["BTC_IV","BTC_RV20","BTC_RV60"]].dropna()
        if not btc_rv.empty: st.line_chart(btc_rv, height=240)
        st.markdown("#### BTC Spread")
        btc_sp = chart_idx[["BTC_Spread20","BTC_Spread60"]].dropna()
        if not btc_sp.empty: st.line_chart(btc_sp, height=180)
    with sp2:
        st.markdown("#### ETH IV vs RV20/RV60")
        eth_rv = chart_idx[["ETH_IV","ETH_RV20","ETH_RV60"]].dropna()
        if not eth_rv.empty: st.line_chart(eth_rv, height=240)
        st.markdown("#### ETH Spread")
        eth_sp = chart_idx[["ETH_Spread20","ETH_Spread60"]].dropna()
        if not eth_sp.empty: st.line_chart(eth_sp, height=180)

    st.markdown("#### スプレッド サマリー")
    st.dataframe(pd.DataFrame([
        {"Symbol":"BTC","IV":biv,"RV20":brv20,"RV60":brv60,
         "Spread20":f"{bsp20:+.2f}","Spread60":f"{bsp60:+.2f}",
         "Status20":"RICH" if bsp20>IV_RV_ALERT else ("CHEAP" if bsp20<-IV_RV_ALERT else "NORMAL"),
         "Status60":"RICH" if bsp60>IV_RV_ALERT else ("CHEAP" if bsp60<-IV_RV_ALERT else "NORMAL")},
        {"Symbol":"ETH","IV":eiv,"RV20":erv20,"RV60":erv60,
         "Spread20":f"{esp20:+.2f}","Spread60":f"{esp60:+.2f}",
         "Status20":"RICH" if esp20>IV_RV_ALERT else ("CHEAP" if esp20<-IV_RV_ALERT else "NORMAL"),
         "Status60":"RICH" if esp60>IV_RV_ALERT else ("CHEAP" if esp60<-IV_RV_ALERT else "NORMAL")},
    ]), use_container_width=True)

# ────────────────────────────────────────────────────────────
# TAB 5: Strategy
# ────────────────────────────────────────────────────────────
with tab_strat:
    strat_key = state_info.get("strategy")
    if strat_key and strat_key in STRATEGIES:
        strat = STRATEGIES[strat_key]; color = state_info["color"]
        st.markdown(
            f'<div class="signal-card" style="border-color:{color}55;">'
            f'<div style="display:flex;justify-content:space-between;align-items:start;">'
            f'<div><div class="label">現在の推奨戦略</div>'
            f'<div style="font-size:1.1rem;font-weight:700;color:{color}">{strat["name"]}</div></div>'
            f'<span class="state-pill" style="background:{color}22;color:{color};'
            f'border:1px solid {color}44;font-size:.85rem">{strat["action"]}</span></div>'
            f'<div style="margin-top:.7rem;display:grid;grid-template-columns:1fr 1fr;gap:.5rem;">'
            f'<div><div class="label">Legs</div>'
            f'<div style="color:#c8cad4;font-size:.85rem">{" · ".join(strat["legs"])}</div></div>'
            f'<div><div class="label">Target / Stop</div>'
            f'<span class="tag green">TP +{strat["target_pnl_pct"]*100:.0f}%</span>'
            f'<span class="tag red">SL {strat["stop_pnl_pct"]*100:.0f}%</span></div>'
            f'<div style="grid-column:span 2"><div class="label">Rationale</div>'
            f'<div style="color:#a0a8b8;font-size:.82rem">{strat["rationale"]}</div></div>'
            f'<div style="grid-column:span 2"><div class="label">Risk</div>'
            f'<div style="color:#886060;font-size:.82rem">{strat["risk"]}</div></div>'
            f'</div></div>', unsafe_allow_html=True)
    else:
        st.info("現在、明確な戦略シグナルはありません。")

    st.markdown("#### 全戦略マップ")
    for sk, sv in STRATEGIES.items():
        sf  = [k for k,v in MARKET_STATES.items() if v.get("strategy")==sk]
        sc  = MARKET_STATES[sf[0]]["color"] if sf else "#888"
        bd  = f"border:1px solid {sc}88;" if sk == strat_key else ""
        st.markdown(
            f'<div class="card" style="{bd}margin-bottom:.3rem;">'
            f'<div style="display:flex;justify-content:space-between;">'
            f'<b style="color:{sc}">{sv["name"]}</b>'
            f'<span class="tag" style="background:{sc}22;color:{sc}">{sv["action"]}</span></div>'
            f'<div style="font-size:.78rem;color:#888;margin-top:.3rem">{sv["rationale"]}</div>'
            f'</div>', unsafe_allow_html=True)

    # IVスキュー分析
    st.markdown("#### 📐 IVスキュー分析")
    try:
        from strategy.skew import get_skew_data
        skew_data = get_skew_data("BTC")
        if skew_data:
            sk = st.columns(5)
            sk[0].metric("ATM IV",        f"{skew_data['atm_iv']:.1f}")
            sk[1].metric("OTM Call IV",   f"{skew_data['otm_call_iv']:.1f}")
            sk[2].metric("OTM Put IV",    f"{skew_data['otm_put_iv']:.1f}")
            rr = skew_data['risk_reversal']
            sk[3].metric("Risk Reversal", f"{rr:+.2f}",
                        delta="上昇期待 🟢" if rr > 0 else "下落警戒 🔴")
            bf = skew_data['butterfly']
            sk[4].metric("Butterfly",     f"{bf:+.2f}",
                        delta="テール需要高 ⚠️" if bf > 2 else "通常")
            st.caption(f"限月: {skew_data.get('expiry','?')} · Spot: ${skew_data.get('spot',0):,.0f}")
        else:
            st.info("スキューデータ取得中...")
    except Exception as e:
        st.warning(f"スキュー取得エラー: {e}")

    if not signal_history_df.empty:
        st.markdown("#### 📋 シグナル履歴")
        st.dataframe(signal_history_df.tail(50)[::-1], use_container_width=True, height=260)

# ────────────────────────────────────────────────────────────
# TAB 6: Paper Trading
# ────────────────────────────────────────────────────────────
with tab_paper:
    cap_pct = (port_store["capital"] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    m = st.columns(4)
    m[0].metric("元本",       f"${INITIAL_CAPITAL:,.0f}")
    m[1].metric("現在の資産", f"${port_store['capital']:,.0f}", delta=f"{cap_pct:+.1f}%")
    m[2].metric("総トレード", stats["total_trades"])
    m[3].metric("勝率",       f"{stats['win_rate']}%")
    m2 = st.columns(5)
    m2[0].metric("累計P&L",    f"${stats['total_pnl_usd']:+,.2f}")
    m2[1].metric("平均P&L%",   f"{stats['avg_pnl_pct']:+.2f}%")
    m2[2].metric("Sharpe",     f"{stats['sharpe']:.2f}")
    m2[3].metric("Max DD",     f"{stats['max_drawdown']:.1f}%")
    m2[4].metric("Profit Factor", f"{stats['profit_factor']:.2f}")

    st.divider()
    st.markdown(f"#### 🔓 オープンポジション ({len(port_store['positions'])} / {MAX_POSITIONS})")
    if not port_store["positions"]:
        st.info("現在、オープンポジションはありません。")
    else:
        for pos in port_store["positions"]:
            si   = MARKET_STATES.get(pos["state"], MARKET_STATES["unknown"])
            pa, pb = st.columns([3, 1])
            with pa:
                st.markdown(position_card(pos, si), unsafe_allow_html=True)
            with pb:
                if st.button("手動決済", key=f"cls_{pos['id']}"):
                    close_manually(pos, port_store, df)
                    st.rerun()

    st.divider()
    st.markdown(f"#### 📋 決済済みトレード ({len(port_store['closed_trades'])} 件)")
    if port_store["closed_trades"]:
        rows = [{
            "ID": t["id"], "戦略": t["strategy_name"],
            "状態": MARKET_STATES.get(t["state"],{}).get("label","?"),
            "エントリー": t["entry_time"], "決済": t.get("exit_time","—"),
            "P&L%": f'{t.get("current_pnl_pct",0)*100:+.2f}%',
            "P&L$": f'${t["current_pnl_usd"]:+.2f}',
            "手数料$": f'${t.get("cost_usd",0):.3f}',
            "終了": t.get("status","?"),
        } for t in reversed(port_store["closed_trades"])]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=260)

        eq = equity_curve(port_store)
        st.markdown("#### Equity Curve")
        st.line_chart(pd.DataFrame({"Equity": eq}), color="#00c896", height=200)
    else:
        st.info("まだ決済済みトレードはありません。")

# ────────────────────────────────────────────────────────────
# TAB 7: Research Dashboard
# ────────────────────────────────────────────────────────────
with tab_res:
    st.markdown("### 🔬 Volatility Research Dashboard")
    st.caption("研究者向け統計分析 · 分布 · 相関 · ローリング統計")

    btc_s = df["BTC_IV"].dropna()
    eth_s = df["ETH_IV"].dropna()
    rat_s = df["BTC_ETH_Ratio"].dropna()

    st.markdown("#### 記述統計")
    st.dataframe(
        pd.DataFrame({"BTC IV": btc_s.describe(), "ETH IV": eth_s.describe(), "Ratio": rat_s.describe()}).round(4),
        use_container_width=True,
    )
    st.divider()

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("#### BTC IV 分布")
        if len(btc_s) >= 5:
            vals, bins = np.histogram(btc_s, bins=30)
            centers = (bins[:-1]+bins[1:])/2
            st.bar_chart(pd.DataFrame({"BTC_IV":vals}, index=centers.round(1)), color="#f7931a", height=220)
    with r2:
        st.markdown("#### ETH IV 分布")
        if len(eth_s) >= 5:
            vals, bins = np.histogram(eth_s, bins=30)
            centers = (bins[:-1]+bins[1:])/2
            st.bar_chart(pd.DataFrame({"ETH_IV":vals}, index=centers.round(1)), color="#627eea", height=220)
    with r3:
        st.markdown("#### Ratio 分布")
        if len(rat_s) >= 5:
            vals, bins = np.histogram(rat_s, bins=30)
            centers = (bins[:-1]+bins[1:])/2
            st.bar_chart(pd.DataFrame({"Ratio":vals}, index=centers.round(4)), color="#ffd700", height=220)

    st.divider()
    st.markdown("#### Correlation Matrix")
    corr_cols = [c for c in ["BTC_IV","ETH_IV","BTC_ETH_Ratio","BTC_Z","ETH_Z","Ratio_Z","BTC_RV10","ETH_RV10"] if c in df.columns]
    if len(corr_cols) >= 3:
        corr_df = df[corr_cols].dropna().corr().round(3)
        def _color(v):
            if v >= 0.7:  return "background-color:#00c89644"
            if v >= 0.3:  return "background-color:#00c89622"
            if v <= -0.7: return "background-color:#ff4b4b44"
            if v <= -0.3: return "background-color:#ff4b4b22"
            return ""
        st.dataframe(corr_df.style.map(_color), use_container_width=True)

    st.divider()
    st.markdown("#### Rolling Correlation (BTC IV ↔ ETH IV)")
    rc_data = {}
    for w in [20, 60]:
        rc_data[f"RollCorr_{w}"] = df["BTC_IV"].rolling(w).corr(df["ETH_IV"])
    rc_df = pd.DataFrame(rc_data, index=df["Timestamp"]).dropna()
    if not rc_df.empty:
        st.line_chart(rc_df, height=200)

    # 4資産IV比較
    st.divider()
    st.markdown("#### 4資産 IV 比較")
    multi_iv_cols = [c for c in ["BTC_IV","ETH_IV","SOL_IV","BNB_IV"] if c in chart_idx.columns]
    multi_iv_data = chart_idx[multi_iv_cols].dropna()
    if not multi_iv_data.empty:
        st.line_chart(multi_iv_data, height=220)

    # 相関マトリクス（4資産）
    corr_cols2 = [c for c in ["BTC_IV","ETH_IV","SOL_IV","BNB_IV"] if c in df.columns]
    if len(corr_cols2) >= 2:
        st.markdown("#### 4資産 IV 相関")
        corr2 = df[corr_cols2].dropna().corr().round(3)
        def _color2(v):
            if v >= 0.7:  return "background-color:#00c89644"
            if v >= 0.3:  return "background-color:#00c89622"
            if v <= -0.7: return "background-color:#ff4b4b44"
            if v <= -0.3: return "background-color:#ff4b4b22"
            return ""
        st.dataframe(corr2.style.map(_color2), use_container_width=True)

    st.markdown("#### IVパーセンタイル分析")
    qs = [0.05,0.10,0.25,0.50,0.75,0.90,0.95]
    q_rows = [{"Pct":f"{int(q*100)}th",
               "BTC IV":round(btc_s.quantile(q),2) if len(btc_s)>5 else np.nan,
               "ETH IV":round(eth_s.quantile(q),2) if len(eth_s)>5 else np.nan,
               "Ratio": round(rat_s.quantile(q),4) if len(rat_s)>5 else np.nan}
              for q in qs]
    st.dataframe(pd.DataFrame(q_rows), use_container_width=True)

# ────────────────────────────────────────────────────────────
# TAB 8: Backtest
# ────────────────────────────────────────────────────────────
with tab_bt:
    st.markdown("### ⏮️ Backtest Engine")
    st.caption("シグナル発生後 1h / 6h / 24h のIV変化からP&Lを推計")

    if signal_history_df.empty:
        st.info("シグナル履歴が蓄積されると自動的に分析されます。（最低10件）")
    else:
        st.success(f"✅ {len(signal_history_df)} 件のシグナルでバックテスト実行")
        bt_results = run_backtest(df, signal_history_df)

        if not bt_results:
            st.warning("バックテスト結果を計算できませんでした。データが不十分な可能性があります。")
        else:
            st.markdown("#### 戦略別パフォーマンスランキング (Sharpe降順)")
            rank_rows = []
            for sname, rdf in bt_results.items():
                for horizon, row in rdf.iterrows():
                    rank_rows.append({"Strategy":sname,"Horizon":horizon,
                                      "WinRate%":row.get("WinRate%",0),
                                      "AvgRet%":row.get("AvgRet%",0),
                                      "Sharpe":row.get("Sharpe",0),
                                      "MaxDD%":row.get("MaxDD%",0),
                                      "N":int(row.get("N",0))})
            if rank_rows:
                rank_df = pd.DataFrame(rank_rows).sort_values("Sharpe", ascending=False)
                def _rc(v):
                    if v > 0: return "color:#00c896"
                    if v < 0: return "color:#ff4b4b"
                    return ""
                st.dataframe(
                    rank_df.style.map(_rc, subset=["AvgRet%","Sharpe"]),
                    use_container_width=True, height=280,
                )

            for sname, rdf in bt_results.items():
                with st.expander(f"📊 {sname}"):
                    st.dataframe(rdf, use_container_width=True)

        st.markdown("#### シグナル履歴テーブル")
        st.dataframe(signal_history_df[::-1], use_container_width=True, height=260)

        # 教師あり学習
        st.markdown("#### 🤖 教師あり学習（自動ラベリング）")
        try:
            from strategy.labeler import label_signals, train_simple_classifier
            labeled_df = label_signals(df, signal_history_df)
            if labeled_df.empty:
                st.info("ラベル付けにはシグナル発生後1時間以上のデータが必要です。")
            else:
                st.success(f"✅ {len(labeled_df)} 件のラベル付きデータ")
                win_rate = labeled_df["Label_1h"].mean() * 100
                st.metric("予測正解率", f"{win_rate:.1f}%")
                st.dataframe(labeled_df[::-1], use_container_width=True, height=200)

                clf, msg = train_simple_classifier(labeled_df)
                if clf:
                    st.success(f"🎯 RandomForest学習完了 · {msg}")
                    importances = clf.feature_importances_
                    st.markdown(f"特徴量重要度: Confidence={importances[0]:.2f} / OppScore={importances[1]:.2f}")
                else:
                    st.info(msg)
        except Exception as e:
            st.warning(f"教師あり学習エラー: {e}")

# ────────────────────────────────────────────────────────────
# TAB 9: Data & System
# ────────────────────────────────────────────────────────────
with tab_data:
    st.markdown("#### 🗄️ データ & システム")
    d1, d2 = st.columns(2)

    with d1:
        st.markdown("##### IV Data")
        if sheet_ok:
            st.success(f"✅ {len(df)} 行蓄積済み")
            st.markdown(f"[📊 スプレッドシートを開く](https://docs.google.com/spreadsheets/d/{SHEET_ID})")
        else:
            st.warning("CSV モード動作中")
        st.dataframe(df.tail(40)[::-1], use_container_width=True, height=260)
        st.download_button("⬇️ IV Data CSV",
                           df.to_csv(index=False).encode("utf-8"),
                           f"iv_data_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv")

    with d2:
        st.markdown("##### Signal History")
        if signals_ok:
            st.success(f"✅ signal_history · {len(signal_history_df)} 件")
        if not signal_history_df.empty:
            st.dataframe(signal_history_df[::-1], use_container_width=True, height=260)
            st.download_button("⬇️ Signal CSV",
                               signal_history_df.to_csv(index=False).encode("utf-8"),
                               f"signals_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv")
        else:
            st.info("シグナルはまだありません。")

    with st.expander("🔬 データ品質詳細"):
        dq_s2, dq_st2 = dq_score(dq_store)
        qc = st.columns(4)
        qc[0].metric("品質スコア",  f"{dq_s2}/100")
        qc[1].metric("API呼び出し", dq_store["api_calls"])
        qc[2].metric("API失敗",     dq_store["api_failures"])
        qc[3].metric("欠損行",      dq_store["missing_count"])
        st.progress(dq_s2/100, text=f"System Health: {dq_st2}")
        miss_r = dq_store["missing_count"] / max(dq_store["total_rows"], 1) * 100
        fail_r = dq_store["api_failures"]  / max(dq_store["api_calls"],  1) * 100
        st.caption(f"欠損率: {miss_r:.1f}% | API失敗率: {fail_r:.1f}% | "
                   f"最終成功: {dq_store['last_success'].strftime('%Y-%m-%d %H:%M:%S') if dq_store['last_success'] else 'N/A'}")

    with st.expander("🗂️ 生データ (最新200件)"):
        st.dataframe(df.tail(200)[::-1], use_container_width=True, height=280)

    with st.expander("🚀 Future Trading Layer (設計のみ・未接続)"):
        st.json(trading_if.status())
        st.markdown("""
**リアル取引移行手順:**
1. Deribit APIキー取得（read + trade権限）
2. `.streamlit/secrets.toml` に `DERIBIT_API_KEY` / `DERIBIT_API_SECRET` を追加
3. `execution/trading_interface.py` の `DeribitLiveTradingEngine` を実装
4. `paper_trade/engine.py` の `open_trade()` → `place_order()` に差し替え
        """)

    with st.expander("📐 アーキテクチャ概要"):
        st.markdown("""
```
crypto_vol_platform_v3/
├── app.py
├── requirements.txt
├── config/settings.py
├── data/collector.py
├── data/features.py
├── strategy/classifier.py
├── strategy/signals.py
├── paper_trade/engine.py
├── backtest/engine.py
├── execution/alerts.py
├── execution/trading_interface.py
└── ui/styles.py
```
        """)

# ══════════════════════════════════════════════════════════════
#  AUTO REFRESH
# ══════════════════════════════════════════════════════════════
time.sleep(10)
st.rerun()
