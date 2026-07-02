"""
╔══════════════════════════════════════════════════════════════════╗
║   Crypto Observatory  –  Live State Viewer  v4.1               ║
╚══════════════════════════════════════════════════════════════════╝
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import time

import gspread
from google.oauth2.service_account import Credentials
import numpy as np

SHEET_ID = "1k3lJ1_ru8ubhqV5p_CQgjlEtYSgxeVEDTraPUFifOE0"
SCOPES   = ["https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"]

MARKET_STATES = {
    "risk_on":  {"label": "Risk-On 🟢",   "color": "#00c896"},
    "hedging":  {"label": "Hedging 🟡",   "color": "#ffd700"},
    "squeeze":  {"label": "Squeeze 🔵",   "color": "#4b9eff"},
    "panic":    {"label": "Panic 🔴",     "color": "#ff4b4b"},
    "unknown":  {"label": "Observing ⚪", "color": "#888888"},
}

st.set_page_config(
    page_title="Crypto Observatory v4.1",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
[data-testid="stApp"] { background: #0d0e13; color: #c8cad4; }
.metric-card { background: #161820; border: 1px solid #2a2d3e; border-radius: 10px; padding: 14px 18px; text-align: center; }
.metric-label { font-size: .72rem; color: #666; text-transform: uppercase; letter-spacing: .08em; }
.metric-value { font-size: 1.4rem; font-weight: 700; color: #e0e2ec; margin-top: .2rem; }
.pos-card { background: #161820; border: 1px solid #2a2d3e; border-radius: 8px; padding: 12px 16px; margin: .4rem 0; font-size: .85rem; }
.grid-row { display: flex; gap: 10px; align-items: center; padding: 6px 0; border-bottom: 1px solid #222; font-size: .82rem; }
.badge { display: inline-block; padding: .2em .7em; border-radius: 4px; font-size: .75rem; font-weight: 600; }
.badge-buy  { background: #00c89622; color: #00c896; border: 1px solid #00c89644; }
.badge-sell { background: #ff4b4b22; color: #ff4b4b; border: 1px solid #ff4b4b44; }
.badge-wait { background: #88888822; color: #888; border: 1px solid #88888844; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  Google Sheets 接続
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def get_sheet():
    try:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=SCOPES)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(SHEET_ID)
        return sh.worksheet("live_state"), None
    except Exception as e:
        return None, str(e)


def load_live_state(ws):
    if ws is None:
        return None, None
    try:
        row = ws.row_values(2)
        if len(row) < 2:
            return None, None
        return row[0], json.loads(row[1])
    except Exception as e:
        return None, None


# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("# 🛰️ Crypto Observatory `v4.1`")

ws, conn_err = get_sheet()

if ws is None:
    st.error(f"⚠️ Google Sheets接続失敗: `{conn_err}`")
    st.info("Streamlit Cloud → Settings → Secrets に `gcp_service_account` が設定されているか確認してください。")
    st.stop()

updated_at, live_state = load_live_state(ws)

if live_state is None:
    st.warning("live_stateデータなし。VMエンジンが起動していない可能性があります。")
    time.sleep(60)
    st.rerun()

st.caption(f"VM最終更新: {updated_at or '—'}　|　60秒ごとに自動更新")

ms    = live_state.get("market_state", {})
skey  = ms.get("state", "unknown")
sinfo = MARKET_STATES.get(skey, MARKET_STATES["unknown"])

h1, h2, h3, h4, h5 = st.columns(5)
h1.markdown(f'<div class="metric-card"><div class="metric-label">市場状態</div><div class="metric-value" style="color:{sinfo["color"]}">{sinfo["label"]}</div></div>', unsafe_allow_html=True)
h2.markdown(f'<div class="metric-card"><div class="metric-label">Confidence</div><div class="metric-value">{ms.get("confidence",0)*100:.0f}%</div></div>', unsafe_allow_html=True)
h3.markdown(f'<div class="metric-card"><div class="metric-label">機会スコア</div><div class="metric-value">{ms.get("opp_score",0)}/100</div></div>', unsafe_allow_html=True)
h4.markdown(f'<div class="metric-card"><div class="metric-label">GMM</div><div class="metric-value">{"稼働中 ✅" if ms.get("gmm_active") else "学習中 ⏳"}</div></div>', unsafe_allow_html=True)
total_rows = ms.get("total_rows", ms.get("n_samples", 0))
h5.markdown(f'<div class="metric-card"><div class="metric-label">データ数</div><div class="metric-value">{total_rows:,}</div></div>', unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
tab_grid, tab_pmr, tab_bt, tab_paper, tab_state, tab_sys = st.tabs([
    "🟪 Grid BOT", "💹 PMR BOT", "🔬 バックテスト", "💼 Paper Trading", "🧠 Market State", "🗄️ System",
])


# ════════════════════════════════════════════════════════════
#  TAB 1: Grid BOT
# ════════════════════════════════════════════════════════════
with tab_grid:
    gb = live_state.get("grid_bot", {})
    gs = gb.get("summary", {})

    st.markdown("## 🟪 グリッド BOT")
    st.caption(f"VM最終更新: {updated_at}")

    active = gb.get("active", False)
    paused = gb.get("paused_by_gmm", False)
    status_label = "稼働中 🟢" if active else ("GMM停止中 🔵" if paused else "停止中 ⚪")

    g1, g2, g3, g4, g5, g6 = st.columns(6)
    g1.metric("稼働時間",  gs.get("duration", "—") or "—")
    g2.metric("実現P&L",  f"¥{gs.get('realized_pnl', 0):+.1f}")
    g3.metric("総資産",   f"¥{gs.get('equity', 30000):,.1f}", delta=f"{gs.get('equity_pct',0):+.3f}%")
    g4.metric("状態",     status_label)
    g5.metric("取引回数", gs.get("total_trades", 0))
    g6.metric("最大DD",   f"{gs.get('max_drawdown',0):.2f}%")

    g7, g8, g9, g10 = st.columns(4)
    n_trades = gs.get("total_trades", 0)
    g7.metric("勝率",     f"{gs.get('win_rate',0):.1f}%" if n_trades > 0 else "—")
    g8.metric("PF",       f"{gs.get('profit_factor',0):.2f}" if gs.get("profit_factor") else "—")
    g9.metric("平均保有", f"{gs.get('avg_hold_min',0):.1f}分")
    cp = gb.get("center_price")
    g10.metric("センター", f"¥{cp:,.0f}" if cp else "—")

    pause_reason = gb.get("pause_reason", "")
    if pause_reason:
        st.warning(f"停止理由: {pause_reason}")

    st.divider()

    grids = gb.get("grids", [])
    if grids:
        st.markdown("#### グリッドレベル")
        lower = gb.get("lower", 0)
        upper = gb.get("upper", 0)
        st.caption(f"レンジ: ¥{lower:,.0f} ～ ¥{upper:,.0f}")
        for g in grids:
            status = g.get("status", "")
            price  = g.get("price", 0)
            fills  = g.get("fills", 0)
            badge  = ('<span class="badge badge-sell">SELL WAIT</span>' if status == "sell_wait"
                      else '<span class="badge badge-buy">BUY WAIT</span>' if status == "buy_wait"
                      else '<span class="badge badge-wait">DONE</span>')
            st.markdown(
                f'<div class="grid-row"><span style="width:180px;color:#888;font-size:.8rem">¥{price:,.0f}</span>'
                f'{badge}<span style="color:#666;font-size:.78rem;margin-left:8px">約定{fills}回</span></div>',
                unsafe_allow_html=True)

    holdings = gb.get("holdings", [])
    if holdings:
        st.markdown(f"#### 保有ポジション ({len(holdings)}件)")
        for h in holdings:
            st.markdown(
                f'<div class="pos-card">入値: <b>¥{h.get("buy_price",0):,.0f}</b> → '
                f'売り目標: <b style="color:#00c896">¥{h.get("sell_target",0):,.0f}</b> '
                f'| {h.get("size_btc",0):.5f} BTC</div>', unsafe_allow_html=True)

    rcurve = gb.get("realized_curve", [])
    if len(rcurve) > 1:
        st.markdown("#### 実現P&L推移")
        color = "#00c896" if rcurve[-1] >= 0 else "#ff4b4b"
        rgb   = ",".join(str(int(color.lstrip("#")[i:i+2], 16)) for i in (0,2,4))
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=rcurve, mode="lines",
            line=dict(color=color, width=2), fill="tozeroy",
            fillcolor=f"rgba({rgb},0.1)", name="実現P&L"))
        fig.update_layout(height=220, margin=dict(l=0,r=0,t=20,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="#222", zerolinecolor="#444"),
            xaxis=dict(gridcolor="#222"), font=dict(color="#aaa"))
        st.plotly_chart(fig, use_container_width=True)

    trade_log = gb.get("trade_log", [])
    st.markdown(f"#### 約定ログ（最新{len(trade_log)}件）")
    if trade_log:
        st.dataframe(pd.DataFrame(trade_log[::-1]), use_container_width=True, hide_index=True)
    else:
        st.info("まだ約定なし。グリッドレンジ内の価格変動を待機中。")


# ════════════════════════════════════════════════════════════
#  TAB 2: PMR BOT
# ════════════════════════════════════════════════════════════
with tab_pmr:
    pmr = live_state.get("pmr_bot", {})
    ps  = pmr.get("summary", {})

    st.markdown("## 💹 プレミアム平均回帰 BOT")
    st.caption(f"VM最終更新: {updated_at}")

    if not ps:
        st.info("PMR BOT 初期化中... データ蓄積待ち（最低100サンプル必要）")
    else:
        p1, p2, p3, p4, p5, p6 = st.columns(6)
        p1.metric("実現P&L",  f"¥{ps.get('realized_pnl',0):+.1f}")
        p2.metric("含み損益", f"¥{ps.get('unrealized_pnl',0):+.1f}")
        ep_p = ps.get('equity_pct', 0.0)
        p3.metric("総資産",   f"¥{ps.get('equity',30000):,.1f}", delta=f"{ep_p:+.3f}%" if ep_p else None)
        p4.metric("最大DD",   f"{ps.get('max_drawdown',0):.2f}%")
        p5.metric("取引回数", ps.get("total_trades", 0))
        p6.metric("勝率",     f"{ps.get('win_rate',0):.1f}%" if ps.get("total_trades",0) > 0 else "—")

        p7, p8, p9, p10 = st.columns(4)
        p7.metric("平均保有", f"{ps.get('avg_hold_min',0):.1f}分")
        has_pos  = ps.get("has_position", False)
        pos_data = ps.get("position")
        cur_pct  = ps.get("current_pct")
        p8.metric("ポジション", "保有中 🟢" if has_pos else "待機中 ⚪")
        p9.metric("現在%tile", f"{cur_pct:.1f}" if cur_pct is not None else "—")
        if pos_data:
            p9.metric("入りパーセンタイル",  f"{pos_data.get('entry_percentile',0):.1f}%tile")
            p10.metric("現在パーセンタイル", f"{pos_data.get('current_percentile',0):.1f}%tile")

        st.divider()

        if has_pos and pos_data:
            pnl_c = pos_data.get("current_pnl", 0)
            icon  = "🟢" if pnl_c >= 0 else "🔴"
            st.markdown(f"""<div class="pos-card" style="border-color:#4b9eff44;">
<b>オープンポジション</b><br>
入値: ¥{pos_data.get('entry_price',0):,.0f} | 入りプレミアム: {pos_data.get('entry_premium',0):+.4f}% | 保有: {pos_data.get('bars_held',0)}分<br>
現在値: ¥{pos_data.get('current_price',0):,.0f} | 現在プレミアム: {pos_data.get('current_premium',0):+.4f}% ({pos_data.get('current_percentile',0):.1f}%ile)<br>
{icon} 含み損益: <b>¥{pnl_c:+.1f}</b> ({pos_data.get('current_pct',0):+.3f}%)
</div>""", unsafe_allow_html=True)
            st.write("")

        rcurve_p = ps.get("realized_curve", [])
        if len(rcurve_p) > 1:
            st.markdown("#### 実現P&L推移")
            color_p = "#00c896" if rcurve_p[-1] >= 0 else "#ff4b4b"
            rgb_p   = ",".join(str(int(color_p.lstrip("#")[i:i+2], 16)) for i in (0,2,4))
            figp = go.Figure()
            figp.add_trace(go.Scatter(y=rcurve_p, mode="lines",
                line=dict(color=color_p, width=2), fill="tozeroy",
                fillcolor=f"rgba({rgb_p},0.1)", name="実現P&L"))
            figp.update_layout(height=220, margin=dict(l=0,r=0,t=20,b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(gridcolor="#222", zerolinecolor="#444"),
                xaxis=dict(gridcolor="#222"), font=dict(color="#aaa"))
            st.plotly_chart(figp, use_container_width=True)

        tlog_p = ps.get("trade_log", [])
        st.markdown(f"#### 約定ログ（最新{len(tlog_p)}件）")
        if tlog_p:
            st.dataframe(pd.DataFrame(tlog_p[::-1]), use_container_width=True, hide_index=True)
        else:
            st.info("まだ取引なし。プレミアムが20パーセンタイル以下になるとエントリーします。")

        with st.expander("📖 戦略説明"):
            st.markdown("""
| 条件 | 内容 |
|------|------|
| **エントリー** | プレミアムが過去データの **20パーセンタイル以下** |
| **利確** | プレミアムが **50パーセンタイル以上** に回帰 かつ **含み損が-0.1%以内** |
| **SL（価格）** | BTC価格が入値から **-1.5%** 以上下落 |
| **SL（プレミアム）** | プレミアムがさらに **-0.15%** 悪化 |
| **タイムアウト** | **4時間**（240分）経過 |
| **資金** | ¥30,000（グリッドBOTとは独立） |
""")


# ════════════════════════════════════════════════════════════
#  TAB 3: バックテスト
# ════════════════════════════════════════════════════════════
with tab_bt:
    st.markdown("## 🔬 PMR バックテスト")
    st.caption("過去データで複数の設定を一括シミュレーション")

    @st.cache_resource
    def get_iv_sheet():
        try:
            creds = Credentials.from_service_account_info(
                st.secrets["gcp_service_account"], scopes=SCOPES)
            gc = gspread.authorize(creds)
            sh = gc.open_by_key(SHEET_ID)
            return sh.worksheet("iv_data")
        except:
            return None

    @st.cache_data(ttl=3600)
    def load_iv_data():
        ws_iv = get_iv_sheet()
        if ws_iv is None:
            return None
        try:
            # 全件取得（行数上限を避けるためrange指定）
            total = ws_iv.row_count
            rows = ws_iv.get(f"A1:Z{min(total, 50000)}")
            if len(rows) < 2:
                return None
            df = pd.DataFrame(rows[1:], columns=rows[0])
            df["BTC_IV"] = pd.to_numeric(df["BTC_IV"], errors="coerce")
            df["BTC_Spot"] = pd.to_numeric(df["BTC_Spot"], errors="coerce")
            # 0や欠損を除外
            df = df.dropna(subset=["BTC_IV", "BTC_Spot"])
            df = df[df["BTC_Spot"] > 100_000]  # BTC価格が異常な行を除外
            df = df[df["BTC_IV"].abs() < 10]    # プレミアムが±10%超は異常値
            df = df.reset_index(drop=True)
            return df
        except Exception as e:
            return None

    def run_pmr_backtest(df, entry_pct, exit_pct, stop_loss_pct, stop_prem_delta, timeout_bars, capital=30000, tp_min_pnl=None, ai_model=None):
        """PMRロジックをヒストリカルデータでシミュレーション"""
        prems = df["BTC_IV"].values
        prices = df["BTC_Spot"].values
        n = len(prems)
        min_samples = 100

        trades = []
        position = None
        low_dur = 0
        pcts_hist = []

        for i in range(min_samples, n):
            hist = prems[max(0, i-500):i]
            cur_prem = prems[i]
            cur_price = prices[i]
            # 異常値スキップ（price=0やNaN）
            if cur_price <= 100_000 or np.isnan(cur_price) or np.isnan(cur_prem):
                continue
            pct = float((hist < cur_prem).mean() * 100)
            pcts_hist.append(pct)
            low_dur = low_dur + 1 if pct < entry_pct else 0

            if position is None:
                if pct <= entry_pct:
                    if ai_model is not None:
                        try:
                            def _r(n): return (prices[i]-prices[i-n])/prices[i-n] if i>=n and prices[i-n]>0 else 0.0
                            _ps = prems[max(0,i-30):i+1]
                            _pr30 = prices[max(0,i-30):i+1]
                            _pr60 = prices[max(0,i-60):i+1]
                            import numpy as _np2
                            _v30 = float(_np2.std(_np2.diff(_pr30)/_pr30[:-1]))*100 if len(_pr30)>2 else 0.0
                            _v60 = float(_np2.std(_np2.diff(_pr60)/_pr60[:-1]))*100 if len(_pr60)>2 else 0.0
                            _ph = pcts_hist[:]
                            _psl10 = (_ph[-1]-_ph[-11]) if len(_ph)>=11 else 0.0
                            _psl30 = (_ph[-1]-_ph[-31]) if len(_ph)>=31 else 0.0
                            _pmn30 = min(_ph[-30:]) if len(_ph)>=30 else pct
                            try:
                                _ts = df['Timestamp'].iloc[i] if 'Timestamp' in df.columns else None
                                _hr = float(_ts.hour) if _ts is not None else 0.0
                                _iw = float(int(_ts.weekday()>=5)) if _ts is not None else 0.0
                            except Exception:
                                _hr = 0.0; _iw = 0.0
                            _feat = [pct,_r(10),_r(30),_r(60),_r(120),_v30,_v60,cur_prem,
                                np.mean(_ps),np.std(_ps),cur_prem-prems[i-10],cur_prem-prems[i-30],
                                _psl10,_psl30,_pmn30,_hr,_iw,float(low_dur)]
                            _p2 = ai_model["model"].predict_proba([_feat])[0][1]
                            if _p2 < ai_model["threshold"]:
                                continue
                        except Exception:
                            pass
                    position = {
                        "entry_idx": i, "entry_price": cur_price,
                        "entry_prem": cur_prem, "entry_pct": round(pct, 1),
                    }
            else:
                bars_held = i - position["entry_idx"]
                pnl_pct = (cur_price - position["entry_price"]) / position["entry_price"]
                reason = None
                if pct >= exit_pct:
                    if tp_min_pnl is None or pnl_pct >= tp_min_pnl:
                        reason = "TP"
                elif pnl_pct <= stop_loss_pct:
                    reason = "SL_PRICE"
                elif cur_prem <= position["entry_prem"] + stop_prem_delta:
                    reason = "SL_PREM"
                elif bars_held >= timeout_bars:
                    reason = "TIMEOUT"

                if reason:
                    pnl = round(capital * pnl_pct, 1)
                    trades.append({
                        "理由": reason,
                        "入りpct": position["entry_pct"],
                        "出りpct": round(pct, 1),
                        "保有(分)": bars_held,
                        "P&L(¥)": pnl,
                        "勝ち": pnl > 0,
                    })
                    position = None
        low_dur = 0
        pcts_hist = []

        if not trades:
            return None
        df_t = pd.DataFrame(trades)
        n_t = len(df_t)
        win_rate = df_t["勝ち"].mean() * 100
        total_pnl = df_t["P&L(¥)"].sum()
        avg_hold = df_t["保有(分)"].mean()
        wins = df_t[df_t["勝ち"]]["P&L(¥)"].sum()
        losses = abs(df_t[~df_t["勝ち"]]["P&L(¥)"].sum())
        pf = wins / losses if losses > 0 else 99.0
        return {
            "取引数": n_t, "勝率(%)": round(win_rate, 1),
            "総P&L(¥)": round(total_pnl, 1),
            "PF": round(pf, 2),
            "平均保有(分)": round(avg_hold, 1),
            "trades": df_t,
        }

    if st.button("📊 バックテスト実行", type="primary"):
        with st.spinner("iv_dataを読み込み中..."):
            df_iv = load_iv_data()

        if df_iv is None:
            st.error("iv_dataの読み込みに失敗しました")
        else:
            hours = len(df_iv) / 60
            st.success(f"✅ {len(df_iv):,}件のデータ（約{hours:.1f}時間分）でシミュレーション")

            # 複数設定を一括テスト
            configs = [
                {"entry": 15, "exit": 50, "label": "15/50 (現在の設定)", "tp_min_pnl": None},
                {"entry": 20, "exit": 50, "label": "20/50 (緩めエントリー)", "tp_min_pnl": None},
                {"entry": 20, "exit": 60, "label": "20/60", "tp_min_pnl": None},
                {"entry": 15, "exit": 50, "label": "15/50 + TP条件(-0.2%)", "tp_min_pnl": -0.002},
                {"entry": 20, "exit": 50, "label": "20/50 + TP条件(-0.2%)", "tp_min_pnl": -0.002},
                {"entry": 20, "exit": 50, "label": "20/50 + TP条件(-0.1%)", "tp_min_pnl": -0.001},
                {"entry": 15, "exit": 50, "label": "15/50 + TP条件(-0.1%)", "tp_min_pnl": -0.001},
                {"entry": 10, "exit": 50, "label": "10/50 (厳しめエントリー)", "tp_min_pnl": None},
                {"entry": 15, "exit": 40, "label": "15/40 (早めエグジット)", "tp_min_pnl": None},
                {"entry": 10, "exit": 40, "label": "10/40", "tp_min_pnl": None},
            ]

            _bt_ai_model = None
            import pickle as _pkl_mod
            with st.expander('AIフィルター設定'):
                if st.checkbox('AIフィルターを有効化', value=False, key='ai_bt'):
                    _f = st.file_uploader('pmr_filter.pkl', type=['pkl'], key='ai_pkl')
                    if _f:
                        _bt_ai_model = _pkl_mod.load(_f)
                        st.success(f'AIモデル読込済 閾値={_bt_ai_model["threshold"]:.2f}')
                    else:
                        st.caption('VMからpmr_filter.pklをダウンロードしてアップロード')
            results = []
            for cfg in configs:
                r = run_pmr_backtest(
                    df_iv, cfg["entry"], cfg["exit"],
                    stop_loss_pct=-0.015, stop_prem_delta=-0.15, timeout_bars=240,
                    tp_min_pnl=cfg.get("tp_min_pnl"), ai_model=_bt_ai_model
                )
                if r:
                    results.append({
                        "設定": cfg["label"],
                        "取引数": r["取引数"],
                        "勝率(%)": r["勝率(%)"],
                        "総P&L(¥)": r["総P&L(¥)"],
                        "PF": r["PF"],
                        "平均保有(分)": r["平均保有(分)"],
                    })

            if results:
                st.markdown("#### 設定別パフォーマンス比較")
                df_res = pd.DataFrame(results).sort_values("総P&L(¥)", ascending=False)

                def color_pnl(v):
                    if isinstance(v, (int, float)):
                        return "color:#00c896" if v > 0 else "color:#ff4b4b"
                    return ""

                st.dataframe(
                    df_res.style.map(color_pnl, subset=["総P&L(¥)"]),
                    use_container_width=True, hide_index=True
                )

                # 最良設定のP&L推移
                best_label = df_res.iloc[0]["設定"]
                best_cfg = next((c for c in configs if c["label"] == best_label), configs[1])
                best_r = run_pmr_backtest(
                    df_iv, best_cfg["entry"], best_cfg["exit"],
                    stop_loss_pct=-0.015, stop_prem_delta=-0.15, timeout_bars=240,
                    tp_min_pnl=cfg.get("tp_min_pnl"), ai_model=_bt_ai_model
                )
                if best_r:
                    st.markdown(f"#### 最良設定「{df_res.iloc[0]['設定']}」のP&L推移")
                    cumsum = best_r["trades"]["P&L(¥)"].cumsum().tolist()
                    color = "#00c896" if cumsum[-1] >= 0 else "#ff4b4b"
                    rgb = ",".join(str(int(color.lstrip("#")[i:i+2], 16)) for i in (0,2,4))
                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(
                        y=cumsum, mode="lines",
                        line=dict(color=color, width=2),
                        fill="tozeroy", fillcolor=f"rgba({rgb},0.1)",
                        name="累積P&L"
                    ))
                    fig_bt.update_layout(
                        height=250, margin=dict(l=0,r=0,t=20,b=0),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        yaxis=dict(gridcolor="#222", zerolinecolor="#444"),
                        xaxis=dict(gridcolor="#222", title="取引番号"),
                        font=dict(color="#aaa")
                    )
                    st.plotly_chart(fig_bt, use_container_width=True)

                    # 理由別集計
                    reason_summary = best_r["trades"].groupby("理由")["P&L(¥)"].agg(["count","sum"]).rename(columns={"count":"件数","sum":"合計P&L(¥)"})
                    reason_summary["合計P&L(¥)"] = reason_summary["合計P&L(¥)"].round(1)
                    st.markdown("**クローズ理由の内訳**")
                    st.dataframe(reason_summary, use_container_width=True)

                    with st.expander("📋 全取引ログ"):
                        st.dataframe(best_r["trades"], use_container_width=True, hide_index=True)
    else:
        st.info("「バックテスト実行」ボタンを押すと過去4,000件以上のデータで設定を比較します。")


# ════════════════════════════════════════════════════════════
#  TAB 4: Paper Trading
# ════════════════════════════════════════════════════════════
with tab_paper:
    pt  = live_state.get("paper_trading", {})
    st_ = pt.get("stats", {})

    st.markdown("## 💼 ペーパートレード")
    st.caption(f"VM最終更新: {updated_at}")

    cap      = pt.get("capital", 10000)
    cap_init = 10000
    cap_pct  = (cap - cap_init) / cap_init * 100

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("元本",     f"${cap_init:,.0f}")
    m2.metric("現在資産", f"${cap:,.2f}", delta=f"{cap_pct:+.1f}%")
    m3.metric("取引回数", st_.get("total_trades", 0))
    m4.metric("勝率",     f"{st_.get('win_rate',0)}%")
    m5.metric("累計P&L",  f"${st_.get('total_pnl_usd',0):+,.2f}")

    st.divider()
    positions = pt.get("positions", [])
    st.markdown(f"#### オープンポジション ({len(positions)}件)")
    if positions:
        for pos in positions:
            pnl_pct = pos.get("current_pnl_pct", 0) * 100
            icon = "🟢" if pnl_pct >= 0 else "🔴"
            st.markdown(
                f'<div class="pos-card">{icon} <b>{pos.get("strategy_name","?")}</b> | '
                f'状態: {pos.get("state","?")} | エントリー: {pos.get("entry_time","?")} | '
                f'P&L: <b>${pos.get("current_pnl_usd",0):+.2f}</b> ({pnl_pct:+.2f}%)</div>',
                unsafe_allow_html=True)
    else:
        st.info("オープンポジションなし")

    closed = pt.get("closed_trades", [])
    if closed:
        st.markdown(f"#### 決済済みトレード ({len(closed)}件)")
        rows = [{"戦略": t.get("strategy_name","?"), "状態": t.get("state","?"),
                 "エントリー": t.get("entry_time","?"), "決済": t.get("exit_time","—"),
                 "P&L%": f'{t.get("current_pnl_pct",0)*100:+.2f}%',
                 "P&L$": f'${t.get("current_pnl_usd",0):+.2f}',
                 "終了": t.get("status","?")} for t in reversed(closed)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════
#  TAB 4: Market State
# ════════════════════════════════════════════════════════════
with tab_state:
    st.markdown("## 🧠 市場状態")
    st.caption(f"VM最終更新: {updated_at}")

    ms2 = live_state.get("market_state", {})
    sk2 = ms2.get("state", "unknown")
    si2 = MARKET_STATES.get(sk2, MARKET_STATES["unknown"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("現在状態",   si2["label"])
    m2.metric("Confidence", f"{ms2.get('confidence',0)*100:.1f}%")
    m3.metric("Opp Score",  f"{ms2.get('opp_score',0)}/100")
    m4.metric("手法",       ms2.get("method", "—"))

    reasons = ms2.get("opp_reasons", [])
    if reasons:
        st.info("機会理由: " + " / ".join(reasons))

    history = ms2.get("state_history", [])
    if history:
        st.markdown("#### 状態遷移履歴")
        prev = None
        for row in reversed(history):
            cur = row.get("state", "unknown")
            if cur == prev:
                continue
            prev = cur
            si_ = MARKET_STATES.get(cur, MARKET_STATES["unknown"])
            c   = si_["color"]
            st.markdown(
                f'<div style="display:flex;gap:12px;padding:5px 0;border-bottom:1px solid #222;">'
                f'<span style="color:#555;font-size:.78rem;min-width:130px">{row.get("timestamp","?")}</span>'
                f'<span class="badge" style="background:{c}22;color:{c};border:1px solid {c}44">{si_["label"]}</span>'
                f'<span style="color:#666;font-size:.78rem">{row.get("confidence",0)*100:.0f}% · {row.get("method","?")}</span>'
                f'</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  TAB 5: System
# ════════════════════════════════════════════════════════════
with tab_sys:
    st.markdown("## 🗄️ システム")
    st.metric("VM最終更新", updated_at or "—")
    st.markdown(f"[📊 スプレッドシートを開く](https://docs.google.com/spreadsheets/d/{SHEET_ID})")
    ms3 = live_state.get("market_state", {})
    st.markdown(f"""
| 項目 | 値 |
|------|-----|
| データ数 | {ms3.get('n_samples',0):,} |
| GMM | {"稼働中" if ms3.get("gmm_active") else "学習中"} |
| 現在状態 | {ms3.get("state","?")} |
| Confidence | {ms3.get("confidence",0)*100:.1f}% |
""")
    with st.expander("🔍 生データ (live_state JSON)"):
        st.json(live_state)


# ══════════════════════════════════════════════════════════════
#  AUTO REFRESH
# ══════════════════════════════════════════════════════════════
time.sleep(60)
st.rerun()