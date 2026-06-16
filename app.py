"""
╔══════════════════════════════════════════════════════════════╗
║   Crypto Options Market State Engine  v1.1                  ║
║   ─────────────────────────────────────────────────────      ║
║   Step 1  ✅  Data Collection + IV Visualization             ║
║   Step 2  ✅  Anomaly Detection (Z-score + Spike)            ║
║   Step 3  ✅  Market State Classification (GMM + Rules)      ║
║   Step 4  ✅  State-based Strategy Engine + Signal Gen       ║
║   Step 5  ✅  Paper Trading (P&L, Position Tracker)          ║
║   Step 6  ✅  Google Sheets永続ストレージ                     ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time, os, json
from threading import Thread, Lock
from collections import deque

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore as scipy_zscore

import gspread
from google.oauth2.service_account import Credentials

# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════
CSV_FILE             = "iv_data.csv"
SHEET_ID             = "1C4Gd0AqHcMNg-QHhMCGoNzONyse5NMBGCfHjcuVcOKY"
SHEET_NAME           = "iv_data"

ZSCORE_WINDOW        = 20
ZSCORE_THRESHOLD     = 2.0
SPIKE_THRESHOLD_PCT  = 0.03
MAX_ROWS_DISPLAY     = 300
GMM_COMPONENTS       = 4
GMM_MIN_ROWS         = 30

INITIAL_CAPITAL      = 10_000
POSITION_SIZE_PCT    = 0.05
MAX_POSITIONS        = 3

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

MARKET_STATES = {
    "risk_on":  {"label":"Risk-On 🟢",      "color":"#00c896","emoji":"🟢","description":"IV低下・安定。市場は楽観的。",          "strategy":"sell_vol"},
    "panic":    {"label":"Panic 🔴",         "color":"#ff4b4b","emoji":"🔴","description":"IV急騰・スパイク。恐怖の支配。",        "strategy":"buy_vol"},
    "hedging":  {"label":"Hedging Flow 🟡",  "color":"#ffd700","emoji":"🟡","description":"BTC IV > ETH IV 乖離拡大。機関のヘッジ。","strategy":"btc_skew"},
    "squeeze":  {"label":"Vol Squeeze 🔵",   "color":"#4b9eff","emoji":"🔵","description":"IV極端に低水準で圧縮。爆発前夜。",      "strategy":"long_gamma"},
    "unknown":  {"label":"Observing ⚪",     "color":"#888888","emoji":"⚪","description":"データ収集中。状態未確定。",            "strategy":None},
}

STRATEGIES = {
    "sell_vol":   {"name":"Short Straddle (Sell Vol)", "action":"SELL",   "legs":["ATM Call","ATM Put"],           "rationale":"IV > HV → プレミアム売りで時間価値を収集","risk":"Gap riskあり。ストップ: IV +15%","target_pnl_pct":0.30,"stop_pnl_pct":-0.50},
    "buy_vol":    {"name":"Long Straddle (Buy Vol)",   "action":"BUY",    "legs":["ATM Call","ATM Put"],           "rationale":"パニック時のIV急騰でロングガンマが有利", "risk":"時間価値の減衰。数日で決済",      "target_pnl_pct":0.50,"stop_pnl_pct":-0.25},
    "btc_skew":   {"name":"BTC/ETH IV Spread",         "action":"SPREAD", "legs":["Long BTC IV","Short ETH IV"],  "rationale":"Ratio乖離のリバージョンを狙う",          "risk":"相関崩壊リスク",                 "target_pnl_pct":0.20,"stop_pnl_pct":-0.15},
    "long_gamma": {"name":"Long Gamma (Strangle)",     "action":"BUY",    "legs":["OTM Call +10%","OTM Put -10%"],"rationale":"Vol Squeezeの解放で大きなガンマ収益",    "risk":"Squeeze長期化でセータ損失",      "target_pnl_pct":1.00,"stop_pnl_pct":-0.40},
}

_lock = Lock()

# ══════════════════════════════════════════════════════════════
#  GOOGLE SHEETS 接続
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def get_gsheet():
    """Google Sheetsクライアントを返す。失敗したらNone。"""
    try:
        # Streamlit Cloud: st.secrets から読む
        if "gcp_service_account" in st.secrets:
            creds = Credentials.from_service_account_info(
                st.secrets["gcp_service_account"], scopes=SCOPES
            )
        # Codespaces: ローカルのcredentials.jsonから読む
        elif os.path.exists("credentials.json"):
            creds = Credentials.from_service_account_file(
                "credentials.json", scopes=SCOPES
            )
        else:
            return None
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(SHEET_ID)
        # シートがなければ作成
        try:
            ws = sh.worksheet(SHEET_NAME)
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=SHEET_NAME, rows=100000, cols=10)
            ws.append_row(["Timestamp","BTC_IV","ETH_IV","BTC_ETH_Ratio","BTC_Spot","ETH_Spot"])
        return ws
    except Exception as e:
        return None

def sheet_append(ws, row: list):
    """Google Sheetsに1行追記。失敗しても止まらない。"""
    try:
        if ws:
            ws.append_row(row, value_input_option="USER_ENTERED")
    except Exception:
        pass

def load_from_sheet(ws) -> pd.DataFrame:
    """Google Sheetsから全データを読み込む。"""
    try:
        if ws is None:
            return pd.DataFrame()
        records = ws.get_all_values()
        if len(records) < 2:
            return pd.DataFrame()
        df = pd.DataFrame(records[1:], columns=records[0])
        for col in ["BTC_IV","ETH_IV","BTC_ETH_Ratio","BTC_Spot","ETH_Spot"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()

# ══════════════════════════════════════════════════════════════
#  STATE STORE
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def get_state_store():
    return {
        "positions": [], "closed_trades": [],
        "capital": INITIAL_CAPITAL,
        "state_history": deque(maxlen=500),
        "last_signal": None,
        "gmm_model": None, "gmm_scaler": None, "gmm_label_map": {},
    }

# ══════════════════════════════════════════════════════════════
#  BACKGROUND BOT
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def launch_background_bot():
    ws = get_gsheet()

    if not os.path.exists(CSV_FILE):
        pd.DataFrame(columns=["Timestamp","BTC_IV","ETH_IV","BTC_ETH_Ratio","BTC_Spot","ETH_Spot"]
                     ).to_csv(CSV_FILE, index=False)

    def get_dvol(symbol):
        url = f"https://www.deribit.com/api/v2/public/get_index_price?index_name={symbol.lower()}dvol_usdc"
        try:
            r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=5)
            if r.status_code == 200:
                return float(r.json()["result"]["index_price"])
        except Exception:
            return None

    def get_spot(symbol):
        url = f"https://www.deribit.com/api/v2/public/get_index_price?index_name={symbol.lower()}_usd"
        try:
            r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=5)
            if r.status_code == 200:
                return float(r.json()["result"]["index_price"])
        except Exception:
            return None

    def bot_loop():
        while True:
            try:
                btc_iv = get_dvol("btc"); time.sleep(0.5)
                eth_iv = get_dvol("eth"); time.sleep(0.5)
                btc_sp = get_spot("btc"); time.sleep(0.5)
                eth_sp = get_spot("eth")

                if btc_iv and eth_iv:
                    ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ratio = round(btc_iv / eth_iv, 6)
                    bsp   = round(btc_sp, 2) if btc_sp else ""
                    esp   = round(eth_sp, 2) if eth_sp else ""

                    # CSVに保存（ローカルバックアップ）
                    row_df = pd.DataFrame([[ts, round(btc_iv,4), round(eth_iv,4), ratio, bsp, esp]])
                    with _lock:
                        row_df.to_csv(CSV_FILE, mode="a", header=False, index=False)

                    # Google Sheetsに保存（永続ストレージ）
                    sheet_append(ws, [ts, round(btc_iv,4), round(eth_iv,4), ratio, bsp, esp])

            except Exception:
                pass
            time.sleep(58)

    t = Thread(target=bot_loop, daemon=True)
    t.start()
    return t

# ══════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, prefix in [("BTC_IV","BTC"),("ETH_IV","ETH")]:
        roll = df[col].rolling(ZSCORE_WINDOW, min_periods=3)
        df[f"{prefix}_Z"]     = (df[col] - roll.mean()) / roll.std().replace(0, np.nan)
        df[f"{prefix}_Spike"] = df[col].pct_change().abs() > SPIKE_THRESHOLD_PCT
        df[f"{prefix}_Mom5"]  = df[col].pct_change(5)
        rv = df[col].rolling(10, min_periods=3).std()
        df[f"{prefix}_RV10"]  = rv / df[col] * 100

    roll_r = df["BTC_ETH_Ratio"].rolling(ZSCORE_WINDOW, min_periods=3)
    df["Ratio_Z"]    = (df["BTC_ETH_Ratio"] - roll_r.mean()) / roll_r.std().replace(0, np.nan)
    df["Ratio_Mom5"] = df["BTC_ETH_Ratio"].pct_change(5)

    df["Anomaly"] = (
        (df["BTC_Z"].abs() > ZSCORE_THRESHOLD) |
        (df["ETH_Z"].abs() > ZSCORE_THRESHOLD) |
        (df["Ratio_Z"].abs() > ZSCORE_THRESHOLD) |
        df["BTC_Spike"] | df["ETH_Spike"]
    )

    reasons = []
    for _, row in df.iterrows():
        r = []
        bz = row.get("BTC_Z") or 0
        ez = row.get("ETH_Z") or 0
        rz = row.get("Ratio_Z") or 0
        if abs(bz) > ZSCORE_THRESHOLD: r.append(f"BTC Z={bz:.1f}")
        if abs(ez) > ZSCORE_THRESHOLD: r.append(f"ETH Z={ez:.1f}")
        if abs(rz) > ZSCORE_THRESHOLD: r.append(f"Ratio Z={rz:.1f}")
        if row.get("BTC_Spike"): r.append("BTC Spike")
        if row.get("ETH_Spike"): r.append("ETH Spike")
        reasons.append(", ".join(r) if r else "—")
    df["Anomaly_Reason"] = reasons
    return df

# ══════════════════════════════════════════════════════════════
#  GMM STATE CLASSIFIER
# ══════════════════════════════════════════════════════════════
FEATURE_COLS = ["BTC_Z","ETH_Z","Ratio_Z","BTC_RV10","ETH_RV10","Ratio_Mom5"]

def _map_gmm_cluster_to_state(gmm, scaler) -> dict:
    means = scaler.inverse_transform(gmm.means_)
    label_map = {}
    for i, m in enumerate(means):
        btc_z, eth_z, ratio_z, btc_rv, eth_rv, ratio_mom = m
        if btc_z > 1.2 and eth_z > 1.2:
            label_map[i] = "panic"
        elif btc_z < -0.8 and eth_z < -0.8:
            label_map[i] = "squeeze" if btc_rv < 1.5 else "risk_on"
        elif ratio_z > 1.0 or ratio_mom > 0.02:
            label_map[i] = "hedging"
        else:
            label_map[i] = "risk_on"
    return label_map

def fit_or_update_gmm(store, df):
    feat = df[FEATURE_COLS].dropna()
    if len(feat) < GMM_MIN_ROWS:
        return
    scaler = StandardScaler()
    X = scaler.fit_transform(feat)
    gmm = GaussianMixture(n_components=GMM_COMPONENTS, covariance_type="full", random_state=42, max_iter=200)
    gmm.fit(X)
    store["gmm_model"]    = gmm
    store["gmm_scaler"]   = scaler
    store["gmm_label_map"]= _map_gmm_cluster_to_state(gmm, scaler)

def classify_state(store, df):
    if len(df) < 5:
        return "unknown", 0.0, "insufficient_data"

    latest = df.iloc[-1]
    gmm    = store.get("gmm_model")
    scaler = store.get("gmm_scaler")
    lmap   = store.get("gmm_label_map", {})

    if gmm and scaler:
        row_feat = pd.to_numeric(
            latest[FEATURE_COLS],
            errors="coerce"
        ).astype(float).values

        if not np.isnan(row_feat).any():
            x = scaler.transform([row_feat])
            probs = gmm.predict_proba(x)[0]
            cluster = int(np.argmax(probs))
            return lmap.get(cluster, "unknown"), float(probs[cluster]), "GMM"

    btc_z  = latest.get("BTC_Z") or 0
    eth_z  = latest.get("ETH_Z") or 0
    rz     = latest.get("Ratio_Z") or 0
    btc_rv = latest.get("BTC_RV10") or 0

    if btc_z > ZSCORE_THRESHOLD and eth_z > ZSCORE_THRESHOLD:
        return "panic", 0.85, "Rule"

    if btc_z < -1.5 and eth_z < -1.5 and btc_rv < 2:
        return "squeeze", 0.80, "Rule"

    if rz > ZSCORE_THRESHOLD:
        return "hedging", 0.75, "Rule"

    if btc_z < 0 and eth_z < 0:
        return "risk_on", 0.70, "Rule"

    return "unknown", 0.50, "Rule"

def get_state_history_df(store):
    if not store["state_history"]:
        return pd.DataFrame(columns=["Timestamp","State","Confidence","Method"])
    return pd.DataFrame(list(store["state_history"]), columns=["Timestamp","State","Confidence","Method"])

# ══════════════════════════════════════════════════════════════
#  STRATEGY ENGINE
# ══════════════════════════════════════════════════════════════
def generate_signal(state_key, confidence, df, store):
    if state_key == "unknown" or confidence < 0.65: return None
    strat_key = MARKET_STATES[state_key]["strategy"]
    if strat_key is None: return None
    if strat_key in [p["strategy"] for p in store["positions"]]: return None
    latest = df.iloc[-1]
    strat  = STRATEGIES[strat_key]
    return {
        "id": f"{state_key[:3].upper()}-{datetime.now().strftime('%H%M%S')}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "state": state_key, "strategy": strat_key,
        "strategy_name": strat["name"], "action": strat["action"],
        "legs": strat["legs"], "rationale": strat["rationale"], "risk": strat["risk"],
        "confidence": confidence,
        "btc_iv_at_signal": latest.get("BTC_IV", 0),
        "eth_iv_at_signal": latest.get("ETH_IV", 0),
        "target_pnl_pct": strat["target_pnl_pct"],
        "stop_pnl_pct":   strat["stop_pnl_pct"],
    }

# ══════════════════════════════════════════════════════════════
#  PAPER TRADING
# ══════════════════════════════════════════════════════════════
def open_paper_trade(signal, store):
    if len(store["positions"]) >= MAX_POSITIONS: return None
    trade = {**signal, "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             "size_usd": round(store["capital"] * POSITION_SIZE_PCT, 2),
             "current_pnl_usd": 0.0, "current_pnl_pct": 0.0, "status": "OPEN"}
    store["positions"].append(trade)
    return trade

def update_paper_trades(store, df):
    if df.empty or len(df) < 2: return
    latest = df.iloc[-1]; prev = df.iloc[-2]
    btc_chg = (latest["BTC_IV"] - prev["BTC_IV"]) / prev["BTC_IV"]
    eth_chg = (latest["ETH_IV"] - prev["ETH_IV"]) / prev["ETH_IV"]
    avg_chg = (btc_chg + eth_chg) / 2
    closed  = []
    for pos in store["positions"]:
        s = pos["strategy"]
        if s == "sell_vol":   pnl_delta = -avg_chg * 2
        elif s == "buy_vol":  pnl_delta =  avg_chg * 2
        elif s == "btc_skew":
            rc = (latest["BTC_ETH_Ratio"] - prev["BTC_ETH_Ratio"]) / prev["BTC_ETH_Ratio"]
            pnl_delta = -rc * 3
        elif s == "long_gamma": pnl_delta = abs(avg_chg) * 3 - 0.001
        else: pnl_delta = 0.0
        pos["current_pnl_pct"] = pos.get("current_pnl_pct", 0.0) + pnl_delta
        pos["current_pnl_usd"] = round(pos["size_usd"] * pos["current_pnl_pct"], 2)
        if pos["current_pnl_pct"] >= pos["target_pnl_pct"]:  pos["status"] = "CLOSED_TP"; closed.append(pos)
        elif pos["current_pnl_pct"] <= pos["stop_pnl_pct"]:  pos["status"] = "CLOSED_SL"; closed.append(pos)
    for pos in closed:
        pos["exit_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        store["capital"] += pos["current_pnl_usd"]
        store["positions"].remove(pos)
        store["closed_trades"].append(pos)

def get_portfolio_stats(store):
    closed = store["closed_trades"]
    if not closed:
        return {"total_trades":0,"win_rate":0.0,"total_pnl_usd":0.0,"avg_pnl_pct":0.0,"best_trade":0.0,"worst_trade":0.0}
    pnls    = [t["current_pnl_usd"] for t in closed]
    pnl_pct = [t["current_pnl_pct"] for t in closed]
    wins    = [p for p in pnls if p > 0]
    return {"total_trades":len(closed),"win_rate":round(len(wins)/len(closed)*100,1),
            "total_pnl_usd":round(sum(pnls),2),"avg_pnl_pct":round(np.mean(pnl_pct)*100,2),
            "best_trade":round(max(pnl_pct)*100,2),"worst_trade":round(min(pnl_pct)*100,2)}

# ══════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="Market State Engine", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0b0d12; }
  [data-testid="stHeader"]           { background: transparent; }
  [data-testid="stSidebar"]          { background: #0f1117; }
  .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
  h1,h2,h3 { color: #dde0ea; }
  .state-pill { display:inline-block; padding:.3em .9em; border-radius:99px; font-size:1rem; font-weight:700; letter-spacing:.04em; }
  .card { background:#141720; border:1px solid #1e2230; border-radius:10px; padding:1rem 1.2rem; margin-bottom:.6rem; }
  .signal-card { background:#0d1620; border:1px solid #1a3050; border-radius:10px; padding:.9rem 1.1rem; margin-bottom:.6rem; }
  .label { font-size:.7rem; color:#666; text-transform:uppercase; letter-spacing:.08em; margin-bottom:2px; }
  .roadmap { background:#111418; border-left:3px solid #222; border-radius:5px; padding:.35rem .8rem; margin:.2rem 0; font-size:.82rem; color:#555; }
  .roadmap.done   { border-left-color:#00c896; color:#00c896; }
  .roadmap.active { border-left-color:#4b9eff; color:#c8d8ff; }
  .tag { display:inline-block; padding:.15em .6em; border-radius:4px; font-size:.75rem; font-weight:600; margin-right:.3rem; }
  .green { background:#00c89622; color:#00c896; }
  .red   { background:#ff4b4b22; color:#ff4b4b; }
  .blue  { background:#4b9eff22; color:#4b9eff; }
  .gold  { background:#ffd70022; color:#ffd700; }
</style>
""", unsafe_allow_html=True)

# ── INIT ──────────────────────────────────────────────────────
launch_background_bot()
store = get_state_store()
ws    = get_gsheet()
sheet_ok = ws is not None

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 設定")
    auto_trade = st.toggle("🤖 自動シグナル → ペーパートレード", value=True)
    zscore_thr = st.slider("Z-score 閾値", 1.0, 3.5, ZSCORE_THRESHOLD, 0.1)
    spike_thr  = st.slider("Spike 閾値 (%)", 1, 10, int(SPIKE_THRESHOLD_PCT*100), 1)
    gmm_window = st.slider("GMM 学習データ数", 30, 300, 100, 10)
    display_n  = st.slider("チャート表示件数", 50, 500, MAX_ROWS_DISPLAY, 50)

    st.markdown("---")
    st.markdown("### 📡 データソース")
    if sheet_ok:
        st.success("✅ Google Sheets 接続中")
    else:
        st.warning("⚠️ CSV モード（ローカルのみ）")

    st.markdown("---")
    st.markdown("### 🗺️ 進捗")
    for label, cls in [
        ("✅ Step 1  Data Collection",  "done"),
        ("✅ Step 2  Anomaly Detection","done"),
        ("✅ Step 3  Market State AI",  "done"),
        ("✅ Step 4  Strategy Engine",  "done"),
        ("✅ Step 5  Paper Trading",    "done"),
        ("✅ Step 6  Google Sheets",    "done"),
        ("⚪ Step 7  Live Trading",     ""),
    ]:
        st.markdown(f'<div class="roadmap {cls}">{label}</div>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ 全トレードリセット", type="secondary"):
        store["positions"] = []; store["closed_trades"] = []; store["capital"] = INITIAL_CAPITAL
        st.success("リセット完了")

# ── DATA LOAD ─────────────────────────────────────────────────
# Google Sheetsから読む（永続データ）→ なければCSV
if sheet_ok:
    raw = load_from_sheet(ws)
    if raw.empty and os.path.exists(CSV_FILE):
        raw = pd.read_csv(CSV_FILE, names=["Timestamp","BTC_IV","ETH_IV","BTC_ETH_Ratio","BTC_Spot","ETH_Spot"], skiprows=1)
else:
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) < 10:
        st.info("🔄 初回データ取得中... 約1分後に更新してください。")
        time.sleep(10); st.rerun()
    raw = pd.read_csv(CSV_FILE, names=["Timestamp","BTC_IV","ETH_IV","BTC_ETH_Ratio","BTC_Spot","ETH_Spot"], skiprows=1)

if raw.empty:
    st.info("🔄 データがまだありません。しばらくお待ちください。")
    time.sleep(10); st.rerun()

df  = compute_features(raw.copy())
df  = df.replace([np.inf, -np.inf], np.nan)

fit_or_update_gmm(store, df.tail(gmm_window))
state_key, confidence, method = classify_state(store, df)

store["state_history"].append((datetime.now().strftime("%Y-%m-%d %H:%M:%S"), state_key, round(confidence,3), method))

signal = generate_signal(state_key, confidence, df, store)
if auto_trade and signal:
    open_paper_trade(signal, store)
    store["last_signal"] = signal

update_paper_trades(store, df)

state_info    = MARKET_STATES[state_key]
latest        = df.iloc[-1]
anomaly_count = int(df["Anomaly"].fillna(False).sum())
df_disp       = df.tail(display_n)
chart_idx     = df_disp.set_index("Timestamp")
stats         = get_portfolio_stats(store)

# ── HEADER ────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([2, 1])
with col_h1:
    st.markdown("# 🛰️ Crypto Options Market State Engine")
    src_label = "Google Sheets" if sheet_ok else "CSV Local"
    st.caption(f"Deribit DVOL · 1-min · {len(df)} rows · Source: {src_label} · Last: {latest['Timestamp']}")
with col_h2:
    color    = state_info["color"]
    conf_bar = int(confidence * 10)
    st.markdown(
        f'<div class="card" style="border-color:{color}44;">'
        f'<div class="label">CURRENT STATE ({method})</div>'
        f'<span class="state-pill" style="background:{color}22;color:{color};border:1px solid {color}66;">{state_info["label"]}</span>'
        f'<div style="margin-top:.4rem;font-size:.8rem;color:#888;">Confidence: <b style="color:{color}">{confidence*100:.0f}%</b> '
        f'{"█"*conf_bar}{"░"*(10-conf_bar)}</div>'
        f'<div style="font-size:.75rem;color:#666;margin-top:.3rem;">{state_info["description"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── KPI ROW ───────────────────────────────────────────────────
def delta_val(series):
    if len(series) < 2: return None
    return round(float(series.iloc[-1] - series.iloc[-2]), 4)

k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("BTC ATM IV",    f"{latest['BTC_IV']:.2f}",    delta=delta_val(df["BTC_IV"]))
k2.metric("ETH ATM IV",    f"{latest['ETH_IV']:.2f}",    delta=delta_val(df["ETH_IV"]))
k3.metric("BTC/ETH Ratio", f"{latest['BTC_ETH_Ratio']:.4f}", delta=delta_val(df["BTC_ETH_Ratio"]))
k4.metric("BTC Z-score",   f"{(latest.get('BTC_Z') or 0):.2f}")
k5.metric("Anomalies",     f"{anomaly_count}")
pnl_str = f"+${stats['total_pnl_usd']:.0f}" if stats['total_pnl_usd'] >= 0 else f"-${abs(stats['total_pnl_usd']):.0f}"
k6.metric("Paper P&L",     pnl_str, delta=f"{stats['win_rate']}% win rate" if stats["total_trades"] else None)

st.divider()

# ── TABS ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 IV Monitor","🚨 Anomaly","🧠 Market State","⚡ Strategy","💼 Paper Trading","🗄️ Sheets Data"
])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### BTC / ETH ATM IV")
        st.line_chart(chart_idx[["BTC_IV","ETH_IV"]], height=260)
    with c2:
        st.markdown("#### BTC/ETH IV Ratio")
        st.line_chart(chart_idx[["BTC_ETH_Ratio"]], color="#ffd700", height=260)
    c3, c4 = st.columns(2)
    with c3:
        spot_btc = chart_idx[["BTC_Spot"]].dropna()
        if not spot_btc.empty:
            st.markdown("#### BTC Spot"); st.line_chart(spot_btc, color="#f7931a", height=200)
    with c4:
        spot_eth = chart_idx[["ETH_Spot"]].dropna()
        if not spot_eth.empty:
            st.markdown("#### ETH Spot"); st.line_chart(spot_eth, color="#627eea", height=200)
    rv_cols = [c for c in ["BTC_RV10","ETH_RV10"] if c in chart_idx.columns]
    rv_data = chart_idx[rv_cols].dropna()
    if not rv_data.empty:
        st.markdown("#### Realized Vol Proxy (10-period)"); st.line_chart(rv_data, height=200)

with tab2:
    z_cols = [c for c in ["BTC_Z","ETH_Z","Ratio_Z"] if c in chart_idx.columns]
    z_data = chart_idx[z_cols].dropna()
    if not z_data.empty:
        c5, c6 = st.columns(2)
        with c5:
            st.markdown("#### IV Z-scores")
            st.line_chart(z_data[["BTC_Z","ETH_Z"]], height=230)
        with c6:
            st.markdown("#### Ratio Z-score")
            st.line_chart(z_data[["Ratio_Z"]], color="#ffd700", height=230)
    anomalies = df[df["Anomaly"].fillna(False)].copy()
    if anomalies.empty:
        st.success("✅ 現在、異常は検出されていません。")
    else:
        st.error(f"⚠️ {len(anomalies)} 件の異常を検出")
        disp_cols = ["Timestamp","BTC_IV","ETH_IV","BTC_ETH_Ratio","BTC_Z","ETH_Z","Ratio_Z","Anomaly_Reason"]
        existing  = [c for c in disp_cols if c in anomalies.columns]
        st.dataframe(anomalies[existing].tail(60)[::-1], use_container_width=True, height=280)

with tab3:
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        st.markdown("#### 状態定義")
        for key, info in MARKET_STATES.items():
            if key == "unknown": continue
            is_current  = (key == state_key)
            border      = f"border:1px solid {info['color']}88;" if is_current else ""
            curr_badge  = '<br><b style="color:#4b9eff;font-size:.7rem">← 現在</b>' if is_current else ""
            st.markdown(
                f'<div class="card" style="{border}margin-bottom:.4rem;">'
                f'<span class="state-pill" style="background:{info["color"]}22;color:{info["color"]};border:1px solid {info["color"]}44;font-size:.85rem;">{info["label"]}</span>'
                f'{curr_badge}'
                f'<div style="font-size:.75rem;color:#777;margin-top:.3rem;">{info["description"]}</div>'
                f'</div>', unsafe_allow_html=True)
        gmm_status = "✅ GMM稼働中" if store.get("gmm_model") else "⏳ データ蓄積中..."
        st.info(f"分類器: **{method}** · {gmm_status}")
    with col_s2:
        st.markdown("#### 状態遷移履歴")
        hist_df = get_state_history_df(store)
        if not hist_df.empty:
            state_int_map = {"risk_on":1,"hedging":2,"squeeze":3,"panic":4,"unknown":0}
            hist_df["State_Int"] = hist_df["State"].map(state_int_map).fillna(0)
            st.line_chart(hist_df.set_index("Timestamp")[["State_Int","Confidence"]], height=220)
            hist_df["Prev"] = hist_df["State"].shift(1)
            changes = hist_df[hist_df["State"] != hist_df["Prev"]].tail(20)
            for _, row in changes[::-1].iterrows():
                info  = MARKET_STATES.get(row["State"], MARKET_STATES["unknown"])
                color = info["color"]
                st.markdown(
                    f'<div style="display:flex;gap:.8rem;align-items:center;padding:.25rem 0;border-bottom:1px solid #1e2230;">'
                    f'<span style="color:#555;font-size:.75rem;min-width:130px">{row["Timestamp"]}</span>'
                    f'<span class="state-pill" style="background:{color}22;color:{color};border:1px solid {color}44;font-size:.75rem;padding:.15em .6em">{info["label"]}</span>'
                    f'<span style="color:#777;font-size:.75rem">conf: {row["Confidence"]*100:.0f}% · {row["Method"]}</span>'
                    f'</div>', unsafe_allow_html=True)
        else:
            st.info("状態履歴を蓄積中...")
    if store.get("gmm_model"):
        with st.expander("🔬 GMM クラスター詳細"):
            gmm    = store["gmm_model"]; scaler = store["gmm_scaler"]; lmap = store["gmm_label_map"]
            means  = scaler.inverse_transform(gmm.means_)
            rows   = [{"Cluster":i,"→ State":MARKET_STATES[lmap.get(i,"unknown")]["label"],
                       "Weight %":f"{w*100:.1f}%","BTC_Z":f"{m[0]:.2f}","ETH_Z":f"{m[1]:.2f}","Ratio_Z":f"{m[2]:.2f}"}
                      for i,(m,w) in enumerate(zip(means, gmm.weights_))]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

with tab4:
    strat_key = state_info.get("strategy")
    if strat_key and strat_key in STRATEGIES:
        strat = STRATEGIES[strat_key]; color = state_info["color"]
        st.markdown(
            f'<div class="signal-card" style="border-color:{color}55;">'
            f'<div style="display:flex;justify-content:space-between;align-items:start;">'
            f'<div><div class="label">STRATEGY</div><div style="font-size:1.1rem;font-weight:700;color:{color}">{strat["name"]}</div></div>'
            f'<span class="state-pill" style="background:{color}22;color:{color};border:1px solid {color}44;font-size:.85rem">{strat["action"]}</span>'
            f'</div>'
            f'<div style="margin-top:.7rem;display:grid;grid-template-columns:1fr 1fr;gap:.5rem;">'
            f'<div><div class="label">Legs</div><div style="color:#c8cad4;font-size:.85rem">{" · ".join(strat["legs"])}</div></div>'
            f'<div><div class="label">Target / Stop</div><span class="tag green">TP +{strat["target_pnl_pct"]*100:.0f}%</span><span class="tag red">SL {strat["stop_pnl_pct"]*100:.0f}%</span></div>'
            f'<div style="grid-column:span 2"><div class="label">Rationale</div><div style="color:#a0a8b8;font-size:.82rem">{strat["rationale"]}</div></div>'
            f'<div style="grid-column:span 2"><div class="label">Risk</div><div style="color:#886060;font-size:.82rem">{strat["risk"]}</div></div>'
            f'</div></div>', unsafe_allow_html=True)
    else:
        st.info("現在、明確な戦略シグナルはありません。")
    st.markdown("#### 全戦略マップ")
    for sk, sv in STRATEGIES.items():
        states_for = [k for k,v in MARKET_STATES.items() if v.get("strategy")==sk]
        sc = MARKET_STATES[states_for[0]]["color"] if states_for else "#888"
        is_active = (sk == strat_key)
        border = f"border:1px solid {sc}88;" if is_active else ""
        st.markdown(
            f'<div class="card" style="{border}margin-bottom:.3rem;">'
            f'<div style="display:flex;justify-content:space-between;"><b style="color:{sc}">{sv["name"]}</b>'
            f'<span class="tag" style="background:{sc}22;color:{sc}">{sv["action"]}</span></div>'
            f'<div style="font-size:.78rem;color:#888;margin-top:.3rem">{sv["rationale"]}</div>'
            f'</div>', unsafe_allow_html=True)

with tab5:
    capital_pct = (store["capital"] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    pm1,pm2,pm3,pm4 = st.columns(4)
    pm1.metric("元本",       f"${INITIAL_CAPITAL:,.0f}")
    pm2.metric("現在の資産", f"${store['capital']:,.0f}", delta=f"{capital_pct:+.1f}%")
    pm3.metric("総トレード", stats["total_trades"])
    pm4.metric("勝率",       f"{stats['win_rate']}%")
    pm5,pm6,pm7,pm8 = st.columns(4)
    pm5.metric("累計P&L",    f"${stats['total_pnl_usd']:+,.2f}")
    pm6.metric("平均P&L%",   f"{stats['avg_pnl_pct']:+.2f}%")
    pm7.metric("最良",       f"+{stats['best_trade']:.1f}%")
    pm8.metric("最悪",       f"{stats['worst_trade']:.1f}%")
    st.divider()
    st.markdown(f"#### 🔓 オープンポジション ({len(store['positions'])} / {MAX_POSITIONS})")
    if not store["positions"]:
        st.info("現在、オープンポジションはありません。")
    else:
        for pos in store["positions"]:
            pnl = pos["current_pnl_pct"]; pc = "#00c896" if pnl >= 0 else "#ff4b4b"
            si  = MARKET_STATES.get(pos["state"], MARKET_STATES["unknown"]); c = si["color"]
            col_a, col_b = st.columns([3,1])
            with col_a:
                st.markdown(
                    f'<div class="signal-card" style="border-color:{c}55;">'
                    f'<div style="display:flex;justify-content:space-between;">'
                    f'<div><span style="color:#888;font-size:.7rem">{pos["id"]}</span><br>'
                    f'<b style="color:{c}">{pos["strategy_name"]}</b></div>'
                    f'<div style="text-align:right"><div class="label">P&L</div>'
                    f'<span style="font-size:1.2rem;font-weight:700;color:{pc}">{pnl*100:+.2f}%</span><br>'
                    f'<span style="font-size:.8rem;color:{pc}">${pos["current_pnl_usd"]:+.2f}</span></div></div>'
                    f'<div style="font-size:.75rem;color:#666;margin-top:.4rem">'
                    f'Size: ${pos["size_usd"]:,.0f} · Entry: {pos["entry_time"]}</div>'
                    f'</div>', unsafe_allow_html=True)
            with col_b:
                if st.button("手動決済", key=f"close_{pos['id']}"):
                    pos["status"] = "CLOSED_MANUAL"; pos["exit_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    store["capital"] += pos["current_pnl_usd"]
                    store["closed_trades"].append(pos); store["positions"].remove(pos); st.rerun()
    st.divider()
    st.markdown(f"#### 📋 決済済みトレード ({len(store['closed_trades'])} 件)")
    if store["closed_trades"]:
        closed_rows = [{"ID":t["id"],"戦略":t["strategy_name"],
                        "状態":MARKET_STATES.get(t["state"],{}).get("label","?"),
                        "エントリー":t["entry_time"],"決済":t.get("exit_time","—"),
                        "P&L%":f'{t.get("current_pnl_pct",0)*100:+.2f}%',
                        "P&L$":f'${t["current_pnl_usd"]:+.2f}',"終了":t.get("status","?")}
                       for t in reversed(store["closed_trades"])]
        st.dataframe(pd.DataFrame(closed_rows), use_container_width=True, height=300)
        equity = [INITIAL_CAPITAL]
        for t in store["closed_trades"]: equity.append(equity[-1] + t.get("current_pnl_usd",0))
        st.markdown("#### Equity Curve")
        st.line_chart(pd.DataFrame({"Equity": equity}), color="#00c896", height=200)
    else:
        st.info("まだ決済済みトレードはありません。")

with tab6:
    st.markdown("#### 🗄️ Google Sheets データ")
    if sheet_ok:
        st.success(f"✅ 接続中 · {len(df)} 行蓄積済み")
        st.markdown(f"[📊 スプレッドシートを開く](https://docs.google.com/spreadsheets/d/{SHEET_ID})")
        st.dataframe(df.tail(100)[::-1], use_container_width=True, height=400)
        st.download_button("⬇️ 全データ CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"iv_full_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")
    else:
        st.warning("Google Sheets未接続。CSVモードで動作中。")
        st.markdown("**Streamlit CloudでSecretsを設定すると接続できます。**")
        st.dataframe(df.tail(100)[::-1], use_container_width=True, height=400)

with st.expander("🗂️ 生データ"):
    st.dataframe(df.tail(200)[::-1], use_container_width=True, height=300)

# ── AUTO REFRESH ──────────────────────────────────────────────
time.sleep(10)
st.rerun()