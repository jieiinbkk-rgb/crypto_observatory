"""
╔══════════════════════════════════════════════════════════════════════╗
║   Crypto Volatility Research Platform  v2.0                         ║
║   ──────────────────────────────────────────────────────────────     ║
║   Step 1  ✅  Data Collection + IV Visualization                     ║
║   Step 2  ✅  Anomaly Detection (Z-score + Spike)                    ║
║   Step 3  ✅  Market State Classification (GMM + Rules)              ║
║   Step 4  ✅  Strategy Engine + Signal Gen                           ║
║   Step 5  ✅  Paper Trading (P&L, Position Tracker)                  ║
║   Step 6  ✅  Google Sheets Persistent Storage                       ║
║   Step 7  ✅  Opportunity Score Engine                               ║
║   Step 8  ✅  IV-RV Spread Monitor                                   ║
║   Step 9  ✅  Volatility Regime Analyzer                             ║
║   Step 10 ✅  State Transition Matrix                                ║
║   Step 11 ✅  Signal Database (Google Sheets)                        ║
║   Step 12 ✅  Backtest Engine                                        ║
║   Step 13 ✅  Research Dashboard                                     ║
║   Step 14 ✅  Telegram Alert System                                  ║
║   Step 15 ✅  Data Quality Monitor                                   ║
║   Step 16 ✅  Future Trading Layer (Design)                          ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time, os, json, math
from threading import Thread, Lock
from collections import deque
from typing import Optional, Dict, List, Tuple, Any

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore as scipy_zscore
from scipy import stats

import gspread
from google.oauth2.service_account import Credentials

# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════
APP_VERSION          = "2.0.0"
CSV_FILE             = "iv_data.csv"
SHEET_ID             = "1C4Gd0AqHcMNg-QHhMCGoNzONyse5NMBGCfHjcuVcOKY"
SHEET_NAME           = "iv_data"
SIGNAL_SHEET_NAME    = "signal_history"

ZSCORE_WINDOW        = 20
ZSCORE_THRESHOLD     = 2.0
SPIKE_THRESHOLD_PCT  = 0.03
MAX_ROWS_DISPLAY     = 300
GMM_COMPONENTS       = 4
GMM_MIN_ROWS         = 30

INITIAL_CAPITAL      = 10_000
POSITION_SIZE_PCT    = 0.05
MAX_POSITIONS        = 3

# IV-RV Spread alert threshold
IV_RV_SPREAD_ALERT   = 20.0

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

VOL_REGIMES = {
    "low":     {"label":"Low Vol 🟦",     "color":"#4b9eff","description":"IV < 20th percentile","min_z":-99, "max_z":-1.0},
    "normal":  {"label":"Normal Vol 🟩",  "color":"#00c896","description":"IV 20-80th percentile","min_z":-1.0,"max_z":1.0},
    "high":    {"label":"High Vol 🟧",    "color":"#ff9900","description":"IV > 80th percentile","min_z":1.0, "max_z":2.0},
    "extreme": {"label":"Extreme Vol 🟥", "color":"#ff4b4b","description":"IV > 95th percentile","min_z":2.0, "max_z":99},
}

_lock = Lock()

# ══════════════════════════════════════════════════════════════
#  FUTURE TRADING LAYER (Design Only – Not Connected)
# ══════════════════════════════════════════════════════════════
class TradingInterface:
    """
    Abstract trading interface for future live trading integration.
    Currently a stub/design layer. All methods raise NotImplementedError.
    Connect to Deribit Live Trading API by implementing these methods.

    Future implementation steps:
      1. Obtain Deribit API key (read + trade permissions)
      2. Set DERIBIT_API_KEY and DERIBIT_API_SECRET in Streamlit secrets
      3. Replace stub methods with deribit_api_v2 calls
      4. Implement order management, risk checks, and position sizing
    """
    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = True):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.testnet    = testnet
        self.base_url   = "https://test.deribit.com" if testnet else "https://www.deribit.com"
        self.connected  = False

    def connect(self) -> bool:
        """Authenticate with Deribit. Returns True on success."""
        raise NotImplementedError("Live trading not yet implemented. See TradingInterface docstring.")

    def get_account_summary(self, currency: str = "BTC") -> Dict:
        """Fetch account equity, margin, available balance."""
        raise NotImplementedError

    def get_option_chain(self, instrument: str, expiry: str) -> pd.DataFrame:
        """Fetch full option chain for given instrument and expiry."""
        raise NotImplementedError

    def place_order(self, instrument_name: str, amount: float,
                    side: str, order_type: str = "market",
                    price: Optional[float] = None, label: str = "") -> Dict:
        """Place a live order. side='buy'|'sell'. Returns order dict."""
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an existing order."""
        raise NotImplementedError

    def get_positions(self) -> List[Dict]:
        """Fetch all open positions."""
        raise NotImplementedError

    def close_position(self, instrument_name: str) -> Dict:
        """Market-close a position."""
        raise NotImplementedError

    def get_greeks(self, instrument_name: str) -> Dict:
        """Fetch Delta, Gamma, Theta, Vega for an instrument."""
        raise NotImplementedError

    def status(self) -> Dict:
        return {
            "connected": self.connected,
            "testnet": self.testnet,
            "base_url": self.base_url,
            "note": "STUB – not connected to live exchange",
        }


# ══════════════════════════════════════════════════════════════
#  GOOGLE SHEETS 接続
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def get_gsheet_client():
    """Google Sheets client and worksheets. Returns (ws_data, ws_signals) or (None, None)."""
    try:
        if "gcp_service_account" in st.secrets:
            creds = Credentials.from_service_account_info(
                st.secrets["gcp_service_account"], scopes=SCOPES)
        elif os.path.exists("credentials.json"):
            creds = Credentials.from_service_account_file("credentials.json", scopes=SCOPES)
        else:
            return None, None

        gc = gspread.authorize(creds)
        sh = gc.open_by_key(SHEET_ID)

        # iv_data worksheet
        try:
            ws_data = sh.worksheet(SHEET_NAME)
        except gspread.WorksheetNotFound:
            ws_data = sh.add_worksheet(title=SHEET_NAME, rows=100000, cols=10)
            ws_data.append_row(["Timestamp","BTC_IV","ETH_IV","BTC_ETH_Ratio","BTC_Spot","ETH_Spot"])

        # signal_history worksheet
        try:
            ws_signals = sh.worksheet(SIGNAL_SHEET_NAME)
        except gspread.WorksheetNotFound:
            ws_signals = sh.add_worksheet(title=SIGNAL_SHEET_NAME, rows=100000, cols=10)
            ws_signals.append_row([
                "Timestamp","State","Confidence","Signal","Strategy",
                "BTC_IV","ETH_IV","OpportunityScore","Method"
            ])

        return ws_data, ws_signals
    except Exception as e:
        return None, None


def sheet_append(ws, row: list):
    try:
        if ws:
            ws.append_row(row, value_input_option="USER_ENTERED")
    except Exception:
        pass


def load_from_sheet(ws) -> pd.DataFrame:
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


def load_signals_from_sheet(ws_signals) -> pd.DataFrame:
    try:
        if ws_signals is None:
            return pd.DataFrame()
        records = ws_signals.get_all_values()
        if len(records) < 2:
            return pd.DataFrame()
        df = pd.DataFrame(records[1:], columns=records[0])
        for col in ["Confidence","BTC_IV","ETH_IV","OpportunityScore"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════
#  DATA QUALITY MONITOR
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def get_data_quality_store():
    return {
        "api_calls": 0,
        "api_failures": 0,
        "last_success": None,
        "missing_count": 0,
        "total_rows": 0,
    }


def compute_data_quality_score(dq: Dict) -> Tuple[int, str]:
    """Returns (score 0-100, status_label)."""
    score = 100
    if dq["api_calls"] > 0:
        fail_rate = dq["api_failures"] / dq["api_calls"]
        score -= int(fail_rate * 40)
    if dq["total_rows"] > 0:
        miss_rate = dq["missing_count"] / max(dq["total_rows"], 1)
        score -= int(miss_rate * 30)
    if dq["last_success"]:
        delta = (datetime.now() - dq["last_success"]).total_seconds()
        if delta > 300:
            score -= 30
        elif delta > 120:
            score -= 15
    score = max(0, min(100, score))
    if score >= 90:   status = "🟢 Excellent"
    elif score >= 70: status = "🟡 Good"
    elif score >= 50: status = "🟠 Fair"
    else:             status = "🔴 Poor"
    return score, status


# ══════════════════════════════════════════════════════════════
#  STATE STORE
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def get_state_store():
    return {
        "positions":     [],
        "closed_trades": [],
        "capital":       INITIAL_CAPITAL,
        "state_history": deque(maxlen=500),
        "last_signal":   None,
        "gmm_model":     None,
        "gmm_scaler":    None,
        "gmm_label_map": {},
        "telegram_sent": deque(maxlen=50),  # deduplicate alerts
    }


# ══════════════════════════════════════════════════════════════
#  TELEGRAM ALERT SYSTEM
# ══════════════════════════════════════════════════════════════
def get_telegram_config() -> Tuple[Optional[str], Optional[str]]:
    token   = st.secrets.get("TELEGRAM_TOKEN", os.environ.get("TELEGRAM_TOKEN"))
    chat_id = st.secrets.get("TELEGRAM_CHAT_ID", os.environ.get("TELEGRAM_CHAT_ID"))
    return token, chat_id


def send_telegram(message: str, store: Dict) -> bool:
    """Send Telegram message. Returns True on success. Deduplicates within 10 min."""
    token, chat_id = get_telegram_config()
    if not token or not chat_id:
        return False
    # Deduplicate: hash first 80 chars
    key = message[:80]
    if key in store["telegram_sent"]:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=5)
        if resp.status_code == 200:
            store["telegram_sent"].append(key)
            return True
    except Exception:
        pass
    return False


def maybe_send_alerts(state_key: str, btc_z: float, eth_z: float,
                      opp_score: int, confidence: float,
                      btc_iv: float, eth_iv: float, store: Dict):
    """Fire Telegram alerts for panic / high Z-score / high opportunity score."""
    alerts = []

    if state_key == "panic":
        alerts.append(
            f"🚨 <b>PANIC DETECTED</b>\n"
            f"BTC IV: {btc_iv:.1f}  ETH IV: {eth_iv:.1f}\n"
            f"BTC Z: {btc_z:.2f}  ETH Z: {eth_z:.2f}\n"
            f"Opportunity Score: {opp_score}/100\n"
            f"Confidence: {confidence*100:.0f}%"
        )

    if max(abs(btc_z), abs(eth_z)) > 3.0 and state_key != "panic":
        alerts.append(
            f"⚠️ <b>Z-SCORE ALERT</b>\n"
            f"BTC Z: {btc_z:.2f}  ETH Z: {eth_z:.2f}\n"
            f"BTC IV: {btc_iv:.1f}  ETH IV: {eth_iv:.1f}"
        )

    if opp_score > 80:
        alerts.append(
            f"💡 <b>HIGH OPPORTUNITY SCORE</b>: {opp_score}/100\n"
            f"State: {MARKET_STATES[state_key]['label']}\n"
            f"Confidence: {confidence*100:.0f}%"
        )

    for msg in alerts:
        send_telegram(msg, store)


# ══════════════════════════════════════════════════════════════
#  BACKGROUND BOT
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def launch_background_bot():
    ws_data, _ = get_gsheet_client()
    dq         = get_data_quality_store()

    if not os.path.exists(CSV_FILE):
        pd.DataFrame(columns=["Timestamp","BTC_IV","ETH_IV","BTC_ETH_Ratio","BTC_Spot","ETH_Spot"]
                     ).to_csv(CSV_FILE, index=False)

    def get_dvol(symbol: str) -> Optional[float]:
        url = f"https://www.deribit.com/api/v2/public/get_index_price?index_name={symbol.lower()}dvol_usdc"
        dq["api_calls"] += 1
        try:
            r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=5)
            if r.status_code == 200:
                return float(r.json()["result"]["index_price"])
        except Exception:
            pass
        dq["api_failures"] += 1
        return None

    def get_spot(symbol: str) -> Optional[float]:
        url = f"https://www.deribit.com/api/v2/public/get_index_price?index_name={symbol.lower()}_usd"
        dq["api_calls"] += 1
        try:
            r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=5)
            if r.status_code == 200:
                return float(r.json()["result"]["index_price"])
        except Exception:
            pass
        dq["api_failures"] += 1
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

                    row_df = pd.DataFrame([[ts, round(btc_iv,4), round(eth_iv,4), ratio, bsp, esp]])
                    with _lock:
                        row_df.to_csv(CSV_FILE, mode="a", header=False, index=False)
                        dq["last_success"] = datetime.now()

                    sheet_append(ws_data, [ts, round(btc_iv,4), round(eth_iv,4), ratio, bsp, esp])

            except Exception:
                dq["api_failures"] += 1
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

        # RV20 and RV60 (annualized, in vol-point terms matching DVOL scale)
        df[f"{prefix}_RV20"]  = df[col].rolling(20, min_periods=5).std() * np.sqrt(365 * 24 * 60)
        df[f"{prefix}_RV60"]  = df[col].rolling(60, min_periods=10).std() * np.sqrt(365 * 24 * 60)

        # IV-RV Spreads
        df[f"{prefix}_Spread20"] = df[col] - df[f"{prefix}_RV20"]
        df[f"{prefix}_Spread60"] = df[col] - df[f"{prefix}_RV60"]

        # IV Momentum (5-period rate of change)
        df[f"{prefix}_IVMom"]    = df[col].pct_change(5) * 100

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
#  OPPORTUNITY SCORE ENGINE
# ══════════════════════════════════════════════════════════════
def compute_opportunity_score(df: pd.DataFrame, state_key: str, confidence: float) -> Tuple[int, List[str]]:
    """
    Score 0-100 based on:
      - BTC Z-score magnitude      (0-25 pts)
      - ETH Z-score magnitude      (0-20 pts)
      - Ratio Z-score              (0-15 pts)
      - IV Momentum                (0-15 pts)
      - State Confidence           (0-15 pts)
      - Spike Detection            (0-10 pts)
    """
    if df.empty or len(df) < 5:
        return 0, ["データ不足"]

    latest   = df.iloc[-1]
    reasons  = []
    score    = 0

    # BTC Z-score (0-25)
    btc_z = abs(latest.get("BTC_Z") or 0)
    btc_pts = min(25, int(btc_z / 3.0 * 25))
    score += btc_pts
    if btc_pts > 10:
        reasons.append(f"BTC Z={btc_z:.2f} (+{btc_pts}pt)")

    # ETH Z-score (0-20)
    eth_z = abs(latest.get("ETH_Z") or 0)
    eth_pts = min(20, int(eth_z / 3.0 * 20))
    score += eth_pts
    if eth_pts > 8:
        reasons.append(f"ETH Z={eth_z:.2f} (+{eth_pts}pt)")

    # Ratio Z-score (0-15)
    ratio_z = abs(latest.get("Ratio_Z") or 0)
    ratio_pts = min(15, int(ratio_z / 3.0 * 15))
    score += ratio_pts
    if ratio_pts > 6:
        reasons.append(f"Ratio Z={ratio_z:.2f} (+{ratio_pts}pt)")

    # IV Momentum (0-15)
    btc_mom = abs(latest.get("BTC_IVMom") or 0)
    mom_pts = min(15, int(btc_mom / 5.0 * 15))
    score += mom_pts
    if mom_pts > 6:
        reasons.append(f"IV Mom={btc_mom:.2f}% (+{mom_pts}pt)")

    # State Confidence (0-15)
    conf_pts = int(confidence * 15)
    score += conf_pts
    if state_key != "unknown":
        reasons.append(f"{MARKET_STATES[state_key]['label']} conf={confidence*100:.0f}% (+{conf_pts}pt)")

    # Spike (0-10)
    if latest.get("BTC_Spike") or latest.get("ETH_Spike"):
        score += 10
        reasons.append("Spike検出 (+10pt)")

    score = min(100, max(0, score))
    if not reasons:
        reasons.append("通常の市場状態")
    return score, reasons


# ══════════════════════════════════════════════════════════════
#  VOLATILITY REGIME ANALYZER
# ══════════════════════════════════════════════════════════════
def classify_vol_regime(df: pd.DataFrame) -> Tuple[str, str, float]:
    """Returns (regime_key, label, current_iv_pct) based on IV level and Z-score."""
    if df.empty or len(df) < 10:
        return "normal", VOL_REGIMES["normal"]["label"], 50.0

    btc_z = float(df.iloc[-1].get("BTC_Z") or 0)

    if btc_z >= VOL_REGIMES["extreme"]["min_z"]:
        regime = "extreme"
    elif btc_z >= VOL_REGIMES["high"]["min_z"]:
        regime = "high"
    elif btc_z <= VOL_REGIMES["low"]["max_z"]:
        regime = "low"
    else:
        regime = "normal"

    # Percentile of current IV in rolling window
    btc_iv_series = df["BTC_IV"].dropna()
    if len(btc_iv_series) >= 20:
        current = float(btc_iv_series.iloc[-1])
        pct = float(stats.percentileofscore(btc_iv_series.tail(60), current))
    else:
        pct = 50.0

    return regime, VOL_REGIMES[regime]["label"], pct


# ══════════════════════════════════════════════════════════════
#  GMM STATE CLASSIFIER
# ══════════════════════════════════════════════════════════════
FEATURE_COLS = ["BTC_Z","ETH_Z","Ratio_Z","BTC_RV10","ETH_RV10","Ratio_Mom5"]


def _map_gmm_cluster_to_state(gmm, scaler) -> Dict:
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


def fit_or_update_gmm(store: Dict, df: pd.DataFrame):
    feat = df[FEATURE_COLS].dropna()
    if len(feat) < GMM_MIN_ROWS:
        return
    scaler = StandardScaler()
    X = scaler.fit_transform(feat)
    gmm = GaussianMixture(n_components=GMM_COMPONENTS, covariance_type="full",
                          random_state=42, max_iter=200)
    gmm.fit(X)
    store["gmm_model"]     = gmm
    store["gmm_scaler"]    = scaler
    store["gmm_label_map"] = _map_gmm_cluster_to_state(gmm, scaler)


def classify_state(store: Dict, df: pd.DataFrame) -> Tuple[str, float, str]:
    if len(df) < 5:
        return "unknown", 0.0, "insufficient_data"

    latest = df.iloc[-1]
    gmm    = store.get("gmm_model")
    scaler = store.get("gmm_scaler")
    lmap   = store.get("gmm_label_map", {})

    if gmm and scaler:
        row_feat = pd.to_numeric(latest[FEATURE_COLS], errors="coerce").astype(float).values
        if not np.isnan(row_feat).any():
            x = scaler.transform([row_feat])
            probs   = gmm.predict_proba(x)[0]
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


def get_state_history_df(store: Dict) -> pd.DataFrame:
    if not store["state_history"]:
        return pd.DataFrame(columns=["Timestamp","State","Confidence","Method"])
    return pd.DataFrame(list(store["state_history"]),
                        columns=["Timestamp","State","Confidence","Method"])


# ══════════════════════════════════════════════════════════════
#  STATE TRANSITION MATRIX
# ══════════════════════════════════════════════════════════════
def compute_transition_matrix(store: Dict) -> Optional[pd.DataFrame]:
    hist_df = get_state_history_df(store)
    if len(hist_df) < 10:
        return None

    states = [s for s in MARKET_STATES.keys() if s != "unknown"]
    matrix = pd.DataFrame(0.0, index=states, columns=states)
    counts = pd.DataFrame(0,   index=states, columns=states)

    prev = None
    for s in hist_df["State"]:
        if prev and prev in states and s in states:
            counts.loc[prev, s] += 1
        prev = s

    for row_state in states:
        total = counts.loc[row_state].sum()
        if total > 0:
            matrix.loc[row_state] = counts.loc[row_state] / total * 100

    return matrix


# ══════════════════════════════════════════════════════════════
#  STRATEGY ENGINE
# ══════════════════════════════════════════════════════════════
def generate_signal(state_key: str, confidence: float,
                    df: pd.DataFrame, store: Dict) -> Optional[Dict]:
    if state_key == "unknown" or confidence < 0.65:
        return None
    strat_key = MARKET_STATES[state_key]["strategy"]
    if strat_key is None:
        return None
    if strat_key in [p["strategy"] for p in store["positions"]]:
        return None
    latest = df.iloc[-1]
    strat  = STRATEGIES[strat_key]
    return {
        "id":               f"{state_key[:3].upper()}-{datetime.now().strftime('%H%M%S')}",
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "state":            state_key,
        "strategy":         strat_key,
        "strategy_name":    strat["name"],
        "action":           strat["action"],
        "legs":             strat["legs"],
        "rationale":        strat["rationale"],
        "risk":             strat["risk"],
        "confidence":       confidence,
        "btc_iv_at_signal": float(latest.get("BTC_IV") or 0),
        "eth_iv_at_signal": float(latest.get("ETH_IV") or 0),
        "target_pnl_pct":   strat["target_pnl_pct"],
        "stop_pnl_pct":     strat["stop_pnl_pct"],
    }


def save_signal_to_sheet(signal: Dict, opp_score: int, method: str, ws_signals):
    """Persist signal to Google Sheets signal_history tab."""
    if ws_signals is None or signal is None:
        return
    sheet_append(ws_signals, [
        signal["timestamp"],
        signal["state"],
        round(signal["confidence"], 4),
        signal["id"],
        signal["strategy"],
        round(signal["btc_iv_at_signal"], 4),
        round(signal["eth_iv_at_signal"], 4),
        opp_score,
        method,
    ])


# ══════════════════════════════════════════════════════════════
#  PAPER TRADING
# ══════════════════════════════════════════════════════════════
def open_paper_trade(signal: Dict, store: Dict) -> Optional[Dict]:
    if len(store["positions"]) >= MAX_POSITIONS:
        return None
    trade = {
        **signal,
        "entry_time":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "size_usd":         round(store["capital"] * POSITION_SIZE_PCT, 2),
        "current_pnl_usd":  0.0,
        "current_pnl_pct":  0.0,
        "status":           "OPEN",
    }
    store["positions"].append(trade)
    return trade


def update_paper_trades(store: Dict, df: pd.DataFrame):
    if df.empty or len(df) < 2:
        return
    latest  = df.iloc[-1]
    prev    = df.iloc[-2]
    btc_chg = (latest["BTC_IV"] - prev["BTC_IV"]) / max(prev["BTC_IV"], 0.01)
    eth_chg = (latest["ETH_IV"] - prev["ETH_IV"]) / max(prev["ETH_IV"], 0.01)
    avg_chg = (btc_chg + eth_chg) / 2
    closed  = []

    for pos in store["positions"]:
        s = pos["strategy"]
        if   s == "sell_vol":   pnl_delta = -avg_chg * 2
        elif s == "buy_vol":    pnl_delta =  avg_chg * 2
        elif s == "btc_skew":
            rc = (latest["BTC_ETH_Ratio"] - prev["BTC_ETH_Ratio"]) / max(prev["BTC_ETH_Ratio"], 0.01)
            pnl_delta = -rc * 3
        elif s == "long_gamma": pnl_delta = abs(avg_chg) * 3 - 0.001
        else:                   pnl_delta = 0.0

        pos["current_pnl_pct"] = pos.get("current_pnl_pct", 0.0) + pnl_delta
        pos["current_pnl_usd"] = round(pos["size_usd"] * pos["current_pnl_pct"], 2)

        if   pos["current_pnl_pct"] >= pos["target_pnl_pct"]: pos["status"] = "CLOSED_TP"; closed.append(pos)
        elif pos["current_pnl_pct"] <= pos["stop_pnl_pct"]:   pos["status"] = "CLOSED_SL"; closed.append(pos)

    for pos in closed:
        pos["exit_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        store["capital"] += pos["current_pnl_usd"]
        store["positions"].remove(pos)
        store["closed_trades"].append(pos)


def get_portfolio_stats(store: Dict) -> Dict:
    closed = store["closed_trades"]
    if not closed:
        return {"total_trades":0,"win_rate":0.0,"total_pnl_usd":0.0,
                "avg_pnl_pct":0.0,"best_trade":0.0,"worst_trade":0.0,"sharpe":0.0}
    pnls    = [t["current_pnl_usd"] for t in closed]
    pnl_pct = [t["current_pnl_pct"] for t in closed]
    wins    = [p for p in pnls if p > 0]
    sharpe  = 0.0
    if len(pnl_pct) > 1:
        mu  = np.mean(pnl_pct)
        sig = np.std(pnl_pct)
        sharpe = round(mu / sig * np.sqrt(252) if sig > 0 else 0.0, 2)
    return {
        "total_trades":  len(closed),
        "win_rate":      round(len(wins)/len(closed)*100, 1),
        "total_pnl_usd": round(sum(pnls), 2),
        "avg_pnl_pct":   round(np.mean(pnl_pct)*100, 2),
        "best_trade":    round(max(pnl_pct)*100, 2),
        "worst_trade":   round(min(pnl_pct)*100, 2),
        "sharpe":        sharpe,
    }


# ══════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════
def run_backtest(df: pd.DataFrame, signal_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Simulate strategy outcomes by evaluating IV changes at
    +1h, +6h, +24h after each recorded signal.
    Returns dict keyed by strategy name.
    """
    results = {}

    if signal_df.empty or df.empty:
        return results

    df_ts = df.copy()
    df_ts["Timestamp"] = pd.to_datetime(df_ts["Timestamp"], errors="coerce")
    df_ts = df_ts.dropna(subset=["Timestamp"]).set_index("Timestamp").sort_index()

    rows = []
    for _, sig in signal_df.iterrows():
        try:
            entry_time = pd.to_datetime(sig["Timestamp"])
        except Exception:
            continue

        entry_slice = df_ts[df_ts.index <= entry_time]
        if entry_slice.empty:
            continue
        entry_iv = float(entry_slice.iloc[-1]["BTC_IV"])

        outcomes = {}
        for label, hours in [("1h", 1), ("6h", 6), ("24h", 24)]:
            target_time  = entry_time + timedelta(hours=hours)
            future_slice = df_ts[df_ts.index >= target_time]
            if future_slice.empty:
                outcomes[label] = np.nan
                continue
            future_iv = float(future_slice.iloc[0]["BTC_IV"])
            iv_chg = (future_iv - entry_iv) / max(entry_iv, 0.01)

            strat = str(sig.get("Strategy",""))
            if   strat == "sell_vol":   pnl = -iv_chg * 2
            elif strat == "buy_vol":    pnl =  iv_chg * 2
            elif strat == "btc_skew":   pnl = -iv_chg
            elif strat == "long_gamma": pnl =  abs(iv_chg) * 3 - 0.001
            else:                       pnl =  0.0
            outcomes[label] = round(pnl * 100, 2)

        rows.append({
            "Timestamp": sig["Timestamp"],
            "Strategy":  sig.get("Strategy",""),
            "State":     sig.get("State",""),
            "Conf":      sig.get("Confidence", 0),
            "EntryIV":   entry_iv,
            "PnL_1h":    outcomes.get("1h", np.nan),
            "PnL_6h":    outcomes.get("6h", np.nan),
            "PnL_24h":   outcomes.get("24h", np.nan),
        })

    if not rows:
        return results

    all_df = pd.DataFrame(rows)

    for strat_name in all_df["Strategy"].unique():
        sub = all_df[all_df["Strategy"] == strat_name].copy()
        if sub.empty:
            continue
        summary = {}
        for horizon in ["PnL_1h","PnL_6h","PnL_24h"]:
            vals = sub[horizon].dropna()
            if vals.empty:
                continue
            wins    = (vals > 0).sum()
            win_rate = wins / len(vals) * 100
            avg_ret  = vals.mean()
            std_ret  = vals.std()
            sharpe   = avg_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
            cum      = (1 + vals / 100).cumprod()
            roll_max = cum.cummax()
            dd       = (cum - roll_max) / roll_max * 100
            max_dd   = dd.min()
            summary[horizon] = {
                "WinRate%": round(win_rate, 1),
                "AvgRet%":  round(avg_ret, 2),
                "Sharpe":   round(sharpe, 2),
                "MaxDD%":   round(max_dd, 2),
                "N":        len(vals),
            }
        if summary:
            results[strat_name] = pd.DataFrame(summary).T

    return results


# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Crypto Vol Research Platform v2",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0b0d12; }
  [data-testid="stHeader"]           { background: transparent; }
  [data-testid="stSidebar"]          { background: #0f1117; }
  .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
  h1,h2,h3 { color: #dde0ea; }
  .state-pill {
    display:inline-block; padding:.3em .9em; border-radius:99px;
    font-size:1rem; font-weight:700; letter-spacing:.04em;
  }
  .card {
    background:#141720; border:1px solid #1e2230;
    border-radius:10px; padding:1rem 1.2rem; margin-bottom:.6rem;
  }
  .signal-card {
    background:#0d1620; border:1px solid #1a3050;
    border-radius:10px; padding:.9rem 1.1rem; margin-bottom:.6rem;
  }
  .opp-card {
    background: linear-gradient(135deg,#141720,#0d1225);
    border:1px solid #2a3050; border-radius:12px;
    padding:1.2rem 1.5rem; margin-bottom:.8rem;
  }
  .label { font-size:.7rem; color:#666; text-transform:uppercase; letter-spacing:.08em; margin-bottom:2px; }
  .roadmap { background:#111418; border-left:3px solid #222; border-radius:5px; padding:.35rem .8rem; margin:.2rem 0; font-size:.82rem; color:#555; }
  .roadmap.done   { border-left-color:#00c896; color:#00c896; }
  .roadmap.active { border-left-color:#4b9eff; color:#c8d8ff; }
  .tag { display:inline-block; padding:.15em .6em; border-radius:4px; font-size:.75rem; font-weight:600; margin-right:.3rem; }
  .green { background:#00c89622; color:#00c896; }
  .red   { background:#ff4b4b22; color:#ff4b4b; }
  .blue  { background:#4b9eff22; color:#4b9eff; }
  .gold  { background:#ffd70022; color:#ffd700; }
  .score-bar-bg { background:#1e2230; border-radius:6px; height:12px; overflow:hidden; margin-top:4px; }
  .score-bar-fill { height:100%; border-radius:6px; transition: width .4s ease; }
  .regime-badge {
    display:inline-block; padding:.4em 1.1em; border-radius:8px;
    font-size:.9rem; font-weight:700; letter-spacing:.05em; margin-top:.3rem;
  }
  .spread-alert { background:#ff4b4b11; border:1px solid #ff4b4b55; border-radius:8px; padding:.6rem 1rem; margin:.4rem 0; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  INIT
# ══════════════════════════════════════════════════════════════
launch_background_bot()
store    = get_state_store()
dq_store = get_data_quality_store()

ws_data, ws_signals = get_gsheet_client()
sheet_ok    = ws_data is not None
signals_ok  = ws_signals is not None

trading_iface = TradingInterface()  # Future trading layer (stub)

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## ⚙️ 設定  `v{APP_VERSION}`")
    auto_trade  = st.toggle("🤖 自動シグナル → ペーパートレード", value=True)
    zscore_thr  = st.slider("Z-score 閾値", 1.0, 3.5, ZSCORE_THRESHOLD, 0.1)
    spike_thr   = st.slider("Spike 閾値 (%)", 1, 10, int(SPIKE_THRESHOLD_PCT*100), 1)
    gmm_window  = st.slider("GMM 学習データ数", 30, 300, 100, 10)
    display_n   = st.slider("チャート表示件数", 50, 500, MAX_ROWS_DISPLAY, 50)

    st.markdown("---")
    st.markdown("### 📡 データソース")
    if sheet_ok:
        st.success("✅ Google Sheets 接続中")
    else:
        st.warning("⚠️ CSV モード（ローカルのみ）")

    tg_token, tg_chat = get_telegram_config()
    if tg_token and tg_chat:
        st.success("✅ Telegram 接続中")
    else:
        st.info("💬 Telegram 未設定")

    # Data Quality in sidebar
    dq_score, dq_status = compute_data_quality_score(dq_store)
    st.markdown("---")
    st.markdown(f"### 🔬 System Health  {dq_status}")
    st.progress(dq_score / 100, text=f"{dq_score}/100")
    st.caption(
        f"API calls: {dq_store['api_calls']} | "
        f"Failures: {dq_store['api_failures']} | "
        f"Last OK: {dq_store['last_success'].strftime('%H:%M:%S') if dq_store['last_success'] else 'N/A'}"
    )

    st.markdown("---")
    st.markdown("### 🗺️ 進捗")
    for label, cls in [
        ("✅ Step 1  Data Collection",   "done"),
        ("✅ Step 2  Anomaly Detection",  "done"),
        ("✅ Step 3  Market State AI",    "done"),
        ("✅ Step 4  Strategy Engine",    "done"),
        ("✅ Step 5  Paper Trading",      "done"),
        ("✅ Step 6  Google Sheets",      "done"),
        ("✅ Step 7  Opportunity Score",  "done"),
        ("✅ Step 8  IV-RV Spread",       "done"),
        ("✅ Step 9  Vol Regime",         "done"),
        ("✅ Step 10 Transition Matrix",  "done"),
        ("✅ Step 11 Signal DB",          "done"),
        ("✅ Step 12 Backtest Engine",    "done"),
        ("✅ Step 13 Research Dashboard", "done"),
        ("✅ Step 14 Telegram Alerts",    "done"),
        ("✅ Step 15 Data Quality",       "done"),
        ("⚪ Step 16 Live Trading",       ""),
    ]:
        st.markdown(f'<div class="roadmap {cls}">{label}</div>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ 全トレードリセット", type="secondary"):
        store["positions"] = []; store["closed_trades"] = []
        store["capital"]   = INITIAL_CAPITAL
        st.success("リセット完了")


# ── DATA LOAD ─────────────────────────────────────────────────
if sheet_ok:
    raw = load_from_sheet(ws_data)
    if raw.empty and os.path.exists(CSV_FILE):
        raw = pd.read_csv(CSV_FILE,
                          names=["Timestamp","BTC_IV","ETH_IV","BTC_ETH_Ratio","BTC_Spot","ETH_Spot"],
                          skiprows=1)
else:
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) < 10:
        st.info("🔄 初回データ取得中... 約1分後に更新してください。")
        time.sleep(10); st.rerun()
    raw = pd.read_csv(CSV_FILE,
                      names=["Timestamp","BTC_IV","ETH_IV","BTC_ETH_Ratio","BTC_Spot","ETH_Spot"],
                      skiprows=1)

if raw.empty:
    st.info("🔄 データがまだありません。しばらくお待ちください。")
    time.sleep(10); st.rerun()

df  = compute_features(raw.copy())
df  = df.replace([np.inf, -np.inf], np.nan)

# Update data quality
dq_store["total_rows"]   = len(df)
dq_store["missing_count"] = int(df[["BTC_IV","ETH_IV"]].isna().sum().sum())

fit_or_update_gmm(store, df.tail(gmm_window))
state_key, confidence, method = classify_state(store, df)

store["state_history"].append((
    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    state_key, round(confidence, 3), method
))

# Compute derived metrics
opp_score, opp_reasons   = compute_opportunity_score(df, state_key, confidence)
vol_regime, regime_label, iv_pct = classify_vol_regime(df)

latest        = df.iloc[-1]
btc_z_latest  = float(latest.get("BTC_Z") or 0)
eth_z_latest  = float(latest.get("ETH_Z") or 0)

signal = generate_signal(state_key, confidence, df, store)
if auto_trade and signal:
    open_paper_trade(signal, store)
    save_signal_to_sheet(signal, opp_score, method, ws_signals)
    store["last_signal"] = signal

update_paper_trades(store, df)
maybe_send_alerts(state_key, btc_z_latest, eth_z_latest, opp_score, confidence,
                  float(latest.get("BTC_IV") or 0), float(latest.get("ETH_IV") or 0), store)

state_info    = MARKET_STATES[state_key]
anomaly_count = int(df["Anomaly"].fillna(False).sum())
df_disp       = df.tail(display_n)
chart_idx     = df_disp.set_index("Timestamp")
stats         = get_portfolio_stats(store)
transition_mx = compute_transition_matrix(store)

# Signal history from Google Sheets
signal_history_df = load_signals_from_sheet(ws_signals)


# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
col_h1, col_h2, col_h3 = st.columns([2, 1, 1])

with col_h1:
    st.markdown(f"# 🛰️ Crypto Volatility Research Platform `v{APP_VERSION}`")
    src_label = "Google Sheets" if sheet_ok else "CSV Local"
    st.caption(
        f"Deribit DVOL · 1-min · {len(df)} rows · "
        f"Source: {src_label} · Last: {latest['Timestamp']}"
    )

with col_h2:
    color    = state_info["color"]
    conf_bar = int(confidence * 10)
    st.markdown(
        f'<div class="card" style="border-color:{color}44;">'
        f'<div class="label">CURRENT STATE ({method})</div>'
        f'<span class="state-pill" style="background:{color}22;color:{color};border:1px solid {color}66;">'
        f'{state_info["label"]}</span>'
        f'<div style="margin-top:.4rem;font-size:.8rem;color:#888;">'
        f'Confidence: <b style="color:{color}">{confidence*100:.0f}%</b> '
        f'{"█"*conf_bar}{"░"*(10-conf_bar)}</div>'
        f'<div style="font-size:.75rem;color:#666;margin-top:.3rem;">{state_info["description"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with col_h3:
    # Opportunity Score
    score_color = "#ff4b4b" if opp_score >= 80 else ("#ffd700" if opp_score >= 50 else "#00c896")
    fill_pct    = opp_score
    st.markdown(
        f'<div class="opp-card">'
        f'<div class="label">OPPORTUNITY SCORE</div>'
        f'<div style="font-size:2rem;font-weight:900;color:{score_color};line-height:1.1;">'
        f'{opp_score}<span style="font-size:1rem;color:#666;">/100</span></div>'
        f'<div class="score-bar-bg"><div class="score-bar-fill" '
        f'style="width:{fill_pct}%;background:{score_color};"></div></div>'
        f'<div style="font-size:.7rem;color:#666;margin-top:.4rem;">'
        f'{"  ·  ".join(opp_reasons[:2])}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── KPI ROW ───────────────────────────────────────────────────
def delta_val(series):
    if len(series) < 2: return None
    return round(float(series.iloc[-1] - series.iloc[-2]), 4)

k1,k2,k3,k4,k5,k6,k7,k8 = st.columns(8)
k1.metric("BTC ATM IV",    f"{latest['BTC_IV']:.2f}",          delta=delta_val(df["BTC_IV"]))
k2.metric("ETH ATM IV",    f"{latest['ETH_IV']:.2f}",          delta=delta_val(df["ETH_IV"]))
k3.metric("BTC/ETH Ratio", f"{latest['BTC_ETH_Ratio']:.4f}",   delta=delta_val(df["BTC_ETH_Ratio"]))
k4.metric("BTC Z-score",   f"{btc_z_latest:.2f}")
k5.metric("Vol Regime",    regime_label.split(" ")[0] + " " + regime_label.split(" ")[1] if len(regime_label.split()) > 1 else regime_label)
k6.metric("Anomalies",     f"{anomaly_count}")
k7.metric("Opp Score",     f"{opp_score}/100")
pnl_str = f"+${stats['total_pnl_usd']:.0f}" if stats['total_pnl_usd'] >= 0 else f"-${abs(stats['total_pnl_usd']):.0f}"
k8.metric("Paper P&L",     pnl_str,
          delta=f"{stats['win_rate']}% WR" if stats["total_trades"] else None)

st.divider()


# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📈 IV Monitor",
    "🚨 Anomaly",
    "🧠 Market State",
    "📊 IV-RV Spread",
    "⚡ Strategy",
    "💼 Paper Trading",
    "🔬 Research",
    "⏮️ Backtest",
    "🗄️ Data",
])


# ────────────────────────────────────────────────────────────
# TAB 1: IV Monitor
# ────────────────────────────────────────────────────────────
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
        st.markdown("#### Realized Vol Proxy (10-period)")
        st.line_chart(rv_data, height=200)


# ────────────────────────────────────────────────────────────
# TAB 2: Anomaly
# ────────────────────────────────────────────────────────────
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


# ────────────────────────────────────────────────────────────
# TAB 3: Market State
# ────────────────────────────────────────────────────────────
with tab3:
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        st.markdown("#### 状態定義 & Vol Regime")
        for key, info in MARKET_STATES.items():
            if key == "unknown": continue
            is_current = (key == state_key)
            border     = f"border:1px solid {info['color']}88;" if is_current else ""
            curr_badge = '<br><b style="color:#4b9eff;font-size:.7rem">← 現在</b>' if is_current else ""
            st.markdown(
                f'<div class="card" style="{border}margin-bottom:.4rem;">'
                f'<span class="state-pill" style="background:{info["color"]}22;color:{info["color"]};'
                f'border:1px solid {info["color"]}44;font-size:.85rem;">{info["label"]}</span>'
                f'{curr_badge}'
                f'<div style="font-size:.75rem;color:#777;margin-top:.3rem;">{info["description"]}</div>'
                f'</div>', unsafe_allow_html=True)

        # Vol Regime display
        rc = VOL_REGIMES[vol_regime]["color"]
        st.markdown(
            f'<div class="card" style="border-color:{rc}55;">'
            f'<div class="label">VOLATILITY REGIME</div>'
            f'<span class="regime-badge" style="background:{rc}22;color:{rc};border:1px solid {rc}44;">'
            f'{VOL_REGIMES[vol_regime]["label"]}</span>'
            f'<div style="font-size:.75rem;color:#777;margin-top:.3rem;">'
            f'IV Percentile: <b style="color:{rc}">{iv_pct:.0f}th</b> · '
            f'{VOL_REGIMES[vol_regime]["description"]}</div>'
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
                    f'<span class="state-pill" style="background:{color}22;color:{color};'
                    f'border:1px solid {color}44;font-size:.75rem;padding:.15em .6em">{info["label"]}</span>'
                    f'<span style="color:#777;font-size:.75rem">conf: {row["Confidence"]*100:.0f}% · {row["Method"]}</span>'
                    f'</div>', unsafe_allow_html=True)
        else:
            st.info("状態履歴を蓄積中...")

    # Transition Matrix
    st.markdown("#### 状態遷移確率 Matrix")
    if transition_mx is not None:
        # Color map display
        formatted = transition_mx.copy()
        for col in formatted.columns:
            formatted[col] = formatted[col].map(lambda x: f"{x:.1f}%")
        st.dataframe(
            transition_mx.style.background_gradient(cmap="YlOrRd", vmin=0, vmax=100)
                               .format("{:.1f}%"),
            use_container_width=True,
        )
        # Highlight key transitions
        st.markdown("##### 主要遷移ハイライト")
        cols_tr = st.columns(4)
        states_display = [s for s in MARKET_STATES.keys() if s != "unknown"]
        for i, s_from in enumerate(states_display):
            if s_from not in transition_mx.index:
                continue
            row = transition_mx.loc[s_from]
            best_to = row.idxmax()
            best_pct = row.max()
            info_f = MARKET_STATES[s_from]
            info_t = MARKET_STATES.get(best_to, MARKET_STATES["unknown"])
            with cols_tr[i % 4]:
                st.markdown(
                    f'<div class="card" style="padding:.6rem;">'
                    f'<div class="label">FROM</div>'
                    f'<span style="color:{info_f["color"]};font-size:.8rem;">{info_f["label"]}</span>'
                    f'<div style="color:#666;font-size:.7rem;margin:.3rem 0;">→ Most likely</div>'
                    f'<span style="color:{info_t["color"]};font-size:.8rem;">{info_t["label"]}</span>'
                    f'<div style="color:#888;font-size:.75rem;margin-top:.2rem;">{best_pct:.1f}%</div>'
                    f'</div>', unsafe_allow_html=True)
    else:
        st.info("遷移行列の計算には十分な状態履歴が必要です（最低10状態変化）")

    if store.get("gmm_model"):
        with st.expander("🔬 GMM クラスター詳細"):
            gmm   = store["gmm_model"]
            sclr  = store["gmm_scaler"]
            lmap  = store["gmm_label_map"]
            means = sclr.inverse_transform(gmm.means_)
            rows  = [{
                "Cluster": i,
                "→ State": MARKET_STATES[lmap.get(i,"unknown")]["label"],
                "Weight %": f"{w*100:.1f}%",
                "BTC_Z":    f"{m[0]:.2f}",
                "ETH_Z":    f"{m[1]:.2f}",
                "Ratio_Z":  f"{m[2]:.2f}",
            } for i,(m,w) in enumerate(zip(means, gmm.weights_))]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ────────────────────────────────────────────────────────────
# TAB 4: IV-RV Spread Monitor
# ────────────────────────────────────────────────────────────
with tab4:
    st.markdown("#### 📊 IV-RV Spread Monitor")
    st.caption("Implied Volatility vs Realized Volatility — positive spread = options are rich vs historical vol")

    # Current snapshot
    s_cols = st.columns(6)
    def _safe(val): return round(float(val), 2) if pd.notna(val) else 0.0

    btc_iv_now    = _safe(latest.get("BTC_IV"))
    eth_iv_now    = _safe(latest.get("ETH_IV"))
    btc_rv20      = _safe(latest.get("BTC_RV20"))
    btc_rv60      = _safe(latest.get("BTC_RV60"))
    eth_rv20      = _safe(latest.get("ETH_RV20"))
    eth_rv60      = _safe(latest.get("ETH_RV60"))
    btc_sp20      = _safe(latest.get("BTC_Spread20"))
    btc_sp60      = _safe(latest.get("BTC_Spread60"))
    eth_sp20      = _safe(latest.get("ETH_Spread20"))
    eth_sp60      = _safe(latest.get("ETH_Spread60"))

    s_cols[0].metric("BTC IV",       f"{btc_iv_now:.1f}")
    s_cols[1].metric("BTC RV20",     f"{btc_rv20:.1f}")
    s_cols[2].metric("BTC RV60",     f"{btc_rv60:.1f}")
    s_cols[3].metric("ETH IV",       f"{eth_iv_now:.1f}")
    s_cols[4].metric("ETH RV20",     f"{eth_rv20:.1f}")
    s_cols[5].metric("ETH RV60",     f"{eth_rv60:.1f}")

    st.divider()

    # Spread alerts
    alerts_fired = []
    for sym, sp20, sp60 in [("BTC", btc_sp20, btc_sp60), ("ETH", eth_sp20, eth_sp60)]:
        for period, sp in [("20", sp20), ("60", sp60)]:
            if abs(sp) > IV_RV_SPREAD_ALERT:
                direction = "RICH" if sp > 0 else "CHEAP"
                alerts_fired.append((sym, period, sp, direction))

    if alerts_fired:
        for sym, period, sp, direction in alerts_fired:
            clr = "#ff9900" if direction == "RICH" else "#4b9eff"
            st.markdown(
                f'<div class="spread-alert">'
                f'⚠️ <b style="color:{clr}">{sym} IV-RV{period} Spread = {sp:+.1f}</b> '
                f'— Options appear <b style="color:{clr}">{direction}</b> vs {period}-period realized vol'
                f'</div>', unsafe_allow_html=True)
    else:
        st.success(f"✅ IV-RV スプレッドは正常範囲内 (閾値: ±{IV_RV_SPREAD_ALERT})")

    # Charts
    c_sp1, c_sp2 = st.columns(2)
    with c_sp1:
        st.markdown("#### BTC IV vs RV20 / RV60")
        btc_rv_df = chart_idx[["BTC_IV","BTC_RV20","BTC_RV60"]].dropna()
        if not btc_rv_df.empty:
            st.line_chart(btc_rv_df, height=250)

        st.markdown("#### BTC IV-RV Spread")
        btc_spread_df = chart_idx[["BTC_Spread20","BTC_Spread60"]].dropna()
        if not btc_spread_df.empty:
            st.line_chart(btc_spread_df, height=200)

    with c_sp2:
        st.markdown("#### ETH IV vs RV20 / RV60")
        eth_rv_df = chart_idx[["ETH_IV","ETH_RV20","ETH_RV60"]].dropna()
        if not eth_rv_df.empty:
            st.line_chart(eth_rv_df, height=250)

        st.markdown("#### ETH IV-RV Spread")
        eth_spread_df = chart_idx[["ETH_Spread20","ETH_Spread60"]].dropna()
        if not eth_spread_df.empty:
            st.line_chart(eth_spread_df, height=200)

    # Summary table
    st.markdown("#### IV-RV スプレッド サマリー")
    summary_rows = [
        {"Symbol":"BTC","IV":btc_iv_now,"RV20":btc_rv20,"RV60":btc_rv60,
         "Spread20":f"{btc_sp20:+.2f}","Spread60":f"{btc_sp60:+.2f}",
         "Status20": "RICH" if btc_sp20 > IV_RV_SPREAD_ALERT else ("CHEAP" if btc_sp20 < -IV_RV_SPREAD_ALERT else "NORMAL"),
         "Status60": "RICH" if btc_sp60 > IV_RV_SPREAD_ALERT else ("CHEAP" if btc_sp60 < -IV_RV_SPREAD_ALERT else "NORMAL"),
         },
        {"Symbol":"ETH","IV":eth_iv_now,"RV20":eth_rv20,"RV60":eth_rv60,
         "Spread20":f"{eth_sp20:+.2f}","Spread60":f"{eth_sp60:+.2f}",
         "Status20": "RICH" if eth_sp20 > IV_RV_SPREAD_ALERT else ("CHEAP" if eth_sp20 < -IV_RV_SPREAD_ALERT else "NORMAL"),
         "Status60": "RICH" if eth_sp60 > IV_RV_SPREAD_ALERT else ("CHEAP" if eth_sp60 < -IV_RV_SPREAD_ALERT else "NORMAL"),
         },
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)


# ────────────────────────────────────────────────────────────
# TAB 5: Strategy
# ────────────────────────────────────────────────────────────
with tab5:
    strat_key = state_info.get("strategy")
    if strat_key and strat_key in STRATEGIES:
        strat = STRATEGIES[strat_key]; color = state_info["color"]
        st.markdown(
            f'<div class="signal-card" style="border-color:{color}55;">'
            f'<div style="display:flex;justify-content:space-between;align-items:start;">'
            f'<div><div class="label">STRATEGY</div>'
            f'<div style="font-size:1.1rem;font-weight:700;color:{color}">{strat["name"]}</div></div>'
            f'<span class="state-pill" style="background:{color}22;color:{color};'
            f'border:1px solid {color}44;font-size:.85rem">{strat["action"]}</span>'
            f'</div>'
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
        states_for = [k for k,v in MARKET_STATES.items() if v.get("strategy")==sk]
        sc = MARKET_STATES[states_for[0]]["color"] if states_for else "#888"
        is_active = (sk == strat_key)
        border = f"border:1px solid {sc}88;" if is_active else ""
        st.markdown(
            f'<div class="card" style="{border}margin-bottom:.3rem;">'
            f'<div style="display:flex;justify-content:space-between;">'
            f'<b style="color:{sc}">{sv["name"]}</b>'
            f'<span class="tag" style="background:{sc}22;color:{sc}">{sv["action"]}</span></div>'
            f'<div style="font-size:.78rem;color:#888;margin-top:.3rem">{sv["rationale"]}</div>'
            f'</div>', unsafe_allow_html=True)

    if not signal_history_df.empty:
        st.markdown("#### 📋 シグナル履歴（Google Sheets）")
        st.dataframe(signal_history_df.tail(50)[::-1], use_container_width=True, height=280)


# ────────────────────────────────────────────────────────────
# TAB 6: Paper Trading
# ────────────────────────────────────────────────────────────
with tab6:
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
    pm8.metric("Sharpe",     f"{stats['sharpe']:.2f}")

    st.divider()
    st.markdown(f"#### 🔓 オープンポジション ({len(store['positions'])} / {MAX_POSITIONS})")
    if not store["positions"]:
        st.info("現在、オープンポジションはありません。")
    else:
        for pos in store["positions"]:
            pnl = pos["current_pnl_pct"]
            pc  = "#00c896" if pnl >= 0 else "#ff4b4b"
            si  = MARKET_STATES.get(pos["state"], MARKET_STATES["unknown"])
            c   = si["color"]
            col_a, col_b = st.columns([3, 1])
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
                    pos["status"]   = "CLOSED_MANUAL"
                    pos["exit_time"]= datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    store["capital"]+= pos["current_pnl_usd"]
                    store["closed_trades"].append(pos)
                    store["positions"].remove(pos)
                    st.rerun()

    st.divider()
    st.markdown(f"#### 📋 決済済みトレード ({len(store['closed_trades'])} 件)")
    if store["closed_trades"]:
        closed_rows = [{
            "ID":    t["id"],
            "戦略":  t["strategy_name"],
            "状態":  MARKET_STATES.get(t["state"],{}).get("label","?"),
            "エントリー": t["entry_time"],
            "決済":  t.get("exit_time","—"),
            "P&L%":  f'{t.get("current_pnl_pct",0)*100:+.2f}%',
            "P&L$":  f'${t["current_pnl_usd"]:+.2f}',
            "終了":  t.get("status","?"),
        } for t in reversed(store["closed_trades"])]
        st.dataframe(pd.DataFrame(closed_rows), use_container_width=True, height=280)
        equity = [INITIAL_CAPITAL]
        for t in store["closed_trades"]:
            equity.append(equity[-1] + t.get("current_pnl_usd", 0))
        st.markdown("#### Equity Curve")
        st.line_chart(pd.DataFrame({"Equity": equity}), color="#00c896", height=200)
    else:
        st.info("まだ決済済みトレードはありません。")


# ────────────────────────────────────────────────────────────
# TAB 7: Research Dashboard
# ────────────────────────────────────────────────────────────
with tab7:
    st.markdown("### 🔬 Volatility Research Dashboard")
    st.caption("研究者向け統計分析 · 分布・相関・ローリング統計")

    btc_iv_series = df["BTC_IV"].dropna()
    eth_iv_series = df["ETH_IV"].dropna()
    ratio_series  = df["BTC_ETH_Ratio"].dropna()

    # Descriptive statistics
    st.markdown("#### 記述統計")
    desc_data = {
        "BTC IV":    btc_iv_series.describe(),
        "ETH IV":    eth_iv_series.describe(),
        "Ratio":     ratio_series.describe(),
    }
    st.dataframe(pd.DataFrame(desc_data).round(4), use_container_width=True)

    st.divider()

    # Distribution plots via histogram data
    r1c1, r1c2, r1c3 = st.columns(3)

    with r1c1:
        st.markdown("#### BTC IV Distribution")
        if len(btc_iv_series) >= 5:
            hist_vals, hist_bins = np.histogram(btc_iv_series, bins=30)
            bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
            hist_df_btc = pd.DataFrame({"BTC_IV": hist_vals}, index=bin_centers.round(1))
            st.bar_chart(hist_df_btc, color="#f7931a", height=220)

    with r1c2:
        st.markdown("#### ETH IV Distribution")
        if len(eth_iv_series) >= 5:
            hist_vals2, hist_bins2 = np.histogram(eth_iv_series, bins=30)
            bin_centers2 = (hist_bins2[:-1] + hist_bins2[1:]) / 2
            hist_df_eth = pd.DataFrame({"ETH_IV": hist_vals2}, index=bin_centers2.round(1))
            st.bar_chart(hist_df_eth, color="#627eea", height=220)

    with r1c3:
        st.markdown("#### BTC/ETH Ratio Distribution")
        if len(ratio_series) >= 5:
            hist_vals3, hist_bins3 = np.histogram(ratio_series, bins=30)
            bin_centers3 = (hist_bins3[:-1] + hist_bins3[1:]) / 2
            hist_df_r = pd.DataFrame({"Ratio": hist_vals3}, index=bin_centers3.round(4))
            st.bar_chart(hist_df_r, color="#ffd700", height=220)

    st.divider()

    # Correlation Matrix
    st.markdown("#### Correlation Matrix")
    corr_cols = ["BTC_IV","ETH_IV","BTC_ETH_Ratio","BTC_Z","ETH_Z","Ratio_Z","BTC_RV10","ETH_RV10"]
    corr_cols_exist = [c for c in corr_cols if c in df.columns]
    if len(corr_cols_exist) >= 3:
        corr_df = df[corr_cols_exist].dropna().corr().round(3)
        st.dataframe(
            corr_df.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1),
            use_container_width=True,
        )

    st.divider()

    # Rolling Correlation: BTC IV vs ETH IV
    st.markdown("#### Rolling Correlation (BTC IV ↔ ETH IV)")
    roll_corr_data = {}
    for window in [20, 60]:
        rc_series = df["BTC_IV"].rolling(window).corr(df["ETH_IV"])
        roll_corr_data[f"RollCorr_{window}"] = rc_series

    roll_corr_df = pd.DataFrame(roll_corr_data, index=df["Timestamp"])
    roll_corr_df = roll_corr_df.dropna()
    if not roll_corr_df.empty:
        st.line_chart(roll_corr_df, height=220)

    # Z-score time series overlay
    st.markdown("#### Z-score Overlay (全指標)")
    z_all = [c for c in ["BTC_Z","ETH_Z","Ratio_Z"] if c in chart_idx.columns]
    if z_all:
        zall_df = chart_idx[z_all].dropna()
        if not zall_df.empty:
            st.line_chart(zall_df, height=220)

    # Quantile stats table
    st.markdown("#### IVパーセンタイル分析")
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    q_rows = []
    for q in quantiles:
        q_rows.append({
            "Percentile": f"{int(q*100)}th",
            "BTC IV": round(btc_iv_series.quantile(q), 2) if len(btc_iv_series) > 5 else np.nan,
            "ETH IV": round(eth_iv_series.quantile(q), 2) if len(eth_iv_series) > 5 else np.nan,
            "Ratio":  round(ratio_series.quantile(q), 4) if len(ratio_series) > 5 else np.nan,
        })
    st.dataframe(pd.DataFrame(q_rows), use_container_width=True)


# ────────────────────────────────────────────────────────────
# TAB 8: Backtest Engine
# ────────────────────────────────────────────────────────────
with tab8:
    st.markdown("### ⏮️ Backtest Engine")
    st.caption("過去のシグナルに対して1h/6h/24h後のIV変化からP&Lを推計します")

    if signal_history_df.empty:
        st.info("バックテストにはシグナル履歴が必要です。シグナルが蓄積されると自動的に分析されます。")
        st.markdown(
            "**現在のシグナル履歴:** Google Sheetsの `signal_history` シートに保存されます。\n\n"
            "シグナルが十分に蓄積された後（通常 10件以上）、ここに結果が表示されます。"
        )
    else:
        st.success(f"✅ {len(signal_history_df)} 件のシグナル履歴でバックテスト実行中...")

        bt_results = run_backtest(df, signal_history_df)

        if not bt_results:
            st.warning("バックテスト結果を計算できませんでした。データが不十分な可能性があります。")
        else:
            # Strategy ranking
            st.markdown("#### 戦略別パフォーマンスランキング")
            ranking_rows = []
            for strat_name, result_df in bt_results.items():
                for horizon, row in result_df.iterrows():
                    ranking_rows.append({
                        "Strategy":   strat_name,
                        "Horizon":    horizon,
                        "WinRate%":   row.get("WinRate%", 0),
                        "AvgRet%":    row.get("AvgRet%", 0),
                        "Sharpe":     row.get("Sharpe", 0),
                        "MaxDD%":     row.get("MaxDD%", 0),
                        "N":          int(row.get("N", 0)),
                    })

            if ranking_rows:
                rank_df = pd.DataFrame(ranking_rows).sort_values("Sharpe", ascending=False)
                st.dataframe(
                    rank_df.style.background_gradient(subset=["Sharpe","WinRate%","AvgRet%"],
                                                      cmap="RdYlGn"),
                    use_container_width=True,
                    height=300,
                )

            # Per-strategy detail
            st.markdown("#### 戦略別詳細")
            for strat_name, result_df in bt_results.items():
                info_color = "#4b9eff"
                with st.expander(f"📊 {strat_name}"):
                    st.dataframe(
                        result_df.style.background_gradient(cmap="RdYlGn"),
                        use_container_width=True,
                    )

        # Signal history table
        st.markdown("#### シグナル履歴テーブル")
        st.dataframe(signal_history_df[::-1], use_container_width=True, height=300)


# ────────────────────────────────────────────────────────────
# TAB 9: Data
# ────────────────────────────────────────────────────────────
with tab9:
    st.markdown("#### 🗄️ データストレージ")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("##### IV Data")
        if sheet_ok:
            st.success(f"✅ 接続中 · {len(df)} 行蓄積済み")
            st.markdown(f"[📊 iv_data シートを開く](https://docs.google.com/spreadsheets/d/{SHEET_ID})")
        else:
            st.warning("CSV モードで動作中。")
        st.dataframe(df.tail(50)[::-1], use_container_width=True, height=280)
        st.download_button(
            "⬇️ IV Data CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"iv_data_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
        )

    with col_d2:
        st.markdown("##### Signal History")
        if signals_ok:
            st.success(f"✅ signal_history シート · {len(signal_history_df)} 件")
            st.markdown(f"[📊 signal_history シートを開く](https://docs.google.com/spreadsheets/d/{SHEET_ID})")
        else:
            st.warning("シグナル履歴シート未接続。")
        if not signal_history_df.empty:
            st.dataframe(signal_history_df[::-1], use_container_width=True, height=280)
            st.download_button(
                "⬇️ Signal History CSV",
                data=signal_history_df.to_csv(index=False).encode("utf-8"),
                file_name=f"signals_{datetime.now():%Y%m%d_%H%M}.csv",
                mime="text/csv",
            )
        else:
            st.info("シグナル履歴はまだありません。")

    with st.expander("🗂️ 生データ (最新200件)"):
        st.dataframe(df.tail(200)[::-1], use_container_width=True, height=300)

    # Data Quality Detail
    with st.expander("🔬 データ品質詳細"):
        dq_s, dq_st = compute_data_quality_score(dq_store)
        q1,q2,q3,q4 = st.columns(4)
        q1.metric("品質スコア",  f"{dq_s}/100")
        q2.metric("API呼び出し",  dq_store["api_calls"])
        q3.metric("API失敗",      dq_store["api_failures"])
        q4.metric("欠損行",       dq_store["missing_count"])
        miss_rate = dq_store["missing_count"] / max(dq_store["total_rows"], 1) * 100
        fail_rate = dq_store["api_failures"]  / max(dq_store["api_calls"], 1) * 100
        st.progress(dq_s / 100, text=f"System Health: {dq_st}")
        st.caption(
            f"欠損率: {miss_rate:.1f}% | "
            f"API失敗率: {fail_rate:.1f}% | "
            f"最終成功: {dq_store['last_success'].strftime('%Y-%m-%d %H:%M:%S') if dq_store['last_success'] else 'N/A'}"
        )

    # Future Trading Layer status
    with st.expander("🚀 Future Trading Layer (設計のみ・未接続)"):
        ts = trading_iface.status()
        st.json(ts)
        st.markdown("""
**接続手順 (将来実装):**
1. Deribit アカウントで API Key を作成（read + trade 権限）
2. `DERIBIT_API_KEY` と `DERIBIT_API_SECRET` を Streamlit Secrets に設定
3. `TradingInterface` クラスの各メソッドを実装
4. `place_order()` / `get_positions()` / `close_position()` を Paper Trading と統合

詳細は `app.py` の `TradingInterface` クラスの docstring を参照。
        """)


# ── AUTO REFRESH ──────────────────────────────────────────────
time.sleep(10)
st.rerun()