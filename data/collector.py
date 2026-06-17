"""
data/collector.py  -  データ収集・永続化レイヤー
- CSV にデータを蓄積してセッション間で保持
- Google Sheets は任意（未設定でも動く）
- バックグラウンドスレッドで1分ごとに自動収集
"""
import os, time, requests, pandas as pd
from datetime import datetime
from threading import Thread, Lock

import streamlit as st

CSV_FILE  = "iv_data.csv"
_CSV_COLS = ["Timestamp","BTC_IV","ETH_IV","BTC_ETH_Ratio","BTC_Spot","ETH_Spot"]
_lock     = Lock()

SCOPES            = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
SHEET_NAME        = "iv_data"
SIGNAL_SHEET_NAME = "signal_history"
TRADE_SHEET_NAME  = "trade_history"

# ── Google Sheets ────────────────────────────────────────────
@st.cache_resource
def get_gsheet_client():
    try:
        from google.oauth2.service_account import Credentials
        import gspread

        if "gcp_service_account" in st.secrets:
            creds = Credentials.from_service_account_info(
                st.secrets["gcp_service_account"], scopes=SCOPES)
        elif os.path.exists("credentials.json"):
            creds = Credentials.from_service_account_file("credentials.json", scopes=SCOPES)
        else:
            return None, None, None

        from config.settings import SHEET_ID
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(SHEET_ID)

        def _ws(name, header):
            try:
                return sh.worksheet(name)
            except gspread.WorksheetNotFound:
                ws = sh.add_worksheet(title=name, rows=100000, cols=len(header))
                ws.append_row(header)
                return ws

        ws_iv  = _ws(SHEET_NAME,       _CSV_COLS)
        ws_sig = _ws(SIGNAL_SHEET_NAME, ["Timestamp","State","Confidence","SignalID","Strategy",
                                         "BTC_IV","ETH_IV","OppScore","Method"])
        ws_trd = _ws(TRADE_SHEET_NAME,  ["Timestamp","TradeID","Strategy","State","Action",
                                         "SizeUSD","EntryTime","ExitTime","PnL_USD","PnL_Pct","Status"])
        return ws_iv, ws_sig, ws_trd
    except Exception:
        return None, None, None


def sheet_append(ws, row: list):
    try:
        if ws:
            ws.append_row(row, value_input_option="USER_ENTERED")
    except Exception:
        pass


def load_signals(ws_sig) -> pd.DataFrame:
    try:
        if ws_sig is None:
            return pd.DataFrame()
        records = ws_sig.get_all_values()
        if len(records) < 2:
            return pd.DataFrame()
        df = pd.DataFrame(records[1:], columns=records[0])
        for col in ["Confidence","BTC_IV","ETH_IV","OppScore"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def load_trades(ws_trd) -> pd.DataFrame:
    try:
        if ws_trd is None:
            return pd.DataFrame()
        records = ws_trd.get_all_values()
        if len(records) < 2:
            return pd.DataFrame()
        return pd.DataFrame(records[1:], columns=records[0])
    except Exception:
        return pd.DataFrame()


# ── Data Quality ─────────────────────────────────────────────
@st.cache_resource
def get_dq_store():
    return {
        "api_calls":     0,
        "api_failures":  0,
        "last_success":  None,
        "total_rows":    0,
        "missing_count": 0,
    }


def dq_score(store) -> tuple:
    score = 100
    if store["api_calls"] > 0:
        score -= int(store["api_failures"] / store["api_calls"] * 40)
    if store["total_rows"] > 0:
        score -= int(store["missing_count"] / max(store["total_rows"], 1) * 30)
    if store["last_success"]:
        delta = (datetime.now() - store["last_success"]).total_seconds()
        if delta > 300:  score -= 30
        elif delta > 120: score -= 15
    score = max(0, min(100, score))
    if score >= 90:   status = "🟢 Excellent"
    elif score >= 70: status = "🟡 Good"
    elif score >= 50: status = "🟠 Fair"
    else:             status = "🔴 Poor"
    return score, status


# ── Deribit DVOL API ─────────────────────────────────────────
_BASE = "https://www.deribit.com/api/v2/public"


def _get_dvol(symbol: str, dq: dict):
    """Deribit DVOL index 取得（例: btcdvol_usdc）"""
    url = f"{_BASE}/get_index_price?index_name={symbol.lower()}dvol_usdc"
    dq["api_calls"] += 1
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        if r.status_code == 200:
            return float(r.json()["result"]["index_price"])
    except Exception:
        pass
    dq["api_failures"] += 1
    return None


def _get_spot(symbol: str, dq: dict):
    """Deribit スポット価格取得"""
    url = f"{_BASE}/get_index_price?index_name={symbol.lower()}_usd"
    dq["api_calls"] += 1
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        if r.status_code == 200:
            return float(r.json()["result"]["index_price"])
    except Exception:
        pass
    dq["api_failures"] += 1
    return None


# ── CSV 永続化 ───────────────────────────────────────────────
def _ensure_csv():
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) < 5:
        pd.DataFrame(columns=_CSV_COLS).to_csv(CSV_FILE, index=False)


def load_raw_data(ws_iv=None) -> pd.DataFrame:
    """CSVからデータを読み込む（セッション間で永続）"""
    # Google Sheets 優先
    if ws_iv is not None:
        try:
            records = ws_iv.get_all_values()
            if len(records) >= 2:
                df = pd.DataFrame(records[1:], columns=records[0])
                for col in ["BTC_IV","ETH_IV","BTC_ETH_Ratio","BTC_Spot","ETH_Spot"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                return df.dropna(subset=["BTC_IV","ETH_IV"])
        except Exception:
            pass

    # CSV フォールバック
    _ensure_csv()
    try:
        df = pd.read_csv(CSV_FILE)
        if df.empty:
            return pd.DataFrame()
        for col in ["BTC_IV","ETH_IV","BTC_ETH_Ratio","BTC_Spot","ETH_Spot"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["BTC_IV","ETH_IV"])
    except Exception:
        return pd.DataFrame()


# ── バックグラウンド収集スレッド ──────────────────────────────
@st.cache_resource
def launch_collector():
    """1分ごとにDeribitからDVOLデータを取得してCSVに追記"""
    dq = get_dq_store()
    _ensure_csv()
    _ws_container = {"ws": None}

    def _bot():
        while True:
            try:
                btc_iv = _get_dvol("btc", dq); time.sleep(0.5)
                eth_iv = _get_dvol("eth", dq); time.sleep(0.5)
                btc_sp = _get_spot("btc",  dq); time.sleep(0.5)
                eth_sp = _get_spot("eth",  dq)

                if btc_iv and eth_iv:
                    ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ratio = round(btc_iv / eth_iv, 6)
                    bsp   = round(btc_sp, 2) if btc_sp else ""
                    esp   = round(eth_sp, 2) if eth_sp else ""

                    row = [ts, round(btc_iv,4), round(eth_iv,4), ratio, bsp, esp]

                    with _lock:
                        pd.DataFrame([row], columns=_CSV_COLS).to_csv(
                            CSV_FILE, mode="a", header=False, index=False)
                        dq["last_success"] = datetime.now()

                    # Sheets追記（初回のみ接続）
                    if _ws_container["ws"] is None:
                        try:
                            ws_iv, _, _ = get_gsheet_client()
                            _ws_container["ws"] = ws_iv
                        except Exception:
                            pass
                    sheet_append(_ws_container["ws"], row)

            except Exception:
                dq["api_failures"] += 1

            time.sleep(58)

    t = Thread(target=_bot, daemon=True)
    t.start()
    return t