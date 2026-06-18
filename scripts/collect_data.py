"""
scripts/collect_data.py
GitHub Actionsから実行されるデータ収集スクリプト
"""
import os
import sys
import requests
import time
from datetime import datetime

# Google Sheets接続
from google.oauth2.service_account import Credentials
import gspread
import json

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

SHEET_ID  = "1C4Gd0AqHcMNg-QHhMCGoNzONyse5NMBGCfHjcuVcOKY"
BASE      = "https://www.deribit.com/api/v2/public"

def get_dvol(symbol):
    url = f"{BASE}/get_index_price?index_name={symbol.lower()}dvol_usdc"
    try:
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=5)
        if r.status_code == 200:
            return float(r.json()["result"]["index_price"])
    except Exception:
        pass
    return None

def get_spot(symbol):
    url = f"{BASE}/get_index_price?index_name={symbol.lower()}_usd"
    try:
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=5)
        if r.status_code == 200:
            return float(r.json()["result"]["index_price"])
    except Exception:
        pass
    return None

def get_funding():
    url = f"{BASE}/get_funding_rate_value?instrument_name=BTC-PERPETUAL&start_timestamp=0&end_timestamp=9999999999999"
    try:
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=5)
        if r.status_code == 200:
            return float(r.json()["result"])
    except Exception:
        pass
    return None

def get_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        if r.status_code == 200:
            return int(r.json()["data"][0]["value"])
    except Exception:
        pass
    return None

def main():
    # GCP認証情報をGitHub Secretsから取得
    gcp_json = os.environ.get("GCP_SERVICE_ACCOUNT")
    if not gcp_json:
        print("ERROR: GCP_SERVICE_ACCOUNT not set")
        sys.exit(1)

    creds_dict = json.loads(gcp_json)
    creds      = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    gc         = gspread.authorize(creds)
    sh         = gc.open_by_key(SHEET_ID)

    try:
        ws = sh.worksheet("iv_data")
    except Exception:
        ws = sh.add_worksheet(title="iv_data", rows=3000, cols=20)
        ws.append_row(["Timestamp","BTC_IV","ETH_IV","BTC_ETH_Ratio",
                       "BTC_Spot","ETH_Spot","Funding_Rate","Fear_Greed",
                       "BTC_Delta","BTC_Gamma","BTC_Theta","BTC_Vega",
                       "SOL_IV","BNB_IV"])

    # データ取得
    btc_iv = get_dvol("btc"); time.sleep(0.3)
    eth_iv = get_dvol("eth"); time.sleep(0.3)
    btc_sp = get_spot("btc"); time.sleep(0.3)
    eth_sp = get_spot("eth"); time.sleep(0.3)
    funding   = get_funding(); time.sleep(0.3)
    fear_greed = get_fear_greed()

    if not btc_iv or not eth_iv:
        print("ERROR: Failed to get IV data")
        sys.exit(1)

    ts    = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    ratio = round(btc_iv / eth_iv, 6)

    row = [
        ts,
        round(btc_iv, 4),
        round(eth_iv, 4),
        ratio,
        round(btc_sp, 2) if btc_sp else "",
        round(eth_sp, 2) if eth_sp else "",
        round(funding, 6) if funding else "",
        fear_greed if fear_greed else "",
        "", "", "", "", "", ""
    ]

    # 2000行を超えたら古いデータを削除
    all_values = ws.get_all_values()
    if len(all_values) > 2000:
        # ヘッダー + 最新1900行だけ残す
        keep = [all_values[0]] + all_values[-1900:]
        ws.clear()
        ws.update("A1", keep)

    ws.append_row(row, value_input_option="USER_ENTERED")
    print(f"OK: {ts} BTC_IV={btc_iv} ETH_IV={eth_iv}")

if __name__ == "__main__":
    main()
