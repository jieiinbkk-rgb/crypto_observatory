import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import time
import os
from threading import Thread

CSV_FILE = "iv_data.csv"

# --- 🕵️ 裏方ボットをバックグラウンドで動かす魔法（スレッド制御） ---
@st.cache_resource
def launch_background_bot():
    """アプリ起動時に、裏方ボットを1つだけ完全にバックグラウンドで永続走行させる関数"""
    # CSVの初期化
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame(columns=["Timestamp", "BTC_IV", "ETH_IV", "BTC_ETH_Ratio"])
        df.to_csv(CSV_FILE, index=False)

    def get_dvol(crypto_symbol):
        index_name = f"{crypto_symbol.lower()}dvol_usdc"
        url = f"https://www.deribit.com/api/v2/public/get_index_price?index_name={index_name}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            # クラウド回線は爆速なのでタイムアウトは5秒で十分
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                return response.json()['result']['index_price']
        except Exception:
            return None

    def bot_loop():
        while True:
            btc_iv = get_dvol("btc")
            time.sleep(1) # 連続アクセス回避の休憩
            eth_iv = get_dvol("eth")
            
            if btc_iv and eth_iv:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ratio = btc_iv / eth_iv
                # CSVファイルに追記
                new_row = pd.DataFrame([[now, btc_iv, eth_iv, ratio]])
                new_row.to_csv(CSV_FILE, mode='a', header=False, index=False)
            
            # きっちり60秒待機
            time.sleep(60)

    # バックグラウンド（裏の別部屋）でbot_loopを動かし続ける
    t = Thread(target=bot_loop, daemon=True)
    t.start()
    return t

# アプリ起動時にボットを裏で自動スタート（2回目以降のアクセスでは重複起動しない）
launch_background_bot()

# --- 📊 表舞台（ダッシュボード画面） ---
st.set_page_config(page_title="Crypto Options Observatory", layout="wide")
st.title("🔭 Crypto Options Observatory v0.1")

if not os.path.exists(CSV_FILE) or pd.read_csv(CSV_FILE).empty:
    st.info("🔄 クラウド回線で最初のデータを取得中です。1分ほどお待ちいただき、画面を更新してください。")
    time.sleep(10)
    st.rerun()
else:
    df = pd.read_csv(CSV_FILE)
    
    st.subheader("リアルタイム IV データ (1分更新)")
    
    # 最新の数値を大きく表示
    col1, col2, col3 = st.columns(3)
    col1.metric(label="最新 BTC ATM IV", value=f"{df['BTC_IV'].iloc[-1]:.2f}")
    col2.metric(label="最新 ETH ATM IV", value=f"{df['ETH_IV'].iloc[-1]:.2f}")
    col3.metric(label="BTC/ETH IV Ratio", value=f"{df['BTC_ETH_Ratio'].iloc[-1]:.4f}")

    st.divider()

    st.subheader("IV の推移")
    chart_data = df.set_index("Timestamp")
    st.line_chart(chart_data[["BTC_IV", "ETH_IV"]])

    st.subheader("BTC/ETH IV Ratio の推移")
    st.line_chart(chart_data[["BTC_ETH_Ratio"]], color="#ffaa00")

# 10秒ごとに自動で画面だけを再描画
time.sleep(10)
st.rerun()