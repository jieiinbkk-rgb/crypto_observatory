"""
全設定の一元管理
"""

APP_VERSION     = "3.0"
INITIAL_CAPITAL = 100_000
MAX_POSITIONS   = 5
IV_RV_ALERT     = 5.0          # IV-RV スプレッド警告閾値

# Google Sheets ID（secrets.toml で上書き可能）
SHEET_ID = "1C4Gd0AqHcMNg-QHhMCGoNzONyse5NMBGCfHjcuVcOKY"

# ── マーケット状態定義 ─────────────────────────────────────────
MARKET_STATES = {
    "risk_on": {
        "label":       "Risk-On",
        "color":       "#00c896",
        "description": "IV低下・クロス相関高・穏やかな市場",
        "strategy":    "calendar_spread",
        "emoji":       "🟢",
    },
    "hedging": {
        "label":       "Hedging",
        "color":       "#4b9eff",
        "description": "BTC IV上昇・ETH相対安定・ヘッジ需要",
        "strategy":    "ratio_spread",
        "emoji":       "🔵",
    },
    "squeeze": {
        "label":       "Vol Squeeze",
        "color":       "#ffd700",
        "description": "IV急激に低下・ブレイクアウト前兆",
        "strategy":    "butterfly",
        "emoji":       "🟡",
    },
    "panic": {
        "label":       "Panic",
        "color":       "#ff4b4b",
        "description": "IV急上昇・スプレッド拡大・リスクオフ",
        "strategy":    "tail_hedge",
        "emoji":       "🔴",
    },
    "unknown": {
        "label":       "Unknown",
        "color":       "#888888",
        "description": "分類不能・データ不足",
        "strategy":    None,
        "emoji":       "⚪",
    },
}

# ── 戦略定義 ──────────────────────────────────────────────────
STRATEGIES = {
    "vol_sell": {
        "name":           "Volatility Sell",
        "action":         "SHORT VOL",
        "legs":           ["Short Straddle", "Iron Condor"],
        "target_pnl_pct": 0.15,
        "stop_pnl_pct":   -0.25,
        "rationale":      "IV が RV を大幅に上回る局面でのプレミアム売り",
        "risk":           "ガンマリスク・急騰時の損失無限大",
    },
    "ratio_spread": {
        "name":           "BTC/ETH Ratio Spread",
        "action":         "RATIO TRADE",
        "legs":           ["Long BTC Call", "Short ETH Call"],
        "target_pnl_pct": 0.12,
        "stop_pnl_pct":   -0.15,
        "rationale":      "BTC/ETH IV 比率の平均回帰を狙ったスプレッド",
        "risk":           "相関崩壊リスク・レグ間のスリッページ",
    },
    "gamma_long": {
        "name":           "Gamma Long",
        "action":         "LONG VOL",
        "legs":           ["Long Straddle", "Long Strangle"],
        "target_pnl_pct": 0.30,
        "stop_pnl_pct":   -0.20,
        "rationale":      "Vol スクイーズからのブレイクアウトを期待したガンマ買い",
        "risk":           "時間価値の減衰（セータ）・方向性リスク",
    },
    "calendar_spread": {
        "name":           "Calendar Spread",
        "action":         "SPREAD",
        "legs":           ["Short Near-term", "Long Far-term"],
        "target_pnl_pct": 0.10,
        "stop_pnl_pct":   -0.08,
        "rationale":      "短期IVが長期IVより高い局面での時間軸スプレッド",
        "risk":           "方向性リスク・流動性リスク",
    },
    "butterfly": {
        "name":           "Butterfly Spread",
        "action":         "BUY",
        "legs":           ["Long Lower Strike", "Short 2x ATM", "Long Upper Strike"],
        "target_pnl_pct": 0.20,
        "stop_pnl_pct":   -0.10,
        "rationale":      "IV低下局面でのレンジ内収束を狙う",
        "risk":           "レンジ外への大きな動きで全損",
    },
    "tail_hedge": {
        "name":           "Tail Hedge",
        "action":         "BUY PUTS",
        "legs":           ["Long OTM Put", "Put Spread"],
        "target_pnl_pct": 0.50,
        "stop_pnl_pct":   -0.50,
        "rationale":      "パニック相場での下落ヘッジ・テール保護",
        "risk":           "プレミアム消費・回復時の全損",
    },
}

# ── ボラティリティ・レジーム定義 ───────────────────────────────
VOL_REGIMES = {
    "very_low": {
        "label": "Very Low Vol",
        "color": "#00c896",
        "pct_range": (0, 20),
    },
    "low": {
        "label": "Low Vol",
        "color": "#4b9eff",
        "pct_range": (20, 40),
    },
    "normal": {
        "label": "Normal Vol",
        "color": "#ffd700",
        "pct_range": (40, 60),
    },
    "high": {
        "label": "High Vol",
        "color": "#ff9900",
        "pct_range": (60, 80),
    },
    "very_high": {
        "label": "Very High Vol",
        "color": "#ff4b4b",
        "pct_range": (80, 100),
    },
}
