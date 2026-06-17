"""
execution/trading_interface.py
将来の Deribit Live Trading 用スタブ
実装手順はこのファイルの docstring を参照
"""
import streamlit as st


class TradingInterface:
    """
    Live Trading への移行手順:
    1. Deribit API キー取得（read + trade 権限）
    2. .streamlit/secrets.toml に以下を追加:
       DERIBIT_API_KEY = "..."
       DERIBIT_API_SECRET = "..."
    3. DeribitLiveTradingEngine を実装し、以下メソッドを定義:
       - place_order(instrument, amount, type, label) -> order_id
       - cancel_order(order_id)
       - get_positions() -> list
       - get_account_summary() -> dict
    4. paper_trade/engine.py の open_trade() → place_order() に差し替え
    """

    def __init__(self):
        self._live = False
        self._api_key = st.secrets.get("DERIBIT_API_KEY", "")

    @property
    def is_live(self) -> bool:
        return self._live and bool(self._api_key)

    def status(self) -> dict:
        return {
            "mode":       "paper_trade" if not self.is_live else "live",
            "connected":  False,
            "api_key_set": bool(self._api_key),
            "note":       "Live trading not yet implemented",
        }

    def place_order(self, *args, **kwargs) -> dict:
        raise NotImplementedError("Live trading not implemented. Use paper_trade mode.")
