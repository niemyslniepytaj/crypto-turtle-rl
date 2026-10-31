"""Klient Binance Testnet: zlecenia market i odczyt portfela."""

from typing import Optional

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException

from configuration.configuration import BINANCE_TESTNET_API_KEY, BINANCE_TESTNET_API_SECRET


class BinanceTestnetService:
    """Opakowanie python-binance dla środowiska testnet (buy / sell / portfolio)."""

    def __init__(self):
        self._client: Optional[Client] = None

    def _get_client(self) -> Client:
        if self._client is None:
            if not BINANCE_TESTNET_API_KEY or not BINANCE_TESTNET_API_SECRET:
                raise ValueError("BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET must be set in .env")
            self._client = Client(
                api_key=BINANCE_TESTNET_API_KEY,
                api_secret=BINANCE_TESTNET_API_SECRET,
                testnet=True
            )
        return self._client

    def _format_qty(self, symbol: str, qty: float) -> float:
        try:
            info = self._get_client().get_exchange_info()
            for s in info.get("symbols", []):
                if s["symbol"] == symbol.upper():
                    for f in s.get("filters", []):
                        if f.get("filterType") == "LOT_SIZE":
                            step = float(f.get("stepSize", "0.001"))
                            return float(int(qty / step) * step)
        except Exception:
            pass
        return qty

    def buy(self, symbol: str, qty: float) -> dict:
        try:
            client = self._get_client()
            symbol = symbol.upper()
            qty = self._format_qty(symbol, qty)
            if qty <= 0:
                return {"result": "failed", "message": "Invalid quantity"}
            result = client.create_order(
                symbol=symbol,
                side=Client.SIDE_BUY,
                type=Client.ORDER_TYPE_MARKET,
                quantity=qty
            )
            fills = result.get("fills", [])
            price = float(fills[0]["price"]) if fills else 0
            return {
                "result": "success",
                "orderId": result.get("orderId"),
                "symbol": symbol,
                "qty": qty,
                "executed_price": price,
            }
        except (BinanceOrderException, BinanceAPIException, Exception) as e:
            return {"result": "failed", "message": str(e)}

    def sell(self, symbol: str, qty: float) -> dict:
        try:
            client = self._get_client()
            symbol = symbol.upper()
            qty = self._format_qty(symbol, qty)
            if qty <= 0:
                return {"result": "failed", "message": "Invalid quantity"}
            result = client.create_order(
                symbol=symbol,
                side=Client.SIDE_SELL,
                type=Client.ORDER_TYPE_MARKET,
                quantity=qty
            )
            fills = result.get("fills", [])
            price = float(fills[0]["price"]) if fills else 0
            return {
                "result": "success",
                "orderId": result.get("orderId"),
                "symbol": symbol,
                "qty": qty,
                "executed_price": price,
            }
        except (BinanceOrderException, BinanceAPIException, Exception) as e:
            return {"result": "failed", "message": str(e)}

    def get_portfolio(self) -> dict:
        try:
            client = self._get_client()
            account = client.get_account()
            balances = []
            usdt_balance = 0.0
            for b in account.get("balances", []):
                free = float(b.get("free", 0))
                locked = float(b.get("locked", 0))
                total = free + locked
                if total > 0:
                    asset = b["asset"]
                    if asset == "USDT":
                        usdt_balance = free
                    balances.append({
                        "asset": asset,
                        "free": free,
                        "locked": locked,
                        "total": total,
                    })
            return {
                "result": "success",
                "usdt_balance": usdt_balance,
                "balances": balances,
            }
        except Exception as e:
            return {"result": "failed", "message": str(e)}