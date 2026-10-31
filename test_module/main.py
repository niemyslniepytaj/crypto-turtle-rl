"""API HTTP test_module: endpointy Binance Testnet (buy / sell / portfolio)."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from test_module.services.core_logic.binance_testnet_service import BinanceTestnetService

app = FastAPI()
binance_testnet_service = BinanceTestnetService()


@app.get("/")
def root():
    """Endpoint kontrolny dostępności serwisu."""
    return {"message": "Hello from test_module"}


class BuyRequest(BaseModel):
    """Żądanie HTTP kupna na Binance Testnet."""
    symbol: str = "BTCUSDT"
    qty: float = 0.001


class SellRequest(BaseModel):
    """Żądanie HTTP sprzedaży na Binance Testnet."""
    symbol: str = "BTCUSDT"
    qty: float = 0.001


@app.post("/testnet/buy")
def buy_crypto(request: BuyRequest):
    try:
        result = binance_testnet_service.buy(symbol=request.symbol, qty=request.qty)
        if result.get("result") == "success":
            return {"status": "success", **result}
        return {"status": "failed", "message": result.get("message", "Unknown error")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/testnet/sell")
def sell_crypto(request: SellRequest):
    try:
        result = binance_testnet_service.sell(symbol=request.symbol, qty=request.qty)
        if result.get("result") == "success":
            return {"status": "success", **result}
        return {"status": "failed", "message": result.get("message", "Unknown error")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/testnet/portfolio")
def get_portfolio():
    try:
        result = binance_testnet_service.get_portfolio()
        if result.get("result") == "success":
            return {"status": "success", **result}
        return {"status": "failed", "message": result.get("message", "Unknown error")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
