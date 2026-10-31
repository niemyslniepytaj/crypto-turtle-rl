"""Konfiguracja środowiskowa test_module (klucze Binance Testnet z .env)."""

from dotenv import load_dotenv
import os
load_dotenv()

BINANCE_TESTNET_API_KEY = os.getenv("BINANCE_TESTNET_API_KEY")
BINANCE_TESTNET_API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET")
