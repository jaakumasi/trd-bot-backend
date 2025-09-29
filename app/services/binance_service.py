from binance.client import Client
from binance.enums import *
import asyncio
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from ..config import settings
import logging

logger = logging.getLogger(__name__)


class BinanceService:
    def __init__(self):
        self.client = Client(
            api_key=settings.binance_api_key,
            api_secret=settings.binance_secret_key,
            testnet=settings.binance_testnet,
        )
        self.is_connected = False

    async def initialize(self):
        """Initialize connection and validate API keys"""
        try:
            # Test connectivity
            status = self.client.get_system_status()
            account = self.client.get_account()
            self.is_connected = True
            logger.info(
                f"✅ Binance {'Testnet' if settings.binance_testnet else 'Live'} connected successfully"
            )
            return True
        except Exception as e:
            logger.error(f"❌ Binance connection failed: {e}")
            return False

    def get_account_balance(self) -> Dict:
        """Get account balances"""
        try:
            account = self.client.get_account()
            balances = {}
            for balance in account["balances"]:
                if float(balance["free"]) > 0 or float(balance["locked"]) > 0:
                    balances[balance["asset"]] = {
                        "free": float(balance["free"]),
                        "locked": float(balance["locked"]),
                        "total": float(balance["free"]) + float(balance["locked"]),
                    }
            return balances
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return {}

    def get_symbol_price(self, symbol: str) -> float:
        """Get current price for a trading pair"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return 0.0

    def get_kline_data(
        self, symbol: str, interval: str = "1m", limit: int = 100
    ) -> pd.DataFrame:
        """Get candlestick data for technical analysis"""
        try:
            klines = self.client.get_historical_klines(
                symbol, interval, f"{limit} minutes ago UTC"
            )

            df = pd.DataFrame(
                klines,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_asset_volume",
                    "number_of_trades",
                    "taker_buy_base_asset_volume",
                    "taker_buy_quote_asset_volume",
                    "ignore",
                ],
            )

            # Convert to proper data types
            numeric_columns = ["open", "high", "low", "close", "volume"]
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            return df[["timestamp", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.error(f"Error getting kline data for {symbol}: {e}")
            return pd.DataFrame()

    def place_market_order(
        self, symbol: str, side: str, quantity: float, test_mode: bool = True
    ) -> Dict:
        """Place a market order"""
        try:
            if test_mode:
                # Simulate order for testing
                price = self.get_symbol_price(symbol)
                return {
                    "symbol": symbol,
                    "orderId": f"TEST_{asyncio.get_event_loop().time()}",
                    "side": side,
                    "type": "MARKET",
                    "quantity": str(quantity),
                    "price": str(price),
                    "status": "FILLED",
                    "executedQty": str(quantity),
                    "fills": [
                        {
                            "price": str(price),
                            "qty": str(quantity),
                            "commission": str(quantity * price * 0.001),  # 0.1% fee
                        }
                    ],
                }
            else:
                order = self.client.order_market(
                    symbol=symbol, side=side.upper(), quantity=quantity
                )
                return order
        except Exception as e:
            logger.error(f"Error placing {side} order for {symbol}: {e}")
            return {}
