from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import asyncio
import logging
from datetime import datetime
import random
from ..config import settings

logger = logging.getLogger(__name__)


class MockBinanceService:
    """Mock Binance service for testing without real API keys"""

    def __init__(self):
        self.is_connected = False
        self.mock_balances = {
            "USDT": {"free": 1000.0, "locked": 0.0, "total": 1000.0},
            "BTC": {"free": 0.01, "locked": 0.0, "total": 0.01},
            "ETH": {"free": 0.5, "locked": 0.0, "total": 0.5},
        }
        logger.info("🧪 MockBinanceService initialized - This is SIMULATION mode!")

    async def initialize(self):
        """Initialize mock connection"""
        try:
            logger.info("✅ Mock Binance connected successfully (SIMULATION MODE)")
            logger.info("💰 Mock balances: USDT: $1000, BTC: 0.01, ETH: 0.5")
            self.is_connected = True
            return True
        except Exception as e:
            logger.error(f"❌ Mock Binance initialization failed: {e}")
            return False

    def get_account_balance(self) -> Dict:
        """Get mock account balances"""
        logger.debug("📊 Returning mock account balances")
        return self.mock_balances.copy()

    def get_symbol_price(self, symbol: str) -> float:
        """Get mock price for a trading pair"""
        # Generate realistic mock prices
        mock_prices = {
            "BTCUSDT": 65000 + random.uniform(-5000, 5000),
            "ETHUSDT": 2800 + random.uniform(-300, 300),
            "ADAUSDT": 0.45 + random.uniform(-0.05, 0.05),
            "BNBUSDT": 320 + random.uniform(-30, 30),
            "SOLUSDT": 140 + random.uniform(-20, 20),
            "DOTUSDT": 5.5 + random.uniform(-0.5, 0.5),
        }

        price = mock_prices.get(symbol, 100 + random.uniform(-10, 10))
        logger.debug(f"💰 Mock price for {symbol}: ${price:.4f}")
        return price

    def get_kline_data(
        self, symbol: str, interval: str = "1m", limit: int = 100
    ) -> pd.DataFrame:
        """Generate mock candlestick data for technical analysis"""
        try:
            base_price = self.get_symbol_price(symbol)

            # Map interval string to pandas frequency
            interval_map = {
                "1m": "1min",
                "3m": "3min",
                "5m": "5min",
                "15m": "15min",
                "30m": "30min",
                "1h": "1H",
                "2h": "2H",
                "4h": "4H",
                "6h": "6H",
                "12h": "12H",
                "1d": "1D",
            }
            pandas_freq = interval_map.get(interval, "1min")

            # Generate realistic OHLCV data
            timestamps = pd.date_range(
                end=datetime.now(), periods=limit, freq=pandas_freq
            )

            data = []
            # Start near the base price with slight variation
            current_price = base_price

            # Create a trend component (random walk with mean reversion)
            trend = random.uniform(-0.0002, 0.0002)  # Very small trend

            for i, ts in enumerate(timestamps):
                # Mean reversion: pull price back toward base
                mean_reversion = (base_price - current_price) * 0.01

                # Combine trend + noise + mean reversion
                noise = random.uniform(-0.0002, 0.0002)
                change = trend + noise + mean_reversion
                current_price *= 1 + change

                # Realistic intracandle volatility (0.02-0.05% range)
                wick_size = current_price * random.uniform(0.0002, 0.0005)

                # Generate OHLC
                open_price = current_price
                close_price = current_price * (1 + random.uniform(-0.0003, 0.0003))
                high_price = max(open_price, close_price) + wick_size
                low_price = min(open_price, close_price) - wick_size
                volume = random.uniform(5000, 15000)

                data.append(
                    {
                        "timestamp": ts,
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "close": close_price,
                        "volume": volume,
                    }
                )

            df = pd.DataFrame(data)
            logger.debug(
                f"📊 Generated {len(df)} mock {interval} candlesticks for {symbol}"
            )
            return df

        except Exception as e:
            logger.error(f"Error generating mock kline data for {symbol}: {e}")
            return pd.DataFrame()

    def place_market_order(self, symbol: str, side: str, quantity: float) -> Dict:
        """Simulate placing a market order (MockBinanceService always simulates)"""
        try:
            price = self.get_symbol_price(symbol)
            order_id = int(random.random() * 1000000)
            executed_qty = quantity * random.uniform(0.98, 1.0)  # Slight slippage
            commission = executed_qty * price * settings.fee_estimate_pct

            # Update mock balances
            if side.upper() == "BUY":
                cost = executed_qty * price + commission
                if "USDT" in self.mock_balances:
                    self.mock_balances["USDT"]["free"] -= cost

                base_asset = symbol.replace("USDT", "")
                if base_asset not in self.mock_balances:
                    self.mock_balances[base_asset] = {
                        "free": 0,
                        "locked": 0,
                        "total": 0,
                    }
                self.mock_balances[base_asset]["free"] += executed_qty

            elif side.upper() == "SELL":
                base_asset = symbol.replace("USDT", "")
                if base_asset in self.mock_balances:
                    self.mock_balances[base_asset]["free"] -= executed_qty

                revenue = executed_qty * price - commission
                if "USDT" not in self.mock_balances:
                    self.mock_balances["USDT"] = {"free": 0, "locked": 0, "total": 0}
                self.mock_balances["USDT"]["free"] += revenue

            mock_order = {
                "symbol": symbol,
                "orderId": order_id,
                "side": side.upper(),
                "type": "MARKET",
                "quantity": str(quantity),
                "price": str(price),
                "status": "FILLED",
                "executedQty": str(executed_qty),
                "fills": [
                    {
                        "price": str(
                            price * random.uniform(0.9999, 1.0001)
                        ),  # Small price variation
                        "qty": str(executed_qty),
                        "commission": str(commission),
                        "commissionAsset": "USDT",
                    }
                ],
            }

            logger.info(
                f"🧪 MOCK ORDER: {side} {executed_qty:.6f} {symbol} @ ${price:.4f} (Order ID: {order_id})"
            )
            return mock_order

        except Exception as e:
            logger.error(f"Error simulating {side} order for {symbol}: {e}")
            return {}

    def place_order_with_oco(
        self,
        symbol: str,
        side: str,
        quantity: float,
        take_profit_price: float,
        stop_loss_price: float,
    ) -> Dict:
        """
        Simulate placing an OCO order (entry + TP/SL)

        MockBinanceService always simulates:
        1. Entry order execution
        2. OCO order creation on "Binance" (simulated)
        """
        try:
            # Step 1: Execute entry order (same as place_market_order)
            price = self.get_symbol_price(symbol)
            entry_order_id = int(random.random() * 1000000)
            executed_qty = quantity * random.uniform(0.98, 1.0)
            commission = executed_qty * price * settings.fee_estimate_pct

            # Update mock balances
            if side.upper() == "BUY":
                cost = executed_qty * price + commission
                if "USDT" in self.mock_balances:
                    self.mock_balances["USDT"]["free"] -= cost

                base_asset = symbol.replace("USDT", "")
                if base_asset not in self.mock_balances:
                    self.mock_balances[base_asset] = {
                        "free": 0,
                        "locked": 0,
                        "total": 0,
                    }
                self.mock_balances[base_asset]["free"] += executed_qty

            elif side.upper() == "SELL":
                base_asset = symbol.replace("USDT", "")
                if base_asset in self.mock_balances:
                    self.mock_balances[base_asset]["free"] -= executed_qty

                revenue = executed_qty * price - commission
                if "USDT" not in self.mock_balances:
                    self.mock_balances["USDT"] = {"free": 0, "locked": 0, "total": 0}
                self.mock_balances["USDT"]["free"] += revenue

            # Step 2: Create mock OCO order
            oco_order_id = int(random.random() * 1000000)
            exit_side = "SELL" if side.upper() == "BUY" else "BUY"

            entry_order = {
                "symbol": symbol,
                "orderId": entry_order_id,
                "side": side.upper(),
                "type": "MARKET",
                "quantity": str(quantity),
                "price": str(price),
                "status": "FILLED",
                "executedQty": str(executed_qty),
                "fills": [
                    {
                        "price": str(price),
                        "qty": str(executed_qty),
                        "commission": str(commission),
                        "commissionAsset": "USDT",
                    }
                ],
            }

            oco_order = {
                "orderListId": oco_order_id,
                "symbol": symbol,
                "listOrderStatus": "EXECUTING",
                "orders": [
                    {
                        "symbol": symbol,
                        "orderId": int(random.random() * 1000000),
                        "side": exit_side,
                        "type": "LIMIT_MAKER",
                        "price": str(take_profit_price),
                        "quantity": str(executed_qty),
                        "status": "NEW",
                    },
                    {
                        "symbol": symbol,
                        "orderId": int(random.random() * 1000000),
                        "side": exit_side,
                        "type": "STOP_LOSS_LIMIT",
                        "stopPrice": str(stop_loss_price),
                        "price": str(
                            stop_loss_price * 0.999
                            if exit_side == "SELL"
                            else stop_loss_price * 1.001
                        ),
                        "quantity": str(executed_qty),
                        "status": "NEW",
                    },
                ],
            }

            logger.info(f"🧪 MOCK OCO ORDER:")
            logger.info(
                f"   📥 Entry: {side} {executed_qty:.6f} {symbol} @ ${price:.4f}"
            )
            logger.info(f"   🎯 Take Profit: {exit_side} @ ${take_profit_price:.4f}")
            logger.info(f"   🛑 Stop Loss: {exit_side} @ ${stop_loss_price:.4f}")
            logger.info(f"   🆔 OCO Order List ID: {oco_order_id}")

            return {
                "entry_order": entry_order,
                "oco_order": oco_order,
                "take_profit_price": take_profit_price,
                "stop_loss_price": stop_loss_price,
            }

        except Exception as e:
            logger.error(f"Error simulating OCO order for {symbol}: {e}")
            return {}

    def get_oco_order_status(self, order_list_id: str) -> Optional[Dict]:
        """
        Simulate checking OCO order status

        In mock mode, this will simulate the order as EXECUTING
        (position monitoring will happen via price checks in mock mode)
        """
        try:
            logger.debug(f"🧪 Checking mock OCO status: {order_list_id}")
            return {
                "orderListId": order_list_id,
                "listOrderStatus": "EXECUTING",  # Always executing in mock
                "orders": [],
            }
        except Exception as e:
            logger.error(f"Error getting mock OCO status: {e}")
            return None

    def cancel_oco_order(self, symbol: str, order_list_id: str) -> bool:
        """Simulate cancelling an OCO order"""
        try:
            logger.info(f"🧪 MOCK: Cancelled OCO order {order_list_id} for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling mock OCO order: {e}")
            return False
