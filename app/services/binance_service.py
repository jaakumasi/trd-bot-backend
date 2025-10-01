from binance.client import Client
from binance.enums import *
import asyncio
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from ..config import settings
import logging
import time

logger = logging.getLogger(__name__)


class BinanceService:
    def __init__(self):
        self.client = Client(
            api_key=settings.binance_api_key,
            api_secret=settings.binance_secret_key,
            testnet=settings.binance_testnet,
        )
        # Synchronize time to prevent timestamp errors
        try:
            self.client.get_server_time()
        except Exception as e:
            logger.warning(f"⚠️  Could not sync server time: {e}")
        self.is_connected = False

    async def initialize(self):
        """Initialize connection and validate API keys"""
        try:
            # Test connectivity first (no auth needed)
            status = self.client.get_system_status()
            logger.info(f"🌐 Binance {'Testnet' if settings.binance_testnet else 'Live'} network: {status}")
            
            # Get server time and adjust for timestamp sync
            server_time = self.client.get_server_time()
            local_time = int(time.time() * 1000)
            time_offset = server_time['serverTime'] - local_time
            logger.info(f"⏰ Time sync - Server: {server_time['serverTime']}, Local: {local_time}, Offset: {time_offset}ms")
            
            # Test authentication with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    account = self.client.get_account()
                    self.is_connected = True
                    logger.info(f"✅ Binance {'Testnet' if settings.binance_testnet else 'Live'} connected successfully")
                    logger.info(f"📊 Account permissions - Can Trade: {account.get('canTrade', False)}")
                    return True
                except Exception as retry_error:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        logger.warning(f"⚠️  Connection attempt {attempt + 1} failed, retrying in {wait_time}s: {retry_error}")
                        await asyncio.sleep(wait_time)
                    else:
                        raise retry_error
            
        except Exception as e:
            logger.error(f"❌ Binance connection failed: {e}")
            
            # Provide detailed error information
            error_str = str(e)
            if "-2015" in error_str:
                logger.error("💡 Error -2015 means: Invalid API key, IP restriction, or insufficient permissions")
                logger.error("🔧 Solutions:")
                logger.error("   1. Check API key is correct (no extra spaces)")
                logger.error("   2. Check secret key is correct (no extra spaces)")
                logger.error("   3. Verify API permissions include 'Enable Reading' and 'Enable Spot & Margin Trading'")
                logger.error("   4. Check IP restrictions (disable for testing)")
                logger.error(f"   5. Ensure using {'TESTNET' if settings.binance_testnet else 'MAINNET'} keys")
                logger.error("   6. Try regenerating your API keys")
            elif "connectivity" in error_str.lower() or "network" in error_str.lower():
                logger.error("💡 Network connectivity issue - check your internet connection")
            elif "-1021" in error_str:
                logger.error("💡 Error -1021 means: Timestamp for request was ahead of server time")
                logger.error("🔧 Solutions:")
                logger.error("   1. Check system clock is synchronized (Windows: w32tm /resync)")
                logger.error("   2. Try restarting the application")
                logger.error("   3. Check for network latency issues")
            
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
                # Simulate order for testing (No real money)
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

    def place_order_with_oco(
        self,
        symbol: str,
        side: str,
        quantity: float,
        take_profit_price: float,
        stop_loss_price: float,
        test_mode: bool = True,
    ) -> Dict:
        """
        Place market entry order with OCO (One-Cancels-Other) exit
        
        This creates:
        1. Market order for entry (BUY/SELL)
        2. OCO order for exit with Take Profit and Stop Loss
        
        OCO automatically executes on Binance servers:
        - If TP hits → position closes with profit, SL cancels
        - If SL hits → position closes with loss, TP cancels
        """
        try:
            if test_mode:
                # Simulate OCO for testing
                price = self.get_symbol_price(symbol)
                entry_order_id = f"TEST_ENTRY_{asyncio.get_event_loop().time()}"
                oco_order_id = f"TEST_OCO_{asyncio.get_event_loop().time()}"
                
                return {
                    "entry_order": {
                        "symbol": symbol,
                        "orderId": entry_order_id,
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
                                "commission": str(quantity * price * 0.001),
                            }
                        ],
                    },
                    "oco_order": {
                        "orderListId": oco_order_id,
                        "symbol": symbol,
                        "listOrderStatus": "EXECUTING",
                        "orders": [
                            {
                                "symbol": symbol,
                                "orderId": f"TEST_TP_{asyncio.get_event_loop().time()}",
                                "side": "SELL" if side.upper() == "BUY" else "BUY",
                                "type": "LIMIT_MAKER",
                                "price": str(take_profit_price),
                            },
                            {
                                "symbol": symbol,
                                "orderId": f"TEST_SL_{asyncio.get_event_loop().time()}",
                                "side": "SELL" if side.upper() == "BUY" else "BUY",
                                "type": "STOP_LOSS_LIMIT",
                                "stopPrice": str(stop_loss_price),
                            },
                        ],
                    },
                    "take_profit_price": take_profit_price,
                    "stop_loss_price": stop_loss_price,
                }
            else:
                # Real Binance execution
                # Step 1: Place market entry order
                entry_order = self.client.order_market(
                    symbol=symbol, side=side.upper(), quantity=quantity
                )
                
                logger.info(f"✅ Entry order filled: {entry_order['orderId']}")
                
                # Step 2: Determine exit side (opposite of entry)
                exit_side = "SELL" if side.upper() == "BUY" else "BUY"
                
                # Step 3: Calculate stop limit price (slightly worse than stop)
                if exit_side == "SELL":
                    stop_limit_price = stop_loss_price * 0.999  # 0.1% worse for SELL
                else:
                    stop_limit_price = stop_loss_price * 1.001  # 0.1% worse for BUY
                
                # Step 4: Place OCO order
                oco_order = self.client.create_oco_order(
                    symbol=symbol,
                    side=exit_side,
                    quantity=quantity,
                    price=str(take_profit_price),
                    stopPrice=str(stop_loss_price),
                    stopLimitPrice=str(stop_limit_price),
                    stopLimitTimeInForce=TIME_IN_FORCE_GTC,
                )
                
                logger.info(f"✅ OCO order created: {oco_order['orderListId']}")
                
                return {
                    "entry_order": entry_order,
                    "oco_order": oco_order,
                    "take_profit_price": take_profit_price,
                    "stop_loss_price": stop_loss_price,
                }
                
        except Exception as e:
            logger.error(f"Error placing OCO order for {symbol}: {e}")
            return {}

    def get_oco_order_status(self, order_list_id: str) -> Optional[Dict]:
        """
        Check the status of an OCO order
        
        Returns:
        - Status of the OCO order (EXECUTING, ALL_DONE, REJECTED)
        - Which leg executed (TP or SL)
        - Execution details
        """
        try:
            # For test orders, simulate status
            if str(order_list_id).startswith("TEST_"):
                return {
                    "orderListId": order_list_id,
                    "listOrderStatus": "EXECUTING",  # Simulated as still active
                    "orders": []
                }
            
            # Real Binance query
            result = self.client.get_order_list(orderListId=int(order_list_id))
            return result
            
        except Exception as e:
            logger.error(f"Error getting OCO status for {order_list_id}: {e}")
            return None

    def cancel_oco_order(self, symbol: str, order_list_id: str) -> bool:
        """Cancel an active OCO order"""
        try:
            if str(order_list_id).startswith("TEST_"):
                logger.info(f"✅ Test OCO order cancelled: {order_list_id}")
                return True
            
            result = self.client.cancel_order_list(
                symbol=symbol,
                orderListId=int(order_list_id)
            )
            logger.info(f"✅ OCO order cancelled: {order_list_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling OCO order {order_list_id}: {e}")
            return False
