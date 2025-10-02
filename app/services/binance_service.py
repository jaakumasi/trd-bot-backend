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
        self.is_connected = False
        self.symbol_info_cache = {}  # Cache for symbol trading rules

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get trading rules and filters for a symbol"""
        try:
            # Check cache first
            if symbol in self.symbol_info_cache:
                return self.symbol_info_cache[symbol]
            
            exchange_info = self.client.get_symbol_info(symbol)
            if exchange_info:
                self.symbol_info_cache[symbol] = exchange_info
                return exchange_info
            return None
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None

    def format_quantity(self, symbol: str, quantity: float) -> float:
        """
        Format quantity to meet Binance LOT_SIZE filter requirements
        
        LOT_SIZE filter requires:
        - quantity >= minQty
        - quantity <= maxQty
        - (quantity - minQty) % stepSize == 0
        """
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                logger.warning(f"⚠️  Could not get symbol info for {symbol}, using quantity as-is")
                return quantity
            
            # Find LOT_SIZE filter
            lot_size_filter = None
            for f in symbol_info.get('filters', []):
                if f['filterType'] == 'LOT_SIZE':
                    lot_size_filter = f
                    break
            
            if not lot_size_filter:
                logger.warning(f"⚠️  No LOT_SIZE filter found for {symbol}, using quantity as-is")
                return quantity
            
            min_qty = float(lot_size_filter['minQty'])
            max_qty = float(lot_size_filter['maxQty'])
            step_size = float(lot_size_filter['stepSize'])
            
            # Calculate precision from step_size
            # e.g., step_size = 0.00100000 -> precision = 3
            step_size_str = f"{step_size:.8f}".rstrip('0')
            precision = len(step_size_str.split('.')[-1]) if '.' in step_size_str else 0
            
            # Round quantity to step_size precision
            formatted_qty = round(quantity, precision)
            
            # Ensure it meets min/max requirements
            if formatted_qty < min_qty:
                logger.warning(f"⚠️  Quantity {formatted_qty} below minQty {min_qty}, adjusting to {min_qty}")
                formatted_qty = min_qty
            elif formatted_qty > max_qty:
                logger.warning(f"⚠️  Quantity {formatted_qty} above maxQty {max_qty}, adjusting to {max_qty}")
                formatted_qty = max_qty
            
            # Ensure it aligns with step_size
            # Formula: floor((quantity - minQty) / stepSize) * stepSize + minQty
            steps = int((formatted_qty - min_qty) / step_size)
            formatted_qty = steps * step_size + min_qty
            formatted_qty = round(formatted_qty, precision)
            
            logger.debug(f"📏 Quantity formatting: {quantity} -> {formatted_qty} (min={min_qty}, max={max_qty}, step={step_size})")
            
            return formatted_qty
            
        except Exception as e:
            logger.error(f"Error formatting quantity for {symbol}: {e}")
            return quantity

    async def initialize(self):
        """Initialize connection and validate API keys"""
        try:
            # Test connectivity first (no auth needed)
            status = self.client.get_system_status()
            logger.info(f"🌐 Binance {'Testnet' if settings.binance_testnet else 'Live'} network: {status}")
            
            # Get server time and calculate offset
            server_time = self.client.get_server_time()
            local_time = int(time.time() * 1000)
            time_offset = server_time['serverTime'] - local_time
            logger.info(f"⏰ Time sync - Server: {server_time['serverTime']}, Local: {local_time}, Offset: {time_offset}ms")
            
            # Apply timestamp offset to client if significant (>1000ms)
            if abs(time_offset) > 1000:
                logger.warning(f"⚠️  Large time offset detected ({time_offset}ms), adjusting client timestamp")
                self.client.timestamp_offset = time_offset
            
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
                    error_str = str(retry_error)
                    # If it's a timestamp error, try adjusting the offset
                    if "-1021" in error_str and attempt < max_retries - 1:
                        logger.warning(f"⚠️  Timestamp error on attempt {attempt + 1}, recalculating offset...")
                        # Recalculate server time offset
                        server_time = self.client.get_server_time()
                        local_time = int(time.time() * 1000)
                        time_offset = server_time['serverTime'] - local_time
                        self.client.timestamp_offset = time_offset
                        logger.info(f"🔄 Updated timestamp offset to {time_offset}ms")
                        wait_time = (attempt + 1) * 2
                        await asyncio.sleep(wait_time)
                    elif attempt < max_retries - 1:
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
        self, symbol: str, side: str, quantity: float
    ) -> Dict:
        """
        Place a market order on Binance (Testnet or Mainnet based on settings)
        
        The binance_testnet setting determines which API to use:
        - binance_testnet=True → Uses Binance Testnet (safe for testing)
        - binance_testnet=False → Uses Binance Mainnet (real money)
        """
        try:
            # Format quantity to meet Binance LOT_SIZE requirements
            formatted_qty = self.format_quantity(symbol, quantity)
            
            order = self.client.order_market(
                symbol=symbol, side=side.upper(), quantity=formatted_qty
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
    ) -> Dict:
        """
        Place market entry order with OCO (One-Cancels-Other) exit on Binance
        
        The binance_testnet setting determines which API to use:
        - binance_testnet=True → Uses Binance Testnet (safe for testing)
        - binance_testnet=False → Uses Binance Mainnet (real money)
        
        This creates:
        1. Market order for entry (BUY/SELL)
        2. OCO order for exit with Take Profit and Stop Loss
        
        OCO automatically executes on Binance servers:
        - If TP hits → position closes with profit, SL cancels
        - If SL hits → position closes with loss, TP cancels
        """
        try:
            # Format quantity to meet Binance LOT_SIZE requirements
            formatted_qty = self.format_quantity(symbol, quantity)
            
            # Step 1: Place market entry order
            entry_order = self.client.order_market(
                symbol=symbol, side=side.upper(), quantity=formatted_qty
            )
            
            logger.info(f"✅ Entry order filled: {entry_order['orderId']}")
            
            # Step 2: Determine exit side (opposite of entry)
            exit_side = "SELL" if side.upper() == "BUY" else "BUY"
            
            # Step 3: Get executed quantity from entry order (may differ from requested)
            executed_qty = float(entry_order.get('executedQty', formatted_qty))
            
            # Step 4: Calculate stop limit price (slightly worse than stop)
            if exit_side == "SELL":
                stop_limit_price = stop_loss_price * 0.999  # 0.1% worse for SELL
            else:
                stop_limit_price = stop_loss_price * 1.001  # 0.1% worse for BUY
            
            # Step 5: Determine which price is above/below for OCO
            # Above = higher price, Below = lower price
            # 
            # For SELL exit: We're selling, so TP (higher price) is above, SL (lower price) is below
            # For BUY exit: We're buying, so SL (higher price) is above, TP (lower price) is below
            if exit_side == "SELL":
                # SELL exit: Taking profit at higher price, stop loss at lower price
                # Example: Bought @ $50k, Sell TP @ $51k (above), Sell SL @ $49k (below)
                above_price = take_profit_price  # Higher price (take profit)
                above_type = "LIMIT_MAKER"
                below_price = stop_loss_price  # Lower price (stop loss)
                below_type = "STOP_LOSS_LIMIT"
            else:
                # BUY exit: Taking profit at lower price, stop loss at higher price
                # Example: Sold @ $118k, Buy TP @ $117k (below), Buy SL @ $119k (above)
                above_price = stop_loss_price  # Higher price (stop loss)
                above_type = "STOP_LOSS_LIMIT"
                below_price = take_profit_price  # Lower price (take profit)
                below_type = "LIMIT_MAKER"
            
            # Step 6: Place OCO order with new format
            oco_order = self.client.create_oco_order(
                symbol=symbol,
                side=exit_side,
                quantity=executed_qty,
                aboveType=above_type,
                abovePrice=str(above_price),
                aboveStopPrice=str(above_price) if above_type == "STOP_LOSS_LIMIT" else None,
                belowType=below_type,
                belowPrice=str(below_price),
                belowStopPrice=str(below_price) if below_type == "STOP_LOSS_LIMIT" else None,
                aboveTimeInForce=TIME_IN_FORCE_GTC if above_type == "STOP_LOSS_LIMIT" else None,
                belowTimeInForce=TIME_IN_FORCE_GTC if below_type == "STOP_LOSS_LIMIT" else None,
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
        Check the status of an OCO order on Binance
        
        Returns:
        - Status of the OCO order (EXECUTING, ALL_DONE, REJECTED)
        - Which leg executed (TP or SL)
        - Execution details
        """
        try:
            # Use v3_get_order_list as per python-binance API documentation
            result = self.client.v3_get_order_list(orderListId=int(order_list_id))
            return result
            
        except Exception as e:
            logger.error(f"Error getting OCO status for {order_list_id}: {e}")
            return None

    def cancel_oco_order(self, symbol: str, order_list_id: str) -> bool:
        """Cancel an active OCO order on Binance"""
        try:
            self.client.cancel_order_list(
                symbol=symbol,
                orderListId=int(order_list_id)
            )
            logger.info(f"✅ OCO order cancelled: {order_list_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling OCO order {order_list_id}: {e}")
            return False
