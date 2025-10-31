from binance.client import Client
from binance.enums import *
import asyncio
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from ..config import settings
from .service_constants import (
    OCO_BUY_STOP_LIMIT_BUFFER,
    OCO_SELL_STOP_LIMIT_BUFFER,
)
import logging
import time

logger = logging.getLogger(__name__)


class BinanceService:
    def __init__(self):
        # Get appropriate API keys based on trading mode
        api_key, secret_key = settings.get_active_binance_keys()
        
        # Determine testnet flag based on trading mode
        use_testnet = settings.trading_mode == settings.trading_mode.TESTNET
        
        self.client = Client(
            api_key=api_key,
            api_secret=secret_key,
            testnet=use_testnet,
        )
        # Increase recvWindow to allow for more clock variance (default is 5000ms, we use 60000ms)
        self.client.RECV_WINDOW = 60_000
        self.is_connected = False
        self.symbol_info_cache = {}
        self.timestamp_offset = 0
        self.last_time_sync = 0
        
        # Log which network we're using
        network = settings.get_environment_name()
        logger.info(f"🌐 BinanceService initialized for {network}")

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
                logger.warning(
                    f"⚠️  Could not get symbol info for {symbol}, using quantity as-is"
                )
                return quantity

            # Find LOT_SIZE filter
            lot_size_filter = None
            for f in symbol_info.get("filters", []):
                if f["filterType"] == "LOT_SIZE":
                    lot_size_filter = f
                    break

            if not lot_size_filter:
                logger.warning(
                    f"⚠️  No LOT_SIZE filter found for {symbol}, using quantity as-is"
                )
                return quantity

            min_qty = float(lot_size_filter["minQty"])
            max_qty = float(lot_size_filter["maxQty"])
            step_size = float(lot_size_filter["stepSize"])

            # Calculate precision from step_size
            # e.g., step_size = 0.00100000 -> precision = 3
            step_size_str = f"{step_size:.8f}".rstrip("0")
            precision = len(step_size_str.split(".")[-1]) if "." in step_size_str else 0

            # Round quantity to step_size precision
            formatted_qty = round(quantity, precision)

            # Ensure it meets min/max requirements
            if formatted_qty < min_qty:
                logger.warning(
                    f"⚠️  Quantity {formatted_qty} below minQty {min_qty}, adjusting to {min_qty}"
                )
                formatted_qty = min_qty
            elif formatted_qty > max_qty:
                logger.warning(
                    f"⚠️  Quantity {formatted_qty} above maxQty {max_qty}, adjusting to {max_qty}"
                )
                formatted_qty = max_qty

            # Ensure it aligns with step_size
            # Formula: floor((quantity - minQty) / stepSize) * stepSize + minQty
            steps = int((formatted_qty - min_qty) / step_size)
            formatted_qty = steps * step_size + min_qty
            formatted_qty = round(formatted_qty, precision)

            logger.debug(
                f"📏 Quantity formatting: {quantity} -> {formatted_qty} (min={min_qty}, max={max_qty}, step={step_size})"
            )

            return formatted_qty

        except Exception as e:
            logger.error(f"Error formatting quantity for {symbol}: {e}")
            return quantity

    def format_price(self, symbol: str, price: float) -> float:
        """
        Format price to meet Binance PRICE_FILTER requirements

        PRICE_FILTER requires:
        - price >= minPrice
        - price <= maxPrice
        - (price - minPrice) % tickSize == 0
        """
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                logger.warning(
                    f"⚠️  Could not get symbol info for {symbol}, using price as-is"
                )
                return price

            # Find PRICE_FILTER
            price_filter = None
            for f in symbol_info.get("filters", []):
                if f["filterType"] == "PRICE_FILTER":
                    price_filter = f
                    break

            if not price_filter:
                logger.warning(
                    f"⚠️  No PRICE_FILTER found for {symbol}, using price as-is"
                )
                return price

            min_price = float(price_filter["minPrice"])
            max_price = float(price_filter["maxPrice"])
            tick_size = float(price_filter["tickSize"])

            # Calculate precision from tick_size
            tick_size_str = f"{tick_size:.8f}".rstrip("0")
            precision = len(tick_size_str.split(".")[-1]) if "." in tick_size_str else 0

            # Round price to tick_size precision
            formatted_price = round(price, precision)

            # Ensure it meets min/max requirements
            if formatted_price < min_price:
                logger.warning(
                    f"⚠️  Price {formatted_price} below minPrice {min_price}, adjusting to {min_price}"
                )
                formatted_price = min_price
            elif formatted_price > max_price:
                logger.warning(
                    f"⚠️  Price {formatted_price} above maxPrice {max_price}, adjusting to {max_price}"
                )
                formatted_price = max_price

            # Ensure it aligns with tick_size
            steps = round((formatted_price - min_price) / tick_size)
            formatted_price = steps * tick_size + min_price
            formatted_price = round(formatted_price, precision)

            logger.debug(
                f"💲 Price formatting: {price} -> {formatted_price} (min={min_price}, max={max_price}, tick={tick_size})"
            )

            return formatted_price

        except Exception as e:
            logger.error(f"Error formatting price for {symbol}: {e}")
            return price

    async def initialize(self):
        """Initialize connection and validate API keys."""
        try:
            self._log_network_status()
            self._synchronize_time()
            await self._authenticate_with_retries()
            return True
        except Exception as error:
            self._handle_initialization_error(error)
            return False

    def _log_network_status(self) -> None:
        network_label = settings.get_environment_name()
        logger.info(f"🌐 Binance Network: {network_label}")

    def _synchronize_time(self) -> None:
        """Synchronize local time with Binance server time to prevent timestamp errors."""
        try:
            # Get server time
            server_time_response = self.client.get_server_time()
            server_time = server_time_response["serverTime"]

            # Get local time in milliseconds
            local_time = int(time.time() * 1000)

            # Calculate offset (server_time - local_time)
            self.timestamp_offset = server_time - local_time

            # Apply offset to client
            # Subtract a small buffer (1000ms) to ensure we're always slightly behind server time
            self.client.timestamp_offset = self.timestamp_offset - 1000

            # Update last sync time
            self.last_time_sync = time.time()

            logger.info(
                f"⏱️  Time synchronized - Offset: {self.timestamp_offset}ms (applied: {self.client.timestamp_offset}ms)"
            )
            logger.debug(f"🕐 Server time: {server_time}, Local time: {local_time}")
        except Exception as sync_error:
            logger.warning(f"⚠️  Time sync failed: {sync_error}")
            # Set a default conservative offset if sync fails
            self.client.timestamp_offset = -2000  # 2 seconds behind
            logger.info("⏱️  Using default offset: -%s ms", self.client.timestamp_offset)

    def _check_and_resync_time(self) -> None:
        """
        Check if time sync is stale and re-synchronize if needed.
        Called before critical operations to prevent timestamp errors.
        """
        current_time = time.time()
        time_since_last_sync = current_time - self.last_time_sync
        
        # Re-sync every 30 minutes (1800 seconds) to prevent clock drift
        if time_since_last_sync > 1800:
            logger.info(f"🔄 Time sync is stale ({time_since_last_sync:.0f}s old). Re-synchronizing...")
            self._synchronize_time()

    async def _authenticate_with_retries(self, max_retries: int = 3) -> None:
        for attempt in range(1, max_retries + 1):
            try:
                _ = self.client.get_account()
                self.is_connected = True
                logger.info(
                    f"✅ Authenticated successfully (attempt {attempt}/{max_retries})"
                )
                return
            except Exception as auth_error:
                logger.warning(
                    f"⚠️  Authentication attempt {attempt}/{max_retries} failed: {auth_error}"
                )
                if attempt == max_retries:
                    raise
                await asyncio.sleep(2)

    def _handle_initialization_error(self, error: Exception) -> None:
        logger.error(f"❌ Binance initialization failed: {error}")
        logger.error("❌ Check your API keys and network connection")
        self.is_connected = False

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

    def get_all_open_orders(self) -> List[Dict]:
        """
        Get all currently open orders from Binance
        
        Returns list of open orders for position synchronization
        """
        try:
            # Re-sync time if stale to prevent timestamp errors
            self._check_and_resync_time()
            
            open_orders = self.client.get_open_orders()
            logger.debug(f"📊 Retrieved {len(open_orders)} open orders from Binance")
            return open_orders
            
        except Exception as e:
            logger.error(f"❌ Error getting open orders: {e}")
            
            # If timestamp error, force immediate re-sync and retry once
            if "recvWindow" in str(e) or "Timestamp" in str(e):
                logger.warning("⚠️  Timestamp error detected. Force re-syncing time and retrying...")
                self._synchronize_time()
                try:
                    open_orders = self.client.get_open_orders()
                    logger.info(f"✅ Retrieved {len(open_orders)} open orders on retry")
                    return open_orders
                except Exception as retry_error:
                    logger.error(f"❌ Retry failed: {retry_error}")
                    
            return []

    def get_account_positions(self) -> List[Dict]:
        """
        Get all account positions with non-zero balances
        
        Returns list of positions for balance-based position tracking
        """
        try:
            account = self.client.get_account()
            positions = []
            
            for balance in account.get("balances", []):
                asset = balance["asset"]
                free = float(balance["free"])
                locked = float(balance["locked"])
                total = free + locked
                
                # Only include positions with actual holdings
                if total > 0 and asset != "USDT":  # Exclude USDT as it's base currency
                    positions.append({
                        "asset": asset,
                        "free": free,
                        "locked": locked,
                        "total": total
                    })
                    
            logger.debug(f"📊 Retrieved {len(positions)} non-zero positions from Binance")
            return positions
            
        except Exception as e:
            logger.error(f"❌ Error getting account positions: {e}")
            return []

    def get_order_history(self, symbol: str = None, limit: int = 500) -> List[Dict]:
        """
        Get recent order history from Binance for position reconciliation
        
        Args:
            symbol: Specific symbol to get orders for (optional)
            limit: Number of recent orders to retrieve
        """
        try:
            # Re-sync time if stale to prevent timestamp errors
            self._check_and_resync_time()
            
            if symbol:
                orders = self.client.get_all_orders(symbol=symbol, limit=limit)
                logger.debug(f"📊 Retrieved {len(orders)} orders for {symbol} from Binance")
            else:
                # Get all recent orders across all symbols
                # Note: This might be rate-limited, use carefully
                orders = []
                open_orders = self.get_all_open_orders()
                
                # Extract symbols from open orders to get recent history
                symbols = set(order["symbol"] for order in open_orders)
                
                for sym in symbols:
                    try:
                        sym_orders = self.client.get_all_orders(symbol=sym, limit=100)
                        orders.extend(sym_orders)
                    except Exception as sym_error:
                        logger.warning(f"⚠️  Could not get orders for {sym}: {sym_error}")
                        
                logger.debug(f"📊 Retrieved {len(orders)} total orders from Binance")
                
            return orders
            
        except Exception as e:
            logger.error(f"❌ Error getting order history: {e}")
            
            # If timestamp error, force immediate re-sync and retry once
            if "recvWindow" in str(e) or "Timestamp" in str(e):
                logger.warning("⚠️  Timestamp error detected. Force re-syncing time and retrying...")
                self._synchronize_time()
                try:
                    if symbol:
                        orders = self.client.get_all_orders(symbol=symbol, limit=limit)
                    else:
                        orders = []
                        open_orders = self.get_all_open_orders()
                        symbols = set(order["symbol"] for order in open_orders)
                        for sym in symbols:
                            try:
                                sym_orders = self.client.get_all_orders(symbol=sym, limit=100)
                                orders.extend(sym_orders)
                            except Exception:
                                continue
                    logger.info(f"✅ Retrieved {len(orders)} orders on retry")
                    return orders
                except Exception as retry_error:
                    logger.error(f"❌ Order history retry failed: {retry_error}")
                    
            return []

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
            # Use limit parameter instead of time-based query
            # This is more reliable and works for all intervals
            klines = self.client.get_klines(
                symbol=symbol, interval=interval, limit=limit
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

    def place_market_order(self, symbol: str, side: str, quantity: float) -> Dict:
        """
        Place a market order on Binance.
        Trading mode determines which API endpoint is used.
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
        Place market entry order with OCO (One-Cancels-Other) exit on Binance.
        Trading mode determines which API endpoint is used.

        This creates:
        1. Market order for entry (BUY/SELL)
        2. OCO order for exit with Take Profit and Stop Loss

        OCO automatically executes on Binance servers:
        - If TP hits → position closes with profit, SL cancels
        - If SL hits → position closes with loss, TP cancels
        """
        try:
            # Re-sync time if stale to prevent timestamp errors during order placement
            self._check_and_resync_time()
            
            formatted_qty = self.format_quantity(symbol, quantity)
            entry_order = self._place_market_entry(symbol, side, formatted_qty)

            exit_side = self._opposite_side(side)
            executed_qty = self._extract_executed_quantity(entry_order, formatted_qty)

            # Format prices to meet PRICE_FILTER requirements
            formatted_tp = self.format_price(symbol, take_profit_price)
            formatted_sl = self.format_price(symbol, stop_loss_price)

            oco_payload = self._build_oco_payload(
                symbol,
                exit_side,
                executed_qty,
                formatted_tp,
                formatted_sl,
            )

            oco_order = self.client.create_oco_order(
                symbol=symbol, side=exit_side, **oco_payload
            )
            logger.info(f"✅ OCO order created: {oco_order['orderListId']}")

            return {
                "entry_order": entry_order,
                "oco_order": oco_order,
                "take_profit_price": formatted_tp,
                "stop_loss_price": formatted_sl,
            }

        except Exception as error:
            logger.error(f"Error placing OCO order for {symbol}: {error}")
            return {}

    def _place_market_entry(self, symbol: str, side: str, quantity: float) -> Dict:
        entry_order = self.client.order_market(
            symbol=symbol,
            side=side.upper(),
            quantity=quantity,
        )
        logger.info(f"✅ Entry order filled: {entry_order['orderId']}")
        return entry_order

    @staticmethod
    def _opposite_side(side: str) -> str:
        return "SELL" if side.upper() == "BUY" else "BUY"

    @staticmethod
    def _extract_executed_quantity(
        entry_order: Dict, fallback_quantity: float
    ) -> float:
        try:
            return float(entry_order.get("executedQty", fallback_quantity))
        except (TypeError, ValueError):
            return fallback_quantity

    def _build_oco_payload(
        self,
        symbol: str,
        exit_side: str,
        quantity: float,
        take_profit_price: float,
        stop_loss_price: float,
    ) -> Dict:
        # Format stop limit prices to respect PRICE_FILTER
        if exit_side == "SELL":
            stop_limit_price = self.format_price(
                symbol, stop_loss_price * OCO_SELL_STOP_LIMIT_BUFFER
            )
            payload = {
                "quantity": quantity,
                "aboveType": "LIMIT_MAKER",
                "abovePrice": self._to_price_str(take_profit_price),
                "belowType": "STOP_LOSS_LIMIT",
                "belowPrice": self._to_price_str(stop_limit_price),
                "belowStopPrice": self._to_price_str(stop_loss_price),
                "belowTimeInForce": TIME_IN_FORCE_GTC,
            }
        else:
            stop_limit_price = self.format_price(
                symbol, stop_loss_price * OCO_BUY_STOP_LIMIT_BUFFER
            )
            payload = {
                "quantity": quantity,
                "aboveType": "STOP_LOSS_LIMIT",
                "abovePrice": self._to_price_str(stop_limit_price),
                "aboveStopPrice": self._to_price_str(stop_loss_price),
                "aboveTimeInForce": TIME_IN_FORCE_GTC,
                "belowType": "LIMIT_MAKER",
                "belowPrice": self._to_price_str(take_profit_price),
            }
        return self._filter_none(payload)

    @staticmethod
    def _to_price_str(price: float) -> str:
        return f"{price:.8f}"

    @staticmethod
    def _filter_none(payload: Dict) -> Dict:
        return {key: value for key, value in payload.items() if value is not None}

    def get_oco_order_status(self, order_list_id: str) -> Optional[Dict]:
        """
        Check the status of an OCO order on Binance

        Returns:
        - Status of the OCO order (EXECUTING, ALL_DONE, REJECTED)
        - Which leg executed (TP or SL)
        - Execution details
        """
        try:
            # Re-sync time if stale to prevent timestamp errors
            self._check_and_resync_time()
            
            result = self.client.v3_get_order_list(orderListId=int(order_list_id))
            return result

        except Exception as e:
            logger.error(f"Error getting OCO status for {order_list_id}: {e}")
            
            # If timestamp error, force immediate re-sync and retry once
            if "recvWindow" in str(e) or "Timestamp" in str(e):
                logger.warning("⚠️  Timestamp error detected. Force re-syncing time and retrying...")
                self._synchronize_time()
                try:
                    result = self.client.v3_get_order_list(orderListId=int(order_list_id))
                    logger.info("✅ Retry successful after time re-sync")
                    return result
                except Exception as retry_error:
                    logger.error(f"❌ Retry failed: {retry_error}")
            
            return None

    def get_order_status(self, symbol: str, order_id: str) -> Optional[Dict]:
        """Get the status of an individual order by order ID"""
        try:
            # Re-sync time if stale to prevent timestamp errors
            self._check_and_resync_time()
            
            result = self.client.get_order(symbol=symbol, orderId=int(order_id))
            logger.info(f"✅ Order Status Retrieved: {order_id}")
            return result

        except Exception as e:
            logger.error(f"Error getting order status {order_id}: {e}")
            
            # If timestamp error, force immediate re-sync and retry once
            if "recvWindow" in str(e) or "Timestamp" in str(e):
                logger.warning("⚠️  Timestamp error detected. Force re-syncing time and retrying...")
                self._synchronize_time()
                try:
                    result = self.client.get_order(symbol=symbol, orderId=int(order_id))
                    logger.info(f"✅ Order Status Retrieved on retry: {order_id}")
                    return result
                except Exception as retry_error:
                    logger.error(f"❌ Order status retrieval failed after retry: {retry_error}")
                    return None
            return None

    def cancel_oco_order(self, symbol: str, order_list_id: str) -> bool:
        """Cancel an active OCO order on Binance"""
        try:
            # Re-sync time if stale to prevent timestamp errors
            self._check_and_resync_time()
            
            result = self.client._delete(
                "orderList",
                True,
                data={"symbol": symbol, "orderListId": int(order_list_id)},
            )
            logger.info(f"✅ OCO order cancelled: {order_list_id} - {result}")
            return True

        except Exception as e:
            logger.error(f"Error cancelling OCO order {order_list_id}: {e}")
            
            # If timestamp error, force immediate re-sync and retry once
            if "recvWindow" in str(e) or "Timestamp" in str(e):
                logger.warning("⚠️  Timestamp error detected. Force re-syncing time and retrying...")
                self._synchronize_time()
                try:
                    result = self.client._delete(
                        "orderList",
                        True,
                        data={"symbol": symbol, "orderListId": int(order_list_id)},
                    )
                    logger.info(f"✅ OCO order cancelled on retry: {order_list_id}")
                    return True
                except Exception as retry_error:
                    logger.error(f"❌ Retry failed: {retry_error}")
            
            return False
