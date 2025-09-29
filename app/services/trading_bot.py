import asyncio
import logging
from datetime import datetime, time
from typing import Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from ..database import get_db
from ..models.trade import TradingConfig, Trade
from ..models.user import User
from .binance_service import BinanceService
from .ai_analyzer import AIAnalyzer
from .risk_manager import RiskManager
from .websocket_manager import WebSocketManager
import json

logger = logging.getLogger(__name__)


class TradingBot:
    def __init__(self, ws_manager: WebSocketManager):
        self.binance = BinanceService()
        self.ai_analyzer = AIAnalyzer()
        self.risk_manager = RiskManager()
        self.ws_manager = ws_manager
        self.is_running = False
        self.active_users = {}
        self.last_analysis_time = {}

    async def initialize(self):
        """Initialize all services"""
        try:
            binance_connected = await self.binance.initialize()
            if not binance_connected:
                logger.error("Failed to initialize Binance service")
                return False

            logger.info("🤖 Trading Bot initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Bot initialization failed: {e}")
            return False

    async def start(self):
        """Start the trading bot"""
        if self.is_running:
            return True

        initialized = await self.initialize()
        if not initialized:
            return False

        self.is_running = True

        # Start main trading loop
        asyncio.create_task(self.trading_loop())

        # Start daily reset task
        asyncio.create_task(self.daily_reset_task())

        logger.info("🚀 Trading Bot started")
        return True

    async def stop(self):
        """Stop the trading bot"""
        self.is_running = False
        self.active_users.clear()
        logger.info("🛑 Trading Bot stopped")

    async def trading_loop(self):
        """Main trading loop"""
        while self.is_running:
            try:
                # Check if we're in trading hours
                current_time = datetime.now().time()
                if not self._is_trading_hours(current_time):
                    await asyncio.sleep(60)  # Check again in 1 minute
                    continue

                # Load active trading configurations
                async for db in get_db():
                    active_configs = await self._get_active_trading_configs(db)

                    for config in active_configs:
                        try:
                            await self._process_user_trading(db, config)
                        except Exception as e:
                            logger.error(f"Error processing user {config.user_id}: {e}")

                    break  # Exit the db session loop

                # Wait before next analysis cycle (1 minute for scalping)
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(10)  # Short delay before retry

    async def _process_user_trading(self, db: AsyncSession, config: TradingConfig):
        """Process trading for a specific user"""
        try:
            user_id = config.user_id
            symbol = config.trading_pair

            # Rate limiting: Don't analyze too frequently per user
            last_analysis = self.last_analysis_time.get(user_id, 0)
            if (
                asyncio.get_event_loop().time() - last_analysis < 30
            ):  # 30 seconds minimum
                return

            # Get market data
            kline_data = self.binance.get_kline_data(symbol, "1m", 100)
            if kline_data.empty:
                logger.warning(f"No market data for {symbol}")
                return

            # Get AI analysis
            ai_signal = await self.ai_analyzer.analyze_market_data(symbol, kline_data)
            self.last_analysis_time[user_id] = asyncio.get_event_loop().time()

            # Send analysis update via WebSocket
            await self.ws_manager.send_to_user(
                user_id,
                {
                    "type": "ai_analysis",
                    "symbol": symbol,
                    "signal": ai_signal["signal"],
                    "confidence": ai_signal["confidence"],
                    "reasoning": ai_signal["reasoning"],
                    "timestamp": datetime.now().isoformat(),
                },
            )

            # Only proceed if we have a buy/sell signal
            if ai_signal["signal"] == "hold":
                return

            # Get account balance
            balances = self.binance.get_account_balance()
            usdt_balance = balances.get("USDT", {}).get("free", 0.0)

            # Validate trade with risk management
            is_valid, message, trade_params = self.risk_manager.validate_trade_signal(
                user_id, ai_signal, usdt_balance, config.__dict__
            )

            if not is_valid:
                logger.info(f"Trade rejected for user {user_id}: {message}")
                await self.ws_manager.send_to_user(
                    user_id,
                    {
                        "type": "trade_rejected",
                        "reason": message,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
                return

            # Execute trade
            await self._execute_trade(db, config, ai_signal, trade_params)

        except Exception as e:
            logger.error(f"Error in user trading process: {e}")

    async def _execute_trade(
        self, db: AsyncSession, config: TradingConfig, signal: Dict, params: Dict
    ):
        """Execute a trade based on the signal"""
        try:
            user_id = config.user_id
            symbol = config.trading_pair
            side = signal["signal"].upper()  # BUY or SELL

            # Place order on Binance
            order_result = self.binance.place_market_order(
                symbol=symbol,
                side=side,
                quantity=params["position_size"],
                test_mode=config.is_test_mode,
            )

            if not order_result:
                logger.error(f"Failed to place order for user {user_id}")
                return

            # Record trade in database
            trade = Trade(
                user_id=user_id,
                trade_id=order_result["orderId"],
                symbol=symbol,
                side=side.lower(),
                amount=float(order_result["executedQty"]),
                price=float(order_result["fills"][0]["price"]),
                total_value=params["trade_value"],
                fee=float(order_result["fills"][0]["commission"]),
                status="filled",
                is_test_trade=config.is_test_mode,
                strategy_used="scalping_ai",
                ai_signal_confidence=signal["final_confidence"],
            )

            db.add(trade)
            await db.commit()

            # Update risk manager
            self.risk_manager.record_trade(user_id)

            # Send trade notification
            await self.ws_manager.send_to_user(
                user_id,
                {
                    "type": "trade_executed",
                    "trade": {
                        "symbol": symbol,
                        "side": side.lower(),
                        "amount": trade.amount,
                        "price": trade.price,
                        "total_value": trade.total_value,
                        "confidence": signal["final_confidence"],
                        "test_mode": config.is_test_mode,
                    },
                    "timestamp": datetime.now().isoformat(),
                },
            )

            logger.info(
                f"✅ Trade executed for user {user_id}: {side} {params['position_size']} {symbol} @ {params['entry_price']}"
            )

        except Exception as e:
            logger.error(f"Trade execution error: {e}")

    def _is_trading_hours(self, current_time: time) -> bool:
        """Check if current time is within trading hours"""
        # Trading hours: 08:00 - 16:00 GMT (Ghana time)
        start_hour = 8
        end_hour = 16

        return start_hour <= current_time.hour < end_hour

    async def _get_active_trading_configs(self, db: AsyncSession):
        """Get all active trading configurations"""
        from sqlalchemy import select

        query = select(TradingConfig).where(TradingConfig.is_active == True)
        result = await db.execute(query)
        return result.scalars().all()

    async def daily_reset_task(self):
        """Reset daily counters at midnight"""
        while self.is_running:
            now = datetime.now()
            # Calculate seconds until midnight
            tomorrow = now.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + asyncio.timedelta(days=1)
            seconds_until_midnight = (tomorrow - now).total_seconds()

            await asyncio.sleep(seconds_until_midnight)

            if self.is_running:
                self.risk_manager.reset_daily_counters()
                logger.info("🔄 Daily counters reset")
