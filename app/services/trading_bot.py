import asyncio
import logging
from contextlib import suppress
from datetime import datetime, time, timedelta, timezone
from typing import Dict, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..database import get_db, get_async_session
from ..models.trade import TradingConfig, Trade, OpenPosition
from ..models.user import User
from ..models.portfolio import Portfolio
from .binance_service import BinanceService
from .mock_binance_service import MockBinanceService
from .ai_analyzer import AIAnalyzer
from .risk_manager import RiskManager
from .websocket_manager import WebSocketManager
from .market_regime_analyzer import MarketRegimeAnalyzer
from .circuit_breaker import CircuitBreaker, CircuitBreakerTriggered
from ..logging_config import get_trading_metrics_logger
from ..config import settings
from .service_constants import (
    ANALYSIS_RATE_LIMIT_SECONDS,
    POSITION_TIMEOUT_MINUTES,
    TRADING_CYCLE_DELAY_SECONDS,
)
import json

logger = logging.getLogger(__name__)
metrics_logger = get_trading_metrics_logger()


class TradingBot:
    def __init__(self, ws_manager: WebSocketManager):
        self.binance = self._create_binance_service()
        self.ai_analyzer = AIAnalyzer()
        self.risk_manager = RiskManager()
        self.regime_analyzer = MarketRegimeAnalyzer(
            filter_mode=settings.regime_filter_mode
        )
        self.circuit_breaker = CircuitBreaker()
        self.ws_manager = ws_manager
        self.is_running = False
        self.active_users = {}
        self.last_analysis_time = {}
        
        # Consecutive loss tracking for emergency stop
        self.consecutive_losses = {}  # user_id -> count
        self.loss_streak_cooldown = {}  # user_id -> cooldown_until_timestamp

    def _create_binance_service(self):
        if settings.use_mock_binance:
            logger.info("🧪 Using MOCK Binance service - All trades are simulated!")
            return MockBinanceService()

        logger.info("🔧 Using REAL Binance service")
        return BinanceService()

    async def _initialize_binance_service(self) -> bool:
        logger.info("📡 Connecting to Binance API...")
        binance_connected = await self.binance.initialize()
        if not binance_connected:
            logger.error(
                "❌ Failed to initialize Binance service - Trading Bot will not start!"
            )
            return False
        logger.info("✅ Binance service initialized successfully")
        return True

    def _initialize_support_services(self) -> None:
        logger.info("🧠 Initializing AI Analyzer...")
        logger.info("✅ AI Analyzer ready")
        logger.info("🛡️  Initializing Risk Manager...")
        logger.info("✅ Risk Manager ready")
        logger.info("🛡️  Circuit Breaker System ready")
        logger.info("🌐 WebSocket Manager ready")

    def _log_initialization_success(self) -> None:
        logger.info("🤖 Trading Bot initialized successfully")
        metrics_logger.info("BOT_INITIALIZED | STATUS=SUCCESS")

    def _log_trading_window_status(self) -> None:
        logger.info(
            f"⏰ Trading Hours: {settings.trading_active_hours_start} - {settings.trading_active_hours_end}"
        )
        current_time = datetime.now(timezone.utc).time()
        status = "✅ Yes" if self._is_trading_hours(current_time) else "❌ No"
        logger.info(
            f"🕐 Current time: {current_time.strftime('%H:%M:%S')} | Trading active: {status}"
        )

    def _start_background_tasks(self) -> None:
        self.trading_task = asyncio.create_task(self.trading_loop())
        logger.info("🔄 Trading loop task created and started!")

        self.daily_reset_task_handle = asyncio.create_task(self.daily_reset_task())
        logger.info("⏰ Daily reset task created and started!")

    async def initialize(self):
        """Initialize all services"""
        try:
            logger.info("🔧 Initializing Trading Bot services...")

            if not await self._initialize_binance_service():
                return False

            self._initialize_support_services()
            await self._load_open_positions_from_db()
            self._log_initialization_success()
            return True

        except Exception as e:
            logger.error(f"❌ Bot initialization failed: {e}")
            logger.error("❌ This means the trading bot will NOT run!")
            metrics_logger.info(f"BOT_INITIALIZATION_FAILED | ERROR={str(e)}")
            return False

    async def start(self):
        """Start the trading bot"""
        if self.is_running:
            return True

        initialized = await self.initialize()
        if not initialized:
            return False

        self._log_trading_window_status()

        self.is_running = True

        self._start_background_tasks()

        logger.info("🚀 Trading Bot started and running!")
        logger.info(
            f"⚡ You should see trading cycle logs every {TRADING_CYCLE_DELAY_SECONDS} seconds"
        )
        metrics_logger.info("BOT_STARTED | STATUS=RUNNING")
        return True

    async def stop(self):
        """Stop the trading bot"""
        logger.info("🛑 Stopping Trading Bot...")
        self.is_running = False

        # Cancel running tasks
        if hasattr(self, "trading_task") and not self.trading_task.done():
            self.trading_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.trading_task
            logger.debug("Trading task cancelled")

        if (
            hasattr(self, "daily_reset_task_handle")
            and not self.daily_reset_task_handle.done()
        ):
            self.daily_reset_task_handle.cancel()
            with suppress(asyncio.CancelledError):
                await self.daily_reset_task_handle
            logger.debug("Daily reset task cancelled")

        self.active_users.clear()
        logger.info("🛑 Trading Bot stopped")
        metrics_logger.info("BOT_STOPPED | STATUS=STOPPED")

    async def trading_loop(self):
        """Main trading loop"""
        cycle_count = 0
        while self.is_running:
            try:
                cycle_count += 1
                await self._run_trading_cycle(cycle_count)
            except Exception as e:
                logger.error(f"❌ Trading loop error in cycle #{cycle_count}: {e}")
                await asyncio.sleep(10)  # Short delay before retry

    async def _run_trading_cycle(self, cycle_count: int) -> None:
        cycle_start_time = datetime.now(timezone.utc)
        logger.info(
            f"🔄 Starting trading cycle #{cycle_count} at {cycle_start_time.strftime('%H:%M:%S')}"
        )

        current_time = datetime.now(timezone.utc).time()
        if not self._is_trading_hours(current_time):
            logger.info(
                f"⏰ Outside trading hours ({current_time.strftime('%H:%M:%S')}). Next check in {TRADING_CYCLE_DELAY_SECONDS} seconds."
            )
            await asyncio.sleep(TRADING_CYCLE_DELAY_SECONDS)
            return

        async for db in get_db():
            active_configs = await self._get_active_trading_configs(db)
            self._log_cycle_stats(cycle_count, active_configs)

            processed_users = await self._process_active_configs(db, active_configs)

            cycle_duration = (
                datetime.now(timezone.utc) - cycle_start_time
            ).total_seconds()
            self._log_cycle_summary(cycle_count, cycle_duration, processed_users)
            break

        logger.debug(
            f"⏳ Waiting {TRADING_CYCLE_DELAY_SECONDS} seconds before next cycle..."
        )
        await asyncio.sleep(TRADING_CYCLE_DELAY_SECONDS)

    def _log_cycle_stats(self, cycle_count: int, active_configs) -> None:
        total_active_users = len(active_configs)
        connected_users = self.ws_manager.get_connected_users()
        total_connected = len(connected_users)
        connection_count = self.ws_manager.get_connection_count()

        logger.info(f"👥 TRADING CYCLE #{cycle_count} STATS:")
        logger.info(f"   📊 Active trading users: {total_active_users}")
        logger.info(f"   🌐 Connected users: {total_connected}")
        logger.info(f"   🔌 Total connections: {connection_count}")

        metrics_logger.info(
            f"CYCLE_{cycle_count} | ACTIVE_USERS={total_active_users} | CONNECTED_USERS={total_connected} | CONNECTIONS={connection_count}"
        )

        if total_active_users == 0:
            logger.info("⚠️  No active trading users found")
        else:
            logger.info(
                f"🎯 Processing {total_active_users} active trading configurations..."
            )

    async def _process_active_configs(self, db: AsyncSession, active_configs) -> int:
        total_active_users = len(active_configs)
        if total_active_users == 0:
            return 0

        await self._check_oco_orders(db)

        processed_users = 0
        for config in active_configs:
            processed_users += 1
            await self._process_single_config(
                db, config, processed_users, total_active_users
            )

        return processed_users

    async def _process_single_config(
        self,
        db: AsyncSession,
        config: TradingConfig,
        processed_users: int,
        total_users: int,
    ) -> None:
        try:
            logger.debug(
                f"📈 Processing user {config.user_id} ({processed_users}/{total_users}) - {config.trading_pair}"
            )
            await self._check_position_exits(db, config)
            await self._process_user_trading(db, config)
        except Exception as exc:
            logger.error(f"❌ Error processing user {config.user_id}: {exc}")

    async def _process_user_trading(self, db: AsyncSession, config: TradingConfig):
        """Process trading for a specific user"""
        try:
            user_id = config.user_id
            symbol = config.trading_pair

            # CRITICAL: Check consecutive loss cooldown FIRST
            if user_id in self.loss_streak_cooldown:
                cooldown_until = self.loss_streak_cooldown[user_id]
                now = datetime.now(timezone.utc)
                if now < cooldown_until:
                    remaining = (cooldown_until - now).total_seconds() / 60
                    logger.warning(
                        f"🚫 [User {user_id}] LOSS STREAK COOLDOWN ACTIVE: "
                        f"{remaining:.1f} minutes remaining. "
                        f"Consecutive losses: {self.consecutive_losses.get(user_id, 0)}"
                    )
                    return
                else:
                    # Cooldown expired, clear it
                    logger.info(
                        f"✅ [User {user_id}] Loss streak cooldown expired. Resuming trading."
                    )
                    del self.loss_streak_cooldown[user_id]
                    self.consecutive_losses[user_id] = 0

            # Skip analysis if a position is already open for this pair
            open_positions = self.risk_manager.get_open_positions(user_id)
            if any(p.get("symbol") == symbol for p in open_positions):
                logger.info(
                    f"✅ [User {user_id}] Position already open for {symbol}. Skipping new trade analysis."
                )
                return

            logger.debug(f"🔍 [User {user_id}] Starting trading analysis for {symbol}")

            remaining_cooldown = self._remaining_rate_limit(user_id)
            if remaining_cooldown > 0:
                logger.debug(
                    f"⏱️  [User {user_id}] Rate limit active: {remaining_cooldown:.1f}s remaining"
                )
                return

            logger.info(
                f"📊 [User {user_id}] ANALYZING {symbol} (Network: {'Testnet' if settings.binance_testnet else 'Mainnet'})"
            )

            market_snapshot = self._fetch_market_snapshot(symbol, user_id)
            if market_snapshot is None:
                return

            # Unpack multi-timeframe data
            primary_data, context_data, precision_data, current_price = market_snapshot

            # Calculate technical indicators for all timeframes
            logger.debug(
                f"📊 [User {user_id}] Calculating multi-timeframe technical indicators..."
            )
            primary_data = self.ai_analyzer.calculate_technical_indicators(primary_data)
            context_data = self.ai_analyzer.calculate_technical_indicators(context_data)
            precision_data = self.ai_analyzer.calculate_technical_indicators(
                precision_data
            )

            # Verify indicators were calculated
            if primary_data.empty:
                logger.error(
                    f"❌ [User {user_id}] Primary data is empty after indicator calculation"
                )
                return

            required_indicators = ["atr", "adx", "close"]
            missing = [
                ind for ind in required_indicators if ind not in primary_data.columns
            ]
            if missing:
                logger.error(
                    f"❌ [User {user_id}] Missing indicators: {missing}. Available: {list(primary_data.columns)}"
                )
                return

            logger.debug(
                f"✅ [User {user_id}] Indicators calculated: "
                f"ATR={primary_data['atr'].iloc[-1]:.2f}, "
                f"ADX={primary_data['adx'].iloc[-1]:.2f}, "
                f"Price={primary_data['close'].iloc[-1]:.2f}"
            )

            # Run market regime analysis on primary timeframe (15m)
            logger.debug(f"🔍 [User {user_id}] Analyzing market regime...")
            regime_analysis = await self.regime_analyzer.classify_market_regime(
                primary_data, symbol
            )

            trading_quality = regime_analysis.get("trading_quality_score", 0)
            if trading_quality < 50:
                logger.info(
                    f"🚫 [User {user_id}] Low trading quality: {trading_quality}/100 - Regime: {regime_analysis.get('regime')} | "
                    f"ADX: {regime_analysis.get('trend_strength', 0):.1f} | "
                    f"ATR%: {regime_analysis.get('atr_percentage', 0):.2f}%"
                )
                return

            # Day trading: Focus on current market conditions, not historical performance
            # Historical data can create bias and hesitation in fast-moving markets
            user_trade_history = None  # Disabled for day trading

            # AI analysis with multi-timeframe context
            ai_signal = await self._request_ai_signal_mtf(
                symbol,
                primary_data,
                context_data,
                precision_data,
                user_id,
                user_trade_history,
                regime_analysis,
            )
            if ai_signal is None:
                return

            self._mark_analysis_timestamp(user_id)

            signal_details = self._resolve_signal_details(ai_signal)
            self._log_ai_analysis(user_id, symbol, signal_details, current_price)
            await self._emit_ai_analysis(user_id, symbol, signal_details)

            if signal_details["signal"] == "hold":
                logger.debug(f"⏸️  [User {user_id}] HOLD signal - no action taken")
                return

            logger.info(
                f"🎯 [User {user_id}] TRADE SIGNAL DETECTED: {signal_details['signal'].upper()}"
            )

            usdt_balance = self._fetch_usdt_balance(user_id)

            # 🛡️ Circuit Breaker Check - BEFORE trade validation
            try:
                self.circuit_breaker.check_before_trade(user_id, usdt_balance)
            except CircuitBreakerTriggered as cb_error:
                logger.critical(
                    f"🛑 [User {user_id}] CIRCUIT BREAKER TRIGGERED: {cb_error.reason}"
                )
                metrics_logger.info(
                    f"CIRCUIT_BREAKER_HALT | USER={user_id} | REASON={cb_error.reason}"
                )

                # Get detailed status for user notification
                cb_status = self.circuit_breaker.get_user_status(user_id, usdt_balance)
                await self.ws_manager.send_to_user(
                    user_id,
                    {
                        "type": "circuit_breaker_triggered",
                        "reason": cb_error.reason,
                        "status": cb_status,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
                return

            is_valid, message, trade_params = self.risk_manager.validate_trade_signal(
                user_id, ai_signal, usdt_balance, config.__dict__, primary_data
            )

            if not is_valid:
                await self._handle_trade_rejection(user_id, symbol, message)
                return

            self._log_trade_validation_success(user_id, trade_params)

            await self._execute_trade(db, config, ai_signal, trade_params, regime_analysis)

        except CircuitBreakerTriggered:
            # Already handled above, don't log as generic error
            pass
        except Exception as e:
            logger.error(f"❌ [User {user_id}] Error in trading process: {e}")
            metrics_logger.info(f"TRADING_ERROR | USER={user_id} | ERROR={str(e)}")

    def _remaining_rate_limit(self, user_id: int) -> float:
        last_analysis = self.last_analysis_time.get(user_id)
        if not last_analysis:
            return 0.0
        elapsed = asyncio.get_event_loop().time() - last_analysis
        remaining = ANALYSIS_RATE_LIMIT_SECONDS - elapsed
        return remaining if remaining > 0 else 0.0

    def _fetch_market_snapshot(self, symbol: str, user_id: int):
        """
        Fetch multi-timeframe market data for comprehensive day trading analysis.

        Returns tuple of (primary_df, context_df, precision_df, current_price)
        - Primary (15m): Used for entry signals and main analysis
        - Context (1h): Provides trend direction and market structure
        - Precision (5m): Helps with entry timing
        """
        from .service_constants import (
            MTF_PRIMARY_INTERVAL,
            MTF_CONTEXT_INTERVAL,
            MTF_PRECISION_INTERVAL,
            MTF_PRIMARY_CANDLES,
            MTF_CONTEXT_CANDLES,
            MTF_PRECISION_CANDLES,
        )

        logger.debug(
            f"📈 [User {user_id}] Fetching multi-timeframe data for {symbol}..."
        )
        try:
            # Fetch all three timeframes
            primary_data = self.binance.get_kline_data(
                symbol, MTF_PRIMARY_INTERVAL, MTF_PRIMARY_CANDLES
            )
            context_data = self.binance.get_kline_data(
                symbol, MTF_CONTEXT_INTERVAL, MTF_CONTEXT_CANDLES
            )
            precision_data = self.binance.get_kline_data(
                symbol, MTF_PRECISION_INTERVAL, MTF_PRECISION_CANDLES
            )

            if primary_data.empty:
                logger.warning(
                    f"⚠️  [User {user_id}] No primary (15m) data available for {symbol}"
                )
                self.circuit_breaker.record_api_failure(
                    f"Empty primary kline data for {symbol}"
                )
                return None

            if context_data.empty:
                logger.warning(
                    f"⚠️  [User {user_id}] No context (1h) data available for {symbol}"
                )
                # Context data missing is not critical, but log it
                context_data = primary_data  # Fallback to primary

            if precision_data.empty:
                logger.warning(
                    f"⚠️  [User {user_id}] No precision (5m) data available for {symbol}"
                )
                precision_data = primary_data  # Fallback to primary

            # API calls successful
            self.circuit_breaker.record_api_success()

            current_price = float(primary_data.iloc[-1]["close"])
            logger.debug(
                f"💰 [User {user_id}] Multi-timeframe data loaded for {symbol}:\n"
                f"   Primary (15m): {len(primary_data)} candles\n"
                f"   Context (1h): {len(context_data)} candles\n"
                f"   Precision (5m): {len(precision_data)} candles\n"
                f"   Current price: ${current_price:.4f}"
            )

            return primary_data, context_data, precision_data, current_price

        except Exception as e:
            logger.error(f"❌ [User {user_id}] API error fetching market data: {e}")
            self.circuit_breaker.record_api_failure(f"get_kline_data failed: {str(e)}")
            return None

    async def _request_ai_signal(
        self,
        symbol,
        kline_data,
        user_id: int,
        user_trade_history=None,
        regime_analysis=None,
    ):
        """Legacy single-timeframe AI signal (kept for backward compatibility)."""
        logger.debug(f"🤖 [User {user_id}] Running AI analysis...")
        ai_signal = await self.ai_analyzer.analyze_market_data(
            symbol, kline_data, user_trade_history, regime_analysis
        )

        if ai_signal is None:
            logger.error(f"❌ [User {user_id}] AI analyzer returned None")
            return None

        if not isinstance(ai_signal, dict):
            logger.error(
                f"❌ [User {user_id}] AI analyzer returned invalid type: {type(ai_signal)}"
            )
            return None

        return ai_signal

    async def _request_ai_signal_mtf(
        self,
        symbol,
        primary_df,
        context_df,
        precision_df,
        user_id: int,
        user_trade_history=None,
        regime_analysis=None,
    ):
        """
        Multi-timeframe AI analysis for day trading.
        Combines 15m (primary), 1h (context), and 5m (precision) for comprehensive analysis.
        """
        logger.debug(f"🤖 [User {user_id}] Running multi-timeframe AI analysis...")

        # Package multi-timeframe data
        mtf_data = {
            "primary": primary_df,  # 15m - main trading signals
            "context": context_df,  # 1h - trend direction
            "precision": precision_df,  # 5m - entry timing
        }

        ai_signal = await self.ai_analyzer.analyze_market_data_mtf(
            symbol, mtf_data, user_trade_history, regime_analysis
        )

        if ai_signal is None:
            logger.error(f"❌ [User {user_id}] MTF AI analyzer returned None")
            return None

        if not isinstance(ai_signal, dict):
            logger.error(
                f"❌ [User {user_id}] MTF AI analyzer returned invalid type: {type(ai_signal)}"
            )
            return None

        return ai_signal

    def _mark_analysis_timestamp(self, user_id: int) -> None:
        self.last_analysis_time[user_id] = asyncio.get_event_loop().time()

    @staticmethod
    def _resolve_signal_details(ai_signal: Dict) -> Dict:
        confidence = float(
            ai_signal.get("final_confidence", ai_signal.get("confidence", 0))
        )
        return {
            "signal": ai_signal.get("signal", "hold"),
            "confidence": confidence,
            "reasoning": ai_signal.get("reasoning", "No reasoning provided"),
            "raw": ai_signal,
        }

    def _log_ai_analysis(
        self, user_id: int, symbol: str, signal_details: Dict, current_price: float
    ) -> None:
        signal = signal_details["signal"]
        confidence = signal_details["confidence"]
        reasoning = signal_details["reasoning"]
        signal_emoji = {"buy": "🟢", "sell": "🔴", "hold": "🟡"}.get(signal, "❓")

        logger.info(
            f"{signal_emoji} [User {user_id}] AI SIGNAL: {signal.upper()} | Confidence: {confidence:.1f}% | {reasoning}"
        )

        metrics_logger.info(
            f"AI_ANALYSIS | USER={user_id} | SYMBOL={symbol} | SIGNAL={signal} | CONFIDENCE={confidence:.1f} | PRICE={current_price:.4f}"
        )

    async def _emit_ai_analysis(
        self, user_id: int, symbol: str, signal_details: Dict
    ) -> None:
        await self.ws_manager.send_to_user(
            user_id,
            {
                "type": "ai_analysis",
                "symbol": symbol,
                "signal": signal_details["signal"],
                "confidence": signal_details["confidence"],
                "reasoning": signal_details["reasoning"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _fetch_usdt_balance(self, user_id: int) -> float:
        logger.debug(f"💳 [User {user_id}] Checking account balance...")
        balances = self.binance.get_account_balance()
        usdt_balance = balances.get("USDT", {}).get("free", 0.0)
        logger.info(f"💰 [User {user_id}] Available USDT balance: ${usdt_balance:.2f}")
        return usdt_balance

    async def _handle_trade_rejection(
        self, user_id: int, symbol: str, message: str
    ) -> None:
        logger.warning(f"❌ [User {user_id}] TRADE REJECTED: {message}")
        metrics_logger.info(
            f"TRADE_REJECTED | USER={user_id} | SYMBOL={symbol} | REASON={message}"
        )
        await self.ws_manager.send_to_user(
            user_id,
            {
                "type": "trade_rejected",
                "reason": message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _log_trade_validation_success(self, user_id: int, trade_params: Dict) -> None:
        logger.info(
            f"✅ [User {user_id}] Trade validation PASSED - proceeding to execution"
        )
        logger.info(
            f"📋 [User {user_id}] Trade params: Size=${trade_params.get('trade_value', 0):.2f}, Position={trade_params.get('position_size', 0):.6f}"
        )

    async def _check_position_exits(self, db: AsyncSession, config: TradingConfig):
        """Check if any open positions should be closed"""
        try:
            user_id = config.user_id
            symbol = config.trading_pair
            current_positions = self.risk_manager.get_open_positions(user_id)
            if current_positions:
                logger.debug(
                    f"🔍 [User {user_id}] Checking {len(current_positions)} open positions for exits"
                )

            current_price = self.binance.get_symbol_price(symbol)
            if not current_price or current_price <= 0.0:
                logger.debug(
                    f"❌ [User {user_id}] Invalid current price: {current_price}"
                )
                return

            logger.debug(
                f"💰 [User {user_id}] Current {symbol} price: ${current_price:.4f}"
            )

            positions_to_close = self.risk_manager.check_exit_conditions(
                user_id, current_price, symbol
            )

            if not positions_to_close:
                logger.debug(
                    f"✅ [User {user_id}] No positions ready to close at current price ${current_price:.4f}"
                )
                return

            logger.info(
                f"� [User {user_id}] Found {len(positions_to_close)} positions ready to close!"
            )

            for position_info in positions_to_close:
                await self._handle_position_exit(
                    db, config, position_info, current_price
                )

        except Exception as e:
            logger.error(f"❌ [User {user_id}] Error checking position exits: {e}")

    async def _handle_position_exit(
        self,
        db: AsyncSession,
        config: TradingConfig,
        position_info: Dict,
        current_price: float,
    ) -> None:
        user_id = config.user_id
        symbol = config.trading_pair
        position = position_info["position"]
        exit_reason = position_info["exit_reason"]

        logger.info(f"🚨 [User {user_id}] CLOSING POSITION: {exit_reason}")
        logger.info(
            f"   📊 Position: {position['side'].upper()} {position['amount']:.6f} {symbol}"
        )
        logger.info(
            f"   💰 Entry: ${position['entry_price']:.4f} → Exit: ${current_price:.4f}"
        )

        close_side = "SELL" if position["side"].upper() == "BUY" else "BUY"
        close_order_result = self.binance.place_market_order(
            symbol=symbol,
            side=close_side,
            quantity=position["amount"],
        )

        if not close_order_result:
            logger.error(
                f"❌ [User {user_id}] Failed to place close order for {symbol}"
            )
            return

        executed_qty = float(close_order_result["executedQty"])
        exit_price = float(close_order_result["fills"][0]["price"])
        exit_commission = float(close_order_result["fills"][0]["commission"])

        # Calculate P&L metrics (same as OCO/timeout paths)
        entry_price = position["entry_price"]
        entry_fees = position["fees_paid"]
        profit_loss = self._calculate_profit_loss(
            position["side"],
            entry_price,
            exit_price,
            position["amount"],
            entry_fees,
            exit_commission,
        )
        profit_loss_pct = (profit_loss / position["entry_value"]) * 100

        # Calculate duration
        entry_time = position["entry_time"]
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        duration_seconds = (datetime.now(timezone.utc) - entry_time).total_seconds()

        # Build closed position dict
        closed_position = {
            **position,
            "exit_price": exit_price,
            "exit_time": datetime.now(timezone.utc),
            "exit_fees": exit_commission,
            "total_fees": entry_fees + exit_commission,
            "gross_pnl": profit_loss + entry_fees + exit_commission,
            "net_pnl": profit_loss,
            "pnl_percentage": profit_loss_pct,
            "duration_seconds": duration_seconds,
            "exit_reason": exit_reason,
        }

        # Remove from risk manager memory
        self._remove_position_from_risk_manager(user_id, position["trade_id"])

        await self._finalize_position_close(
            db=db,
            config=config,
            position=position,
            closed_position=closed_position,
            close_side=close_side,
            exit_reason=exit_reason,
            executed_qty=executed_qty,
            exit_price=exit_price,
            exit_commission=exit_commission,
        )

    async def _finalize_position_close(
        self,
        db: AsyncSession,
        config: TradingConfig,
        position: Dict,
        closed_position: Dict,
        close_side: str,
        exit_reason: str,
        executed_qty: float,
        exit_price: float,
        exit_commission: float,
    ) -> None:
        user_id = config.user_id
        symbol = config.trading_pair

        self._log_position_close_summary(
            user_id,
            symbol,
            position,
            close_side,
            executed_qty,
            exit_price,
            closed_position,
        )

        fresh_balance = self._get_current_usdt_balance()

        profit_label = "PROFIT" if closed_position["net_pnl"] > 0 else "LOSS"
        metrics_logger.info(
            f"POSITION_CLOSED | USER={user_id} | SYMBOL={symbol} | REASON={exit_reason} | PNL=${closed_position['net_pnl']:+.2f} | PCT={closed_position['pnl_percentage']:+.2f}% | DURATION={closed_position['duration_seconds']:.0f}s | RESULT={profit_label}"
        )

        # 🛡️ Record trade result in Circuit Breaker
        is_winner = closed_position["net_pnl"] > 0
        self.circuit_breaker.record_trade_result(
            user_id=user_id, pnl=closed_position["net_pnl"], is_winner=is_winner
        )
        
        # Track consecutive losses and trigger emergency cooldown
        if is_winner:
            # Reset loss streak on winning trade
            if user_id in self.consecutive_losses:
                logger.info(f"✅ [User {user_id}] Winning trade! Loss streak reset.")
                self.consecutive_losses[user_id] = 0
        else:
            # Increment loss counter
            self.consecutive_losses[user_id] = self.consecutive_losses.get(user_id, 0) + 1
            losses = self.consecutive_losses[user_id]
            
            logger.warning(
                f"❌ [User {user_id}] Consecutive losses: {losses}"
            )
            
            # Trigger cooldown based on loss streak
            if losses >= 3:
                # 3+ losses: 4 hour cooldown (very serious)
                cooldown_hours = 4
                cooldown_until = datetime.now(timezone.utc) + timedelta(hours=cooldown_hours)
                self.loss_streak_cooldown[user_id] = cooldown_until
                logger.error(
                    f"🚨🚨 [User {user_id}] EMERGENCY STOP: {losses} consecutive losses! "
                    f"Trading suspended for {cooldown_hours} hours until {cooldown_until.strftime('%H:%M:%S UTC')}"
                )
            elif losses >= 2:
                # 2 losses: 1 hour cooldown
                cooldown_hours = 1
                cooldown_until = datetime.now(timezone.utc) + timedelta(hours=cooldown_hours)
                self.loss_streak_cooldown[user_id] = cooldown_until
                logger.warning(
                    f"⚠️ [User {user_id}] COOLDOWN: {losses} consecutive losses. "
                    f"Trading paused for {cooldown_hours} hour until {cooldown_until.strftime('%H:%M:%S UTC')}"
                )

        #  Update original trade record with exit details (consistent with OCO/timeout exits)
        await self._update_trade_record_after_exit(
            db,
            trade_id=position["trade_id"],
            exit_price=exit_price,
            exit_fee=exit_commission,
            exit_reason=exit_reason,
            profit_loss=closed_position["net_pnl"],
            profit_loss_pct=closed_position["pnl_percentage"],
            duration_seconds=closed_position["duration_seconds"],
        )

        await self._remove_open_position_from_db(db, position["trade_id"])

        await self._notify_position_closed(
            user_id, closed_position, exit_reason, fresh_balance
        )

    def _log_position_close_summary(
        self,
        user_id: int,
        symbol: str,
        position: Dict,
        close_side: str,
        executed_qty: float,
        exit_price: float,
        closed_position: Dict,
    ) -> None:
        duration = closed_position["duration_seconds"]
        duration_str = (
            f"{int(duration // 60)}m {int(duration % 60)}s"
            if duration > 60
            else f"{duration:.1f}s"
        )

        logger.info("🎯 [User %s] POSITION CLOSED SUMMARY:", user_id)
        logger.info(
            "   📈 Trade: %s → %s",
            position["side"].upper(),
            close_side,
        )
        logger.info(
            "   💰 Entry: $%.4f → Exit: $%.4f",
            position["entry_price"],
            exit_price,
        )
        logger.info("   📊 Quantity: %.6f %s", executed_qty, symbol)
        logger.info("   ⏱️  Duration: %s", duration_str)
        logger.info("   💸 Total Fees: $%.4f", closed_position["total_fees"])
        logger.info(
            "   💵 P&L: $%+.2f (%+.2f%%)",
            closed_position["net_pnl"],
            closed_position["pnl_percentage"],
        )

    def _get_current_usdt_balance(self) -> float:
        fresh_balances = self.binance.get_account_balance()
        balance = fresh_balances.get("USDT", {}).get("free", 0.0)
        logger.info("   🏦 Binance USDT Balance: $%.2f", balance)
        return balance

    def _remove_position_from_risk_manager(self, user_id: int, trade_id: str) -> None:
        """Remove a position from risk manager's in-memory tracking"""
        risk_manager_positions = self.risk_manager.get_open_positions(user_id)
        for i, pos in enumerate(risk_manager_positions):
            if pos.get("trade_id") == trade_id:
                self.risk_manager.open_positions[user_id].pop(i)
                logger.debug(f"✅ Removed position {trade_id} from risk manager memory")
                break

    async def _remove_open_position_from_db(
        self, db: AsyncSession, trade_id: str
    ) -> None:
        position_query = select(OpenPosition).where(OpenPosition.trade_id == trade_id)
        db_position = await db.execute(position_query)
        db_position_obj = db_position.scalar_one_or_none()

        if db_position_obj:
            await db.delete(db_position_obj)
            await db.commit()
            logger.debug(f"✅ [DB] Open position removed for trade {trade_id}")

    async def _notify_position_closed(
        self,
        user_id: int,
        closed_position: Dict,
        exit_reason: str,
        actual_usdt_balance: float,
    ) -> None:
        await self.ws_manager.send_to_user(
            user_id,
            {
                "type": "position_closed",
                "position": closed_position,
                "exit_reason": exit_reason,
                "new_balance": actual_usdt_balance,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def _check_oco_orders(self, db: AsyncSession):
        """
        Check status of all active OCO orders and update database when filled
        This runs periodically to detect when Binance executes TP or SL
        """
        try:
            positions_with_oco = await self._get_positions_with_oco(db)
            if not positions_with_oco:
                logger.debug("🔍 No OCO orders to check")
                return

            logger.info(f"🔍 Checking {len(positions_with_oco)} active OCO orders...")
            for position in positions_with_oco:
                logger.debug(
                    f"   📋 OCO {position.oco_order_id} | Trade {position.trade_id} | User {position.user_id}"
                )

            for position in positions_with_oco:
                await self._process_oco_position(db, position)

        except Exception as e:
            logger.error(f"❌ Error in _check_oco_orders: {e}")

    async def _get_positions_with_oco(self, db: AsyncSession):
        result = await db.execute(
            select(OpenPosition).where(OpenPosition.oco_order_id.isnot(None))
        )
        return result.scalars().all()

    async def _process_oco_position(self, db: AsyncSession, position: OpenPosition):
        try:
            user_id = position.user_id
            oco_order_id = position.oco_order_id
            symbol = position.symbol

            logger.debug(f"🔎 Querying OCO status for order {oco_order_id}...")
            oco_status = self.binance.get_oco_order_status(oco_order_id)
            if not oco_status:
                logger.warning(
                    f"⚠️  [User {user_id}] Could not get OCO status for {oco_order_id}"
                )
                return

            order_status = oco_status.get("listOrderStatus", "UNKNOWN")
            logger.debug(f"📊 OCO {oco_order_id} status: {order_status}")

            if order_status == "ALL_DONE":
                logger.info(
                    f"🎯 [User {user_id}] OCO ORDER EXECUTED/DONE: {oco_order_id}"
                )
                await self._handle_oco_all_done(db, position, oco_status)
            elif order_status == "EXECUTING":
                logger.debug(
                    f"⏳ [User {user_id}] OCO {oco_order_id} still active for {symbol}"
                )
                await self._handle_oco_executing(db, position)
            elif order_status in ["REJECT", "CANCELLING"]:
                logger.warning(
                    f"⚠️  [User {user_id}] OCO {oco_order_id} has status: {order_status}. It may need manual review."
                )
            else:
                logger.warning(
                    f"⚠️  [User {user_id}] OCO {oco_order_id} has unexpected listOrderStatus: {order_status}"
                )

        except Exception as exc:
            logger.error(
                f"❌ Error processing OCO order {position.oco_order_id}: {exc}"
            )

    async def _handle_oco_all_done(
        self, db: AsyncSession, position: OpenPosition, oco_status: Dict
    ) -> None:
        user_id = position.user_id
        symbol = position.symbol

        exit_reason, exit_price = self._extract_oco_exit_details(oco_status)
        if not exit_reason or not exit_price:
            logger.warning(
                f"⚠️  [User {user_id}] Could not determine exit details from OCO status for {position.oco_order_id}"
            )
            # Fallback: try to extract from individual order details
            exit_reason, exit_price = self._fallback_extract_exit_details(
                position, oco_status
            )

            if not exit_reason or not exit_price:
                logger.error(
                    f"❌ [User {user_id}] Fallback extraction failed. Manually closing position for {symbol}"
                )
                await self._manually_close_stale_position(db, position)
                return

        amount = float(position.amount)
        entry_price = float(position.entry_price)
        entry_fees = float(position.fees_paid)
        exit_fee = self._estimate_exit_fee(amount, exit_price)
        profit_loss = self._calculate_profit_loss(
            position.side, entry_price, exit_price, amount, entry_fees, exit_fee
        )
        profit_loss_pct = (profit_loss / float(position.entry_value)) * 100
        duration = (datetime.now(timezone.utc) - position.opened_at).total_seconds()

        await self._update_trade_record_after_exit(
            db,
            position.trade_id,
            exit_price,
            exit_fee,
            exit_reason,
            profit_loss,
            profit_loss_pct,
            duration,
        )

        await self._delete_position_record(db, position)

        # Remove from risk manager memory
        self._remove_position_from_risk_manager(user_id, position.trade_id)

        actual_usdt_balance = self._get_current_usdt_balance()

        self._log_oco_position_closed(
            user_id,
            symbol,
            position.side,
            amount,
            entry_price,
            exit_price,
            exit_reason,
            profit_loss,
            profit_loss_pct,
            entry_fees + exit_fee,
            duration,
            actual_usdt_balance,
        )

        metrics_logger.info(
            f"OCO_POSITION_CLOSED | USER={user_id} | SYMBOL={symbol} | REASON={exit_reason} | ENTRY=${entry_price:.4f} | EXIT=${exit_price:.4f} | PL=${profit_loss:.2f} | PL_PCT={profit_loss_pct:+.2f} | DURATION={duration:.0f}s | BALANCE=${actual_usdt_balance:.2f}"
        )

        # 🛡️ Record trade result in Circuit Breaker
        is_winner = profit_loss > 0
        self.circuit_breaker.record_trade_result(
            user_id=user_id, pnl=profit_loss, is_winner=is_winner
        )

        await self._notify_oco_position_closed(
            user_id,
            symbol,
            position.side,
            amount,
            entry_price,
            exit_price,
            profit_loss,
            profit_loss_pct,
            exit_reason,
            actual_usdt_balance,
        )

    def _extract_oco_exit_details(self, oco_status: Dict) -> tuple:
        """
        Extract exit reason and exit price from OCO status response.
        
        When an OCO order completes (listOrderStatus = "ALL_DONE"), one order will be FILLED 
        and the other CANCELED. The FILLED order tells us whether TP or SL was hit.
        
        Returns:
            tuple: (exit_reason, exit_price) or (None, None) if extraction fails
        """
        try:
            orders = oco_status.get("orders", [])
            if not orders:
                logger.warning("⚠️  OCO status has no orders array")
                return None, None
            
            filled_order = self._find_filled_order(orders)
            if not filled_order:
                return None, None
            
            exit_reason = self._determine_exit_reason(filled_order)
            exit_price = self._extract_exit_price(filled_order)
            
            if exit_price and exit_price > 0:
                logger.info(f"✅ OCO exit extracted: {exit_reason} at ${exit_price:.4f}")
                return exit_reason, exit_price
            
            logger.warning(f"⚠️  Invalid exit price: {exit_price}")
            return None, None
            
        except Exception as e:
            logger.error(f"❌ Error extracting OCO exit details: {e}")
            return None, None

    @staticmethod
    def _find_filled_order(orders: list) -> Optional[Dict]:
        """Find the FILLED order from OCO orders list"""
        for order in orders:
            if order.get("status") == "FILLED":
                return order
        logger.warning("⚠️  No FILLED order found in OCO status")
        return None

    @staticmethod
    def _determine_exit_reason(filled_order: Dict) -> str:
        """Determine if exit was TAKE_PROFIT or STOP_LOSS based on order type"""
        order_type = filled_order.get("type", "")
        
        if "LIMIT_MAKER" in order_type or (order_type == "LIMIT" and "STOP" not in order_type):
            return "TAKE_PROFIT"
        elif "STOP" in order_type:
            return "STOP_LOSS"
        else:
            logger.warning(f"⚠️  Unknown OCO order type: {order_type}")
            return "OCO_EXIT"

    @staticmethod
    def _extract_exit_price(filled_order: Dict) -> Optional[float]:
        """Extract execution price from filled order, with fallback calculation"""
        exit_price = float(filled_order.get("price", 0))
        
        if exit_price > 0:
            return exit_price
        
        # Fallback: calculate from cummulativeQuoteQty / executedQty
        executed_qty = float(filled_order.get("executedQty", 0))
        cumulative_quote = float(filled_order.get("cummulativeQuoteQty", 0))
        
        if executed_qty > 0 and cumulative_quote > 0:
            calculated_price = cumulative_quote / executed_qty
            logger.debug(f"💡 Calculated exit price from quote/qty: ${calculated_price:.4f}")
            return calculated_price
        
        return None

    def _fallback_extract_exit_details(
        self, position: OpenPosition, oco_status: Dict
    ) -> tuple:
        """
        Fallback method to extract exit details when primary extraction fails.
        
        Attempts two strategies:
        1. Check individual order details in OCO status for executedQty > 0
        2. Infer from position's TP/SL prices and current market price
        
        Returns:
            tuple: (exit_reason, exit_price) or (None, None) if all strategies fail
        """
        try:
            logger.info(f"🔍 Attempting fallback exit detail extraction for {position.symbol}")
            
            # Strategy 1: Find executed order in OCO status
            orders = oco_status.get("orders", [])
            result = self._extract_from_executed_order(orders)
            if result != (None, None):
                return result
            
            # Strategy 2: Infer from position TP/SL and current market price
            return self._infer_from_market_price(position)
            
        except Exception as e:
            logger.error(f"❌ Fallback exit detail extraction failed: {e}")
            return None, None

    @staticmethod
    def _extract_from_executed_order(orders: list) -> tuple:
        """Extract exit details from orders with executedQty > 0"""
        for order in orders:
            executed_qty = float(order.get("executedQty", 0))
            if executed_qty <= 0:
                continue
            
            # Found an executed order
            order_type = order.get("type", "")
            exit_reason = "STOP_LOSS" if "STOP" in order_type else "TAKE_PROFIT"
            
            # Get price from stopPrice or price field
            exit_price = float(order.get("stopPrice" if "STOP" in order_type else "price", 0))
            
            # Calculate average price if available
            cumulative_quote = float(order.get("cummulativeQuoteQty", 0))
            if executed_qty > 0 and cumulative_quote > 0:
                exit_price = cumulative_quote / executed_qty
            
            if exit_price > 0:
                logger.info(f"✅ Fallback Strategy 1: {exit_reason} at ${exit_price:.4f}")
                return exit_reason, exit_price
        
        return None, None

    def _infer_from_market_price(self, position: OpenPosition) -> tuple:
        """Infer exit reason/price from position TP/SL and current market price"""
        symbol = position.symbol
        current_price = self.binance.get_symbol_price(symbol)
        tp_price = float(position.take_profit)
        sl_price = float(position.stop_loss)
        
        # Calculate distances from current price
        tp_distance = abs(current_price - tp_price)
        sl_distance = abs(current_price - sl_price)
        
        # Whichever is closer to current price likely executed
        if tp_distance < sl_distance:
            exit_reason = "TAKE_PROFIT"
            exit_price = tp_price
        else:
            exit_reason = "STOP_LOSS"
            exit_price = sl_price
        
        logger.info(
            f"✅ Fallback Strategy 2: Inferred {exit_reason} at ${exit_price:.4f} "
            f"(current: ${current_price:.4f})"
        )
        
        return exit_reason, exit_price

    async def _handle_oco_executing(
        self, db: AsyncSession, position: OpenPosition
    ) -> None:
        """
        Monitor open positions and enforce day trading timeout rules.

        Day Trading Exit Rules:
        1. Maximum Hold: 8 hours (480 minutes) - force close to avoid overnight exposure
        2. OCO Orders: Stop-loss and take-profit handled automatically by Binance

        The adaptive R:R system places stops at support/resistance levels with ATR buffers,
        so we don't need aggressive time-based exits like scalping systems.
        Let the plan, not the clock, dictate the exit.
        """
        user_id = position.user_id
        trade_id = position.trade_id

        # Calculate position duration
        duration_seconds = (
            datetime.now(timezone.utc) - position.opened_at
        ).total_seconds()
        duration_minutes = duration_seconds / 60

        # Day trading timeout: 8 hours maximum hold
        if duration_minutes <= POSITION_TIMEOUT_MINUTES:
            return

        logger.warning(
            f"⏰ [User {user_id}] Position {trade_id} open for {duration_minutes:.1f} minutes "
            f"(max: {POSITION_TIMEOUT_MINUTES}) - FORCE CLOSING (DAY TRADING TIMEOUT)"
        )

        # Cancel OCO and force close
        symbol = position.symbol
        cancel_success = self.binance.cancel_oco_order(symbol, position.oco_order_id)
        if not cancel_success:
            logger.error(
                f"❌ [User {user_id}] Failed to cancel OCO {position.oco_order_id}"
            )
            return

        await self._force_close_timed_out_position(db, position, duration_minutes)

    async def _manually_close_stale_position(
        self, db: AsyncSession, position: OpenPosition
    ) -> None:
        """
        Manually close a position when we can't determine exit details from Binance.
        Uses current market price as the exit price.
        """
        user_id = position.user_id
        symbol = position.symbol

        logger.warning(
            f"🔧 [User {user_id}] Manually closing stale position for {symbol}"
        )

        try:
            # Get current market price as exit price
            current_price = self.binance.get_symbol_price(symbol)
            amount = float(position.amount)
            entry_price = float(position.entry_price)
            entry_fees = float(position.fees_paid)
            exit_fee = self._estimate_exit_fee(amount, current_price)

            profit_loss = self._calculate_profit_loss(
                position.side, entry_price, current_price, amount, entry_fees, exit_fee
            )
            profit_loss_pct = (profit_loss / float(position.entry_value)) * 100
            duration = (datetime.now(timezone.utc) - position.opened_at).total_seconds()

            # Infer exit reason based on P&L
            exit_reason = "TAKE_PROFIT" if profit_loss > 0 else "STOP_LOSS"

            logger.info(f"🔧 [User {user_id}] Manual close details:")
            logger.info(f"   💰 Current Price: ${current_price:.4f}")
            logger.info(f"   📈 Entry: ${entry_price:.4f} → Exit: ${current_price:.4f}")
            logger.info(f"   💵 P&L: ${profit_loss:.2f} ({profit_loss_pct:+.2f}%)")
            logger.info(f"   📝 Inferred Reason: {exit_reason}")

            await self._update_trade_record_after_exit(
                db,
                position.trade_id,
                current_price,
                exit_fee,
                f"MANUAL_CLOSE_{exit_reason}",
                profit_loss,
                profit_loss_pct,
                duration,
            )

            await self._delete_position_record(db, position)
            self._remove_position_from_risk_manager(user_id, position.trade_id)

            actual_usdt_balance = self._get_current_usdt_balance()

            logger.info(f"✅ [User {user_id}] Stale position manually closed")
            logger.info(f"   🏦 USDT Balance: ${actual_usdt_balance:.2f}")

            await self._notify_oco_position_closed(
                user_id,
                symbol,
                position.side,
                amount,
                entry_price,
                current_price,
                profit_loss,
                profit_loss_pct,
                f"MANUAL_CLOSE_{exit_reason}",
                actual_usdt_balance,
            )

        except Exception as e:
            logger.error(f"❌ [User {user_id}] Failed to manually close position: {e}")

    @staticmethod
    def _estimate_exit_fee(amount: float, exit_price: float) -> float:
        return amount * exit_price * 0.001

    @staticmethod
    def _calculate_profit_loss(
        side: str,
        entry_price: float,
        exit_price: float,
        amount: float,
        entry_fees: float,
        exit_fee: float,
    ) -> float:
        side_upper = side.upper()
        if side_upper == "BUY":
            gross = (exit_price - entry_price) * amount
        else:
            gross = (entry_price - exit_price) * amount
        return gross - entry_fees - exit_fee

    async def _update_trade_record_after_exit(
        self,
        db: AsyncSession,
        trade_id: str,
        exit_price: float,
        exit_fee: float,
        exit_reason: str,
        profit_loss: float,
        profit_loss_pct: float,
        duration_seconds: float,
    ) -> None:
        logger.debug(
            f"🔍 Attempting to update trade record {trade_id} with exit details"
        )
        trade_query = select(Trade).where(Trade.trade_id == trade_id)
        trade_result = await db.execute(trade_query)
        trade_obj = trade_result.scalar_one_or_none()

        if trade_obj:
            logger.info(
                f"✅ Found trade {trade_id} in database - updating exit details"
            )
            trade_obj.closed_at = datetime.now(timezone.utc)
            trade_obj.exit_price = exit_price
            trade_obj.exit_fee = exit_fee
            trade_obj.exit_reason = exit_reason
            trade_obj.profit_loss = profit_loss
            trade_obj.profit_loss_percentage = profit_loss_pct
            trade_obj.duration_seconds = int(duration_seconds)
            trade_obj.status = "closed"

            logger.debug(
                f"   📝 Exit Price: ${exit_price:.4f}, Exit Fee: ${exit_fee:.4f}"
            )
            logger.debug(f"   📝 P&L: ${profit_loss:.2f} ({profit_loss_pct:+.2f}%)")
            logger.debug(
                f"   📝 Reason: {exit_reason}, Duration: {int(duration_seconds)}s"
            )

            await db.commit()
            logger.info(f"💾 Trade record {trade_id} committed to database")

            # Verify the update
            await db.refresh(trade_obj)
            if trade_obj.closed_at:
                logger.info(
                    f"✅ Verified: closed_at = {trade_obj.closed_at.isoformat()}"
                )
            else:
                logger.error("❌ ERROR: closed_at is still NULL after commit!")
        else:
            logger.error(f"❌ CRITICAL: Trade record {trade_id} NOT FOUND in database!")
            logger.error("   Cannot update exit details for non-existent trade")
            logger.error(
                "   This indicates a mismatch between OpenPosition and Trade tables"
            )

    async def _delete_position_record(
        self, db: AsyncSession, position: OpenPosition
    ) -> None:
        await db.delete(position)
        await db.commit()
        logger.debug(
            f"✅ [User {position.user_id}] Open position {position.trade_id} removed from database"
        )

    def _log_oco_position_closed(
        self,
        user_id: int,
        symbol: str,
        side: str,
        amount: float,
        entry_price: float,
        exit_price: float,
        exit_reason: str,
        profit_loss: float,
        profit_loss_pct: float,
        total_fees: float,
        duration_seconds: float,
        actual_usdt_balance: float,
    ) -> None:
        profit_emoji = "💰" if profit_loss > 0 else "📉"
        logger.info(f"{profit_emoji} [User {user_id}] POSITION CLOSED VIA OCO:")
        logger.info(f"   📊 {side.upper()} {amount:.6f} {symbol}")
        logger.info(f"   📈 Entry: ${entry_price:.4f} → Exit: ${exit_price:.4f}")
        logger.info(f"   🎯 Reason: {exit_reason}")
        logger.info(f"   💵 P&L: ${profit_loss:.2f} ({profit_loss_pct:+.2f}%)")
        logger.info(f"   💸 Total Fees: ${total_fees:.4f}")
        logger.info(f"   ⏱️  Duration: {duration_seconds/60:.1f} minutes")
        logger.info(f"   🏦 Binance USDT Balance: ${actual_usdt_balance:.2f}")

    async def _notify_oco_position_closed(
        self,
        user_id: int,
        symbol: str,
        side: str,
        amount: float,
        entry_price: float,
        exit_price: float,
        profit_loss: float,
        profit_loss_pct: float,
        exit_reason: str,
        actual_usdt_balance: float,
    ) -> None:
        await self.ws_manager.send_to_user(
            user_id,
            {
                "type": "oco_position_closed",
                "position": {
                    "symbol": symbol,
                    "side": side,
                    "amount": amount,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "profit_loss": profit_loss,
                    "profit_loss_percentage": profit_loss_pct,
                },
                "exit_reason": exit_reason,
                "new_balance": actual_usdt_balance,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    # REMOVED: _manually_close_position_early() - Not needed for day trading
    # Day trading uses adaptive R:R with S/R-based stops, not time-based quick exits

    async def _force_close_timed_out_position(
        self, db: AsyncSession, position: OpenPosition, minutes_open: float
    ) -> None:
        symbol = position.symbol
        user_id = position.user_id
        exit_side = "SELL" if position.side.upper() == "BUY" else "BUY"

        exit_order = self.binance.place_market_order(
            symbol=symbol, side=exit_side, quantity=float(position.amount)
        )

        if not exit_order:
            logger.error(
                f"❌ [User {user_id}] Failed to force close position {position.trade_id}"
            )
            return

        exit_price = float(exit_order["fills"][0]["price"])
        exit_fee = float(exit_order["fills"][0]["commission"])
        amount = float(position.amount)
        entry_price = float(position.entry_price)
        entry_fees = float(position.fees_paid)

        profit_loss = self._calculate_profit_loss(
            position.side, entry_price, exit_price, amount, entry_fees, exit_fee
        )
        profit_loss_pct = (profit_loss / float(position.entry_value)) * 100

        await self._update_trade_record_after_exit(
            db,
            position.trade_id,
            exit_price,
            exit_fee,
            "TIMEOUT_AUTO_CLOSE",
            profit_loss,
            profit_loss_pct,
            minutes_open * 60,
        )

        await self._delete_position_record(db, position)

        # Remove from risk manager memory
        self._remove_position_from_risk_manager(user_id, position.trade_id)

        actual_usdt_balance = self._get_current_usdt_balance()

        logger.info("⏰ [User %s] POSITION FORCE CLOSED (TIMEOUT):", user_id)
        logger.info(
            "   📊 %s %.6f %s",
            position.side.upper(),
            amount,
            symbol,
        )
        logger.info(
            "   📈 Entry: $%.4f → Exit: $%.4f",
            entry_price,
            exit_price,
        )
        logger.info("   💵 P&L: $%.2f (%+.2f%%)", profit_loss, profit_loss_pct)
        logger.info("   ⏱️  Duration: %.1f minutes", minutes_open)
        logger.info("   🏦 USDT Balance: $%.2f", actual_usdt_balance)

        await self._notify_position_timeout_closed(
            user_id,
            symbol,
            position.side,
            amount,
            entry_price,
            exit_price,
            profit_loss,
            profit_loss_pct,
            actual_usdt_balance,
        )

    async def _notify_position_timeout_closed(
        self,
        user_id: int,
        symbol: str,
        side: str,
        amount: float,
        entry_price: float,
        exit_price: float,
        profit_loss: float,
        profit_loss_pct: float,
        new_balance: float,
    ) -> None:
        await self.ws_manager.send_to_user(
            user_id,
            {
                "type": "position_timeout_closed",
                "position": {
                    "symbol": symbol,
                    "side": side,
                    "amount": amount,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "profit_loss": profit_loss,
                    "profit_loss_percentage": profit_loss_pct,
                },
                "exit_reason": "TIMEOUT_AUTO_CLOSE",
                "new_balance": new_balance,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def _execute_trade(
        self, db: AsyncSession, config: TradingConfig, signal: Dict, params: Dict, regime_analysis: Dict = None
    ):
        """Execute a trade based on the signal"""
        trade_start_time = datetime.now(timezone.utc)
        user_id = config.user_id
        symbol = config.trading_pair
        side = signal["signal"].upper()  # BUY or SELL

        try:
            logger.info(f"🚀 [User {user_id}] EXECUTING TRADE:")
            logger.info(f"   📊 Symbol: {symbol}")
            logger.info(f"   📈 Side: {side}")
            logger.info(f"   💰 Quantity: {params['position_size']:.6f}")
            logger.info(f"   💵 Trade Value: ${params.get('trade_value', 0):.2f}")
            logger.info(
                f"   🌐 Binance Network: {'Testnet' if settings.binance_testnet else 'Mainnet (REAL MONEY)'}"
            )
            logger.info(
                f"   🎯 AI Confidence: {signal.get('final_confidence', signal.get('confidence', 0)):.1f}%"
            )

            # Calculate take profit and stop loss prices using ADAPTIVE approach
            current_price = self.binance.get_symbol_price(symbol)

            # Fetch market data for adaptive exit calculations
            kline_data = self.binance.get_kline_data(symbol, "1m", 100)
            df_with_indicators = self.ai_analyzer.calculate_technical_indicators(
                kline_data
            )

            # Get adaptive exit levels based on ATR and historical performance
            # Pass regime_analysis for mean reversion detection
            adaptive_exits = await self.risk_manager.get_adaptive_exit_levels(
                db, user_id, df_with_indicators, side.lower(), current_price,
                regime_analysis=regime_analysis  # NEW: Pass regime info
            )

            take_profit_price = adaptive_exits["take_profit_price"]
            stop_loss_price = adaptive_exits["stop_loss_price"]

            logger.info(
                f"🎯 [User {user_id}] Adaptive Exit Levels: "
                f"Entry=${current_price:.4f}, "
                f"TP=${take_profit_price:.4f} ({adaptive_exits['take_profit_pct']:.2f}%), "
                f"SL=${stop_loss_price:.4f} ({adaptive_exits['stop_loss_pct']:.2f}%), "
                f"R:R={adaptive_exits['risk_reward_ratio']:.2f}:1, "
                f"Source={adaptive_exits['source']}"
            )

            # Place order with OCO (entry + automatic TP/SL)
            logger.debug(
                f"📤 [User {user_id}] Placing {side} order with OCO on Binance..."
            )
            oco_result = self.binance.place_order_with_oco(
                symbol=symbol,
                side=side,
                quantity=params["position_size"],
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
            )

            if not oco_result or "entry_order" not in oco_result:
                logger.error(
                    f"❌ [User {user_id}] TRADE FAILED: OCO order placement failed"
                )
                metrics_logger.info(
                    f"TRADE_FAILED | USER={user_id} | SYMBOL={symbol} | SIDE={side} | REASON=OCO_ORDER_PLACEMENT_FAILED"
                )
                return

            # Extract entry order details
            entry_order = oco_result["entry_order"]
            oco_order = oco_result["oco_order"]
            executed_qty = float(entry_order["executedQty"])
            fill_price = float(entry_order["fills"][0]["price"])
            commission = float(entry_order["fills"][0]["commission"])
            order_id = entry_order["orderId"]
            oco_order_id = oco_order.get("orderListId", None)

            logger.info(f"✅ [User {user_id}] ORDER FILLED:")
            logger.info(f"   🆔 Order ID: {order_id}")
            logger.info(f"   � OCO Order ID: {oco_order_id}")
            logger.info(f"   �📊 Executed Qty: {executed_qty:.6f}")
            logger.info(f"   💰 Fill Price: ${fill_price:.4f}")
            logger.info(f"   💸 Commission: ${commission:.4f}")
            logger.info(f"   🎯 TP Order Active: ${take_profit_price:.4f}")
            logger.info(f"   🛑 SL Order Active: ${stop_loss_price:.4f}")

            # Record trade in database
            logger.debug(f"💾 [User {user_id}] Recording trade in database...")
            trade = Trade(
                user_id=user_id,
                trade_id=str(order_id),
                symbol=symbol,
                side=side.lower(),
                amount=float(executed_qty),
                price=float(fill_price),
                total_value=float(params["trade_value"]),
                fee=float(commission),
                status="filled",
                is_test_trade=config.is_test_mode,
                strategy_used="day_trading_ai",
                ai_signal_confidence=float(
                    signal.get("final_confidence", signal.get("confidence", 0))
                ),
                oco_order_id=str(oco_order_id) if oco_order_id else None,
            )

            db.add(trade)
            await db.commit()
            logger.debug(f"✅ [User {user_id}] Trade recorded in database")

            # Create open position in database
            open_position = OpenPosition(
                user_id=user_id,
                symbol=symbol,
                side=side.lower(),
                amount=executed_qty,
                entry_price=fill_price,
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
                trade_id=str(order_id),
                entry_value=executed_qty * fill_price,
                fees_paid=commission,
                is_test_trade=config.is_test_mode,
                oco_order_id=str(oco_order_id) if oco_order_id else None,
            )

            db.add(open_position)
            await db.commit()
            logger.debug(f"✅ [User {user_id}] Open position saved to database")

            # Add position to tracking system
            position_data = {
                "trade_id": str(order_id),
                "symbol": symbol,
                "side": side.lower(),
                "amount": executed_qty,
                "entry_price": fill_price,
                "stop_loss": stop_loss_price,
                "take_profit": take_profit_price,
                "entry_time": trade_start_time,
                "entry_value": executed_qty * fill_price,
                "fees_paid": commission,
                "oco_order_id": str(oco_order_id) if oco_order_id else None,
            }

            self.risk_manager.add_open_position(user_id, position_data)

            # Update risk manager
            self.risk_manager.record_trade(user_id)

            # Calculate trade metrics
            total_cost = executed_qty * fill_price + commission
            trade_duration = (
                datetime.now(timezone.utc) - trade_start_time
            ).total_seconds()

            # Get current Binance account balance for accurate reporting
            fresh_balances = self.binance.get_account_balance()
            current_usdt_balance = fresh_balances.get("USDT", {}).get("free", 0.0)

            # Log comprehensive trade entry with structured format
            logger.info(f"📈 [User {user_id}] POSITION OPENED:")
            logger.info(f"   📊 Trade: {side} {executed_qty:.6f} {symbol}")
            logger.info(f"   💰 Entry Price: ${fill_price:.4f}")
            logger.info(f"   🎯 Take Profit: ${take_profit_price:.4f} (OCO Active)")
            logger.info(f"   🛑 Stop Loss: ${stop_loss_price:.4f} (OCO Active)")
            logger.info(f"   💵 Position Value: ${total_cost:.2f}")
            logger.info(f"   💸 Entry Fee: ${commission:.4f}")
            logger.info(f"   ⏱️  Execution Time: {trade_duration:.2f}s")
            logger.info(f"   🏦 Binance USDT Balance: ${current_usdt_balance:.2f}")
            logger.info(
                f"   🧠 AI Confidence: {signal.get('final_confidence', signal.get('confidence', 0)):.1f}%"
            )
            logger.info(f"   🤖 OCO Order: {oco_order_id}")

            # Log to metrics file for analysis
            metrics_logger.info(
                f"POSITION_OPENED | USER={user_id} | SYMBOL={symbol} | SIDE={side} | QTY={executed_qty:.6f} | ENTRY=${fill_price:.4f} | TP=${take_profit_price:.4f} | SL=${stop_loss_price:.4f} | VALUE=${total_cost:.2f} | FEE=${commission:.4f} | CONFIDENCE={signal.get('final_confidence', signal.get('confidence', 0)):.1f} | BALANCE=${current_usdt_balance:.2f} | NETWORK={'TESTNET' if settings.binance_testnet else 'MAINNET'} | IS_TEST_TRADE={config.is_test_mode} | OCO={oco_order_id}"
            )

            # Send trade notification
            await self.ws_manager.send_to_user(
                user_id,
                {
                    "type": "trade_executed",
                    "trade": {
                        "symbol": symbol,
                        "side": side.lower(),
                        "amount": executed_qty,
                        "price": fill_price,
                        "total_value": total_cost,
                        "confidence": signal.get(
                            "final_confidence", signal.get("confidence", 0)
                        ),
                        "binance_testnet": settings.binance_testnet,
                        "is_test_trade": config.is_test_mode,
                        "order_id": order_id,
                        "commission": commission,
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Show a summary of open positions
            open_positions = self.risk_manager.get_open_positions(user_id)
            logger.info(
                f"📋 [User {user_id}] Open Positions: {len(open_positions)} | Total Value: ${sum(p['entry_value'] for p in open_positions):.2f}"
            )

            logger.info(f"✅ [User {user_id}] POSITION OPENED SUCCESSFULLY!")

        except Exception as e:
            trade_duration = (
                datetime.now(timezone.utc) - trade_start_time
            ).total_seconds()
            logger.error(
                f"❌ [User {user_id}] TRADE EXECUTION ERROR after {trade_duration:.2f}s: {e}"
            )
            metrics_logger.info(
                f"TRADE_ERROR | USER={user_id} | SYMBOL={symbol} | SIDE={side} | ERROR={str(e)} | DURATION={trade_duration:.2f}s"
            )

            # Send error notification to user
            await self.ws_manager.send_to_user(
                user_id,
                {
                    "type": "trade_error",
                    "error": f"Trade execution failed: {str(e)}",
                    "symbol": symbol,
                    "side": side.lower(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

    def _is_trading_hours(self, current_time: time) -> bool:
        """Check if current time is within trading hours"""
        try:
            # Parse start and end times from configuration
            start_time_str = settings.trading_active_hours_start.strip()
            end_time_str = settings.trading_active_hours_end.strip()

            # Parse start time (format: "HH:MM")
            start_parts = start_time_str.split(":")
            start_hour = int(start_parts[0])
            start_minute = int(start_parts[1]) if len(start_parts) > 1 else 0

            # Parse end time (format: "HH:MM")
            end_parts = end_time_str.split(":")
            end_hour = int(end_parts[0])
            end_minute = int(end_parts[1]) if len(end_parts) > 1 else 0

            # Convert current time to minutes since midnight for easier comparison
            current_minutes = current_time.hour * 60 + current_time.minute
            start_minutes = start_hour * 60 + start_minute
            end_minutes = end_hour * 60 + end_minute

            # Handle case where end time is on the next day (e.g., 23:59)
            if end_minutes <= start_minutes:
                # Trading spans midnight (e.g., 22:00 to 06:00)
                return (
                    current_minutes >= start_minutes or current_minutes <= end_minutes
                )
            else:
                # Normal case (e.g., 08:00 to 16:00 or 00:00 to 23:59)
                return start_minutes <= current_minutes <= end_minutes

        except Exception as e:
            logger.error(f"Error parsing trading hours: {e}")
            # Fallback to 24/7 trading if parsing fails
            logger.warning("Using 24/7 trading as fallback")
            return True

    async def _get_active_trading_configs(self, db: AsyncSession):
        """Get all active trading configurations"""
        from sqlalchemy import select

        query = select(TradingConfig).where(TradingConfig.is_active == True)
        result = await db.execute(query)
        return result.scalars().all()

    async def _load_open_positions_from_db(self):
        """Load open positions from database on startup"""
        try:
            async with get_async_session() as db:
                result = await db.execute(select(OpenPosition))
                open_positions = result.scalars().all()

                # Load positions into risk manager
                for position in open_positions:
                    position_data = {
                        "trade_id": position.trade_id,
                        "symbol": position.symbol,
                        "side": position.side,
                        "amount": float(position.amount),
                        "entry_price": float(position.entry_price),
                        "stop_loss": (
                            float(position.stop_loss) if position.stop_loss else None
                        ),
                        "take_profit": (
                            float(position.take_profit)
                            if position.take_profit
                            else None
                        ),
                        "entry_time": position.opened_at,
                        "entry_value": float(position.entry_value),
                        "fees_paid": float(position.fees_paid),
                        "oco_order_id": (
                            position.oco_order_id if position.oco_order_id else None
                        ),
                    }
                    self.risk_manager.add_open_position(position.user_id, position_data)

                logger.info(
                    f"📊 Loaded {len(open_positions)} open positions from database"
                )
                if open_positions:
                    oco_count = sum(1 for p in open_positions if p.oco_order_id)
                    logger.info(f"   🤖 {oco_count} positions have active OCO orders")

        except Exception as e:
            logger.error(f"❌ Error loading open positions from database: {e}")

    async def daily_reset_task(self):
        """Reset daily counters at midnight"""
        while self.is_running:
            try:
                now = datetime.now(timezone.utc)
                # Calculate seconds until midnight
                tomorrow = now.replace(
                    hour=0, minute=0, second=0, microsecond=0
                ) + timedelta(days=1)
                seconds_until_midnight = (tomorrow - now).total_seconds()

                logger.debug(
                    f"🕐 Next daily reset in {seconds_until_midnight/3600:.1f} hours at {tomorrow.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                await asyncio.sleep(seconds_until_midnight)

                if self.is_running:
                    logger.info("🔄 Performing daily reset...")
                    self.risk_manager.reset_daily_counters()
                    self.last_analysis_time.clear()  # Reset analysis timers

                    # 🛡️ Log Circuit Breaker daily summaries before reset
                    for user_id in self.circuit_breaker.daily_pnl.keys():
                        prev_pnl = self.circuit_breaker.daily_pnl.get(user_id, 0)
                        prev_trades = self.circuit_breaker.daily_trade_count.get(
                            user_id, 0
                        )

                        if prev_trades > 0:
                            logger.info(
                                f"📊 [User {user_id}] Daily Summary: "
                                f"{prev_trades} trades, ${prev_pnl:+.2f} P&L"
                            )
                            metrics_logger.info(
                                f"DAILY_SUMMARY | USER={user_id} | TRADES={prev_trades} | PNL=${prev_pnl:+.2f}"
                            )

                    logger.info("🔄 Daily counters and analysis timers reset")
                    metrics_logger.info("DAILY_RESET | STATUS=COMPLETED")

            except asyncio.CancelledError:
                logger.debug("Daily reset task cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Error in daily reset task: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry

    def _log_cycle_summary(
        self, cycle_count: int, duration: float, processed_users: int
    ):
        """Log a comprehensive summary of the trading cycle"""
        try:
            # Calculate total open positions and values across all users
            total_positions = 0
            total_position_value = 0.0

            user_summaries = []

            # Get all user IDs that have open positions
            all_user_ids = set()
            all_user_ids.update(self.risk_manager.open_positions.keys())

            for user_id in all_user_ids:
                open_positions = self.risk_manager.get_open_positions(user_id)

                if open_positions:
                    user_position_value = sum(p["entry_value"] for p in open_positions)
                    total_positions += len(open_positions)
                    total_position_value += user_position_value

                    user_summaries.append(
                        {
                            "user_id": user_id,
                            "positions": len(open_positions),
                            "value": user_position_value,
                        }
                    )

            # Log cycle completion
            logger.info(f"✅ TRADING CYCLE #{cycle_count} COMPLETED:")
            logger.info(
                f"   ⏱️  Duration: {duration:.2f}s | Processed Users: {processed_users}"
            )
            logger.info(
                f"   📊 System Status: {total_positions} open positions, ${total_position_value:.2f} total value"
            )

            # Log individual user summaries if there are positions
            if user_summaries:
                logger.info("📋 USER POSITION SUMMARY:")
                for summary in user_summaries:
                    logger.info(
                        f"   👤 User {summary['user_id']}: {summary['positions']} positions, ${summary['value']:.2f} invested"
                    )
            else:
                logger.info("📋 No open positions across all users")

            # Log to metrics
            metrics_logger.info(
                f"CYCLE_{cycle_count}_COMPLETED | DURATION={duration:.2f}s | USERS={processed_users} | POSITIONS={total_positions} | VALUE=${total_position_value:.2f}"
            )

        except Exception as e:
            logger.error(f"❌ Error logging cycle summary: {e}")
            # Fallback to simple logging
            logger.info(f"✅ Trading cycle #{cycle_count} completed in {duration:.2f}s")
            metrics_logger.info(
                f"CYCLE_{cycle_count}_COMPLETED | DURATION={duration:.2f}s | PROCESSED_USERS={processed_users}"
            )
