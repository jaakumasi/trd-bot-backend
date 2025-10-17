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
        self.regime_analyzer = MarketRegimeAnalyzer(filter_mode=settings.regime_filter_mode)
        self.circuit_breaker = CircuitBreaker()
        self.ws_manager = ws_manager
        self.is_running = False
        self.active_users = {}
        self.last_analysis_time = {}

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
            logger.error("❌ Failed to initialize Binance service - Trading Bot will not start!")
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
        if hasattr(self, 'trading_task') and not self.trading_task.done():
            self.trading_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.trading_task
            logger.debug("Trading task cancelled")
                
        if hasattr(self, 'daily_reset_task_handle') and not self.daily_reset_task_handle.done():
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

    async def _process_active_configs(
        self, db: AsyncSession, active_configs
    ) -> int:
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

            kline_data, current_price = market_snapshot

            # Calculate technical indicators FIRST (required by regime analyzer)
            logger.debug(f"📊 [User {user_id}] Calculating technical indicators...")
            kline_data = self.ai_analyzer.calculate_technical_indicators(kline_data)
            
            # Run market regime analysis with enriched data
            logger.debug(f"🔍 [User {user_id}] Analyzing market regime...")
            regime_analysis = await self.regime_analyzer.classify_market_regime(kline_data, symbol)
            
            # Check if scalping is allowed in current regime
            if not regime_analysis.get('allow_scalping', False):
                logger.info(
                    f"🚫 [User {user_id}] Scalping blocked - Regime: {regime_analysis.get('regime')} | "
                    f"ADX: {regime_analysis.get('trend_strength', 0):.1f} | "
                    f"ATR%: {regime_analysis.get('atr_percentage', 0):.2f}%"
                )
                return

            # Fetch user's trade history for personalized AI context
            logger.debug(f"📊 [User {user_id}] Fetching trade history for AI context...")
            user_trade_history = await self.ai_analyzer.get_user_trade_history_context(db, user_id)

            ai_signal = await self._request_ai_signal(symbol, kline_data, user_id, user_trade_history, regime_analysis)
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
                logger.critical(f"🛑 [User {user_id}] CIRCUIT BREAKER TRIGGERED: {cb_error.reason}")
                metrics_logger.info(f"CIRCUIT_BREAKER_HALT | USER={user_id} | REASON={cb_error.reason}")
                
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
                user_id, ai_signal, usdt_balance, config.__dict__
            )

            if not is_valid:
                await self._handle_trade_rejection(user_id, symbol, message)
                return

            self._log_trade_validation_success(user_id, trade_params)

            await self._execute_trade(db, config, ai_signal, trade_params)

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
        logger.debug(f"📈 [User {user_id}] Fetching market data for {symbol}...")
        try:
            kline_data = self.binance.get_kline_data(symbol, "1m", 100)
            if kline_data.empty:
                logger.warning(f"⚠️  [User {user_id}] No market data available for {symbol}")
                self.circuit_breaker.record_api_failure(f"Empty kline data for {symbol}")
                return None

            # API call successful
            self.circuit_breaker.record_api_success()
            
            current_price = float(kline_data.iloc[-1]["close"])
            logger.debug(f"💰 [User {user_id}] Current {symbol} price: ${current_price:.4f}")
            return kline_data, current_price
        except Exception as e:
            logger.error(f"❌ [User {user_id}] API error fetching market data: {e}")
            self.circuit_breaker.record_api_failure(f"get_kline_data failed: {str(e)}")
            return None

    async def _request_ai_signal(self, symbol, kline_data, user_id: int, user_trade_history=None, regime_analysis=None):
        logger.debug(f"🤖 [User {user_id}] Running AI analysis...")
        ai_signal = await self.ai_analyzer.analyze_market_data(symbol, kline_data, user_trade_history, regime_analysis)

        if ai_signal is None:
            logger.error(f"❌ [User {user_id}] AI analyzer returned None")
            return None

        if not isinstance(ai_signal, dict):
            logger.error(
                f"❌ [User {user_id}] AI analyzer returned invalid type: {type(ai_signal)}"
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
                logger.debug(f"❌ [User {user_id}] Invalid current price: {current_price}")
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
            logger.error(f"❌ [User {user_id}] Failed to place close order for {symbol}")
            return

        executed_qty = float(close_order_result["executedQty"])
        exit_price = float(close_order_result["fills"][0]["price"])
        exit_commission = float(close_order_result["fills"][0]["commission"])

        # Calculate P&L metrics (same as OCO/timeout paths)
        entry_price = position["entry_price"]
        entry_fees = position["fees_paid"]
        profit_loss = self._calculate_profit_loss(
            position["side"], entry_price, exit_price, position["amount"], entry_fees, exit_commission
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

        profit_label = (
            "PROFIT" if closed_position["net_pnl"] > 0 else "LOSS"
        )
        metrics_logger.info(
            f"POSITION_CLOSED | USER={user_id} | SYMBOL={symbol} | REASON={exit_reason} | PNL=${closed_position['net_pnl']:+.2f} | PCT={closed_position['pnl_percentage']:+.2f}% | DURATION={closed_position['duration_seconds']:.0f}s | RESULT={profit_label}"
        )

        # 🛡️ Record trade result in Circuit Breaker
        is_winner = closed_position["net_pnl"] > 0
        self.circuit_breaker.record_trade_result(
            user_id=user_id,
            pnl=closed_position["net_pnl"],
            is_winner=is_winner
        )

        # ✅ FIX: Update original trade record with exit details (consistent with OCO/timeout exits)
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
            logger.debug(
                f"✅ [DB] Open position removed for trade {trade_id}"
            )

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
                logger.debug(f"   📋 OCO {position.oco_order_id} | Trade {position.trade_id} | User {position.user_id}")

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
                logger.info(f"🎯 [User {user_id}] OCO ORDER EXECUTED/DONE: {oco_order_id}")
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
            exit_reason, exit_price = await self._fallback_extract_exit_details(position, oco_status)
            
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
            user_id=user_id,
            pnl=profit_loss,
            is_winner=is_winner
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

    async def _handle_oco_executing(
        self, db: AsyncSession, position: OpenPosition
    ) -> None:
        """
        Enhanced position monitoring with time-based exit rules.
        
        Exit Rules (optimized for scalping):
        1. Quick Profit: 5min + 0.2% profit → close immediately
        2. Breakeven Timeout: 20min + stagnant (-0.1% to +0.1%) → close at breakeven
        3. Time Stop-Loss: 45min + losing >0.5% → force close
        4. Original Timeout: 60min → force close (final backstop)
        """
        user_id = position.user_id
        symbol = position.symbol
        trade_id = position.trade_id
        
        # Calculate position metrics
        duration_seconds = (datetime.now(timezone.utc) - position.opened_at).total_seconds()
        duration_minutes = duration_seconds / 60
        
        current_price = self.binance.get_symbol_price(symbol)
        entry_price = float(position.entry_price)
        
        # Calculate current P&L percentage
        if position.side.lower() == 'buy':
            current_pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:  # sell
            current_pnl_pct = ((entry_price - current_price) / entry_price) * 100
        
        # TIME-BASED EXIT RULES
        
        # Rule 1: Quick Exit for Small Profits (Scalping Logic)
        # If position is 5+ minutes old and showing 0.2%+ profit, take it
        if duration_seconds > 300 and current_pnl_pct > 0.2:
            logger.info(
                f"💰 [User {user_id}] QUICK PROFIT EXIT: {duration_minutes:.1f}min old, "
                f"{current_pnl_pct:+.2f}% profit"
            )
            await self._manually_close_position_early(db, position, "QUICK_PROFIT", current_price)
            return
        
        # Rule 2: Breakeven Exit for Stagnant Positions
        # If position is 20+ minutes old and still near breakeven (-0.1% to +0.1%), exit
        if duration_seconds > 1200 and -0.1 < current_pnl_pct < 0.1:
            logger.warning(
                f"⚠️  [User {user_id}] BREAKEVEN TIMEOUT EXIT: {duration_minutes:.1f}min old, "
                f"stagnant at {current_pnl_pct:+.2f}%"
            )
            await self._manually_close_position_early(db, position, "BREAKEVEN_TIMEOUT", current_price)
            return
        
        # Rule 3: Force Exit for Significant Losing Positions
        # If position is 45+ minutes old and losing >0.5%, cut it to prevent bigger losses
        if duration_seconds > 2700 and current_pnl_pct < -0.5:
            logger.error(
                f"❌ [User {user_id}] TIME STOP-LOSS EXIT: {duration_minutes:.1f}min old, "
                f"losing {current_pnl_pct:.2f}%"
            )
            await self._manually_close_position_early(db, position, "TIME_STOP_LOSS", current_price)
            return
        
        # Rule 4: Original 30-minute timeout (legacy rule)
        if duration_minutes <= POSITION_TIMEOUT_MINUTES:
            return

        logger.warning(
            f"⏰ [User {user_id}] Position {trade_id} open for {duration_minutes:.1f} minutes - FORCE CLOSING (TIMEOUT)"
        )

        cancel_success = self.binance.cancel_oco_order(symbol, position.oco_order_id)
        if not cancel_success:
            logger.error(
                f"❌ [User {user_id}] Failed to cancel OCO {position.oco_order_id}"
            )
            return

        await self._force_close_timed_out_position(db, position, duration_minutes)

    @staticmethod
    def _extract_oco_exit_details(oco_status: Dict) -> Tuple[Optional[str], Optional[float]]:
        """
        Extract which OCO leg was filled and at what price.
        When an OCO order completes:
        - One order will have status "FILLED" (this is the one that executed)
        - The other order will have status "EXPIRED" (automatically expired by Binance)
        
        Returns: (exit_reason, exit_price)
        """
        logger.debug("🔍 Extracting exit details from OCO status")
        
        filled_order = TradingBot._find_filled_order(oco_status)
        if filled_order:
            return TradingBot._process_filled_order(filled_order)
        
        expired_order = TradingBot._find_expired_order(oco_status)
        if expired_order:
            return TradingBot._infer_from_expired_order(oco_status, expired_order)
        
        logger.warning("⚠️  No FILLED or EXPIRED order found in OCO status")
        return None, None
    
    @staticmethod
    def _find_filled_order(oco_status: Dict) -> Optional[Dict]:
        """Find the FILLED order in OCO status"""
        for order in oco_status.get("orders", []):
            if order.get("status") == "FILLED":
                logger.debug(f"   📋 Found FILLED order {order.get('orderId')}")
                return order
        return None
    
    @staticmethod
    def _find_expired_order(oco_status: Dict) -> Optional[Dict]:
        """Find the EXPIRED order in OCO status"""
        for order in oco_status.get("orders", []):
            if order.get("status") == "EXPIRED":
                logger.debug(f"   📋 Found EXPIRED order {order.get('orderId')}")
                return order
        return None
    
    @staticmethod
    def _process_filled_order(filled_order: Dict) -> Tuple[str, float]:
        """Process a filled order to determine exit reason and price"""
        price = float(filled_order.get("price", 0))
        order_type = filled_order.get("type", "")
        
        if "STOP" in order_type.upper():
            logger.info(f"🛑 Stop Loss triggered at ${price:.4f}")
            return "STOP_LOSS", price
        
        logger.info(f"🎯 Take Profit triggered at ${price:.4f}")
        return "TAKE_PROFIT", price
    
    @staticmethod
    def _infer_from_expired_order(oco_status: Dict, expired_order: Dict) -> Tuple[Optional[str], Optional[float]]:
        """Infer which leg filled based on which one expired"""
        expired_type = expired_order.get("type", "")
        logger.debug(f"   🔄 No FILLED order found, inferring from EXPIRED {expired_type}")
        
        # If STOP expired, then TAKE_PROFIT filled (and vice versa)
        if "STOP" in expired_type.upper():
            return TradingBot._find_take_profit_price(oco_status, expired_order)
        else:
            return TradingBot._find_stop_loss_price(oco_status, expired_order)
    
    @staticmethod
    def _find_take_profit_price(oco_status: Dict, expired_order: Dict) -> Tuple[str, float]:
        """Find take profit price from non-expired order"""
        logger.info("✅ Inferred: Take Profit filled (Stop Loss expired)")
        for order in oco_status.get("orders", []):
            if order.get("orderId") != expired_order.get("orderId"):
                tp_price = float(order.get("price", 0))
                return "TAKE_PROFIT", tp_price
        return None, None
    
    @staticmethod
    def _find_stop_loss_price(oco_status: Dict, expired_order: Dict) -> Tuple[str, float]:
        """Find stop loss price from non-expired order"""
        logger.info("✅ Inferred: Stop Loss filled (Take Profit expired)")
        for order in oco_status.get("orders", []):
            if order.get("orderId") != expired_order.get("orderId"):
                sl_price = float(order.get("stopPrice") or order.get("price", 0))
                return "STOP_LOSS", sl_price
        return None, None

    def _fallback_extract_exit_details(
        self, position: OpenPosition, oco_status: Dict
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Fallback method to extract exit details when primary extraction fails.
        Checks expired orders and uses recent trades to infer which leg executed.
        """
        logger.info(f"🔄 [User {position.user_id}] Attempting fallback extraction for OCO {position.oco_order_id}")
        
        # Strategy 1: Direct check for filled order
        filled_result = self._check_for_filled_order(oco_status)
        if filled_result:
            return filled_result
        
        # Strategy 2: Infer from expired order
        expired_result = self._infer_from_expired_order_fallback(position, oco_status)
        if expired_result:
            return expired_result
        
        logger.warning(f"⚠️  Fallback extraction also failed for OCO {position.oco_order_id}")
        return None, None
    
    @staticmethod
    def _check_for_filled_order(oco_status: Dict) -> Optional[Tuple[str, float]]:
        """Check if we can find a filled order directly"""
        for order in oco_status.get("orders", []):
            if order.get("status") == "FILLED":
                price = float(order.get("price", 0))
                order_type = order.get("type", "")
                
                if "STOP" in order_type.upper():
                    logger.info(f"✅ Fallback found: Stop Loss at ${price:.4f}")
                    return "STOP_LOSS", price
                else:
                    logger.info(f"✅ Fallback found: Take Profit at ${price:.4f}")
                    return "TAKE_PROFIT", price
        return None
    
    def _infer_from_expired_order_fallback(
        self, position: OpenPosition, oco_status: Dict
    ) -> Optional[Tuple[str, float]]:
        """Infer exit details from expired order in fallback scenario"""
        expired_order = self._find_expired_order_in_fallback(oco_status)
        if not expired_order:
            return None
        
        try:
            expired_type = expired_order.get("type", "")
            current_price = self.binance.get_symbol_price(position.symbol)
            expired_price = float(expired_order.get("stopPrice") or expired_order.get("price", 0))
            
            logger.debug(f"📊 Expired order: type={expired_type}, price=${expired_price:.4f}, current=${current_price:.4f}")
            
            # Determine which leg filled based on which expired
            if "STOP" in expired_type.upper():
                return self._extract_take_profit_from_orders(oco_status, expired_order)
            else:
                return self._extract_stop_loss_from_orders(oco_status, expired_order)
                
        except Exception as e:
            logger.error(f"❌ Error in fallback inference: {e}")
            return None
    
    @staticmethod
    def _find_expired_order_in_fallback(oco_status: Dict) -> Optional[Dict]:
        """Find expired order in fallback scenario"""
        for order in oco_status.get("orders", []):
            if order.get("status") == "EXPIRED":
                return order
        return None
    
    @staticmethod
    def _extract_take_profit_from_orders(oco_status: Dict, expired_order: Dict) -> Optional[Tuple[str, float]]:
        """Extract take profit price from non-expired order"""
        for order in oco_status.get("orders", []):
            if order.get("orderId") != expired_order.get("orderId"):
                tp_price = float(order.get("price", 0))
                logger.info(f"✅ Fallback inferred: Take Profit at ${tp_price:.4f}")
                return "TAKE_PROFIT", tp_price
        return None
    
    @staticmethod
    def _extract_stop_loss_from_orders(oco_status: Dict, expired_order: Dict) -> Optional[Tuple[str, float]]:
        """Extract stop loss price from non-expired order"""
        for order in oco_status.get("orders", []):
            if order.get("orderId") != expired_order.get("orderId"):
                sl_price = float(order.get("stopPrice") or order.get("price", 0))
                logger.info(f"✅ Fallback inferred: Stop Loss at ${sl_price:.4f}")
                return "STOP_LOSS", sl_price
        return None

    async def _manually_close_stale_position(
        self, db: AsyncSession, position: OpenPosition
    ) -> None:
        """
        Manually close a position when we can't determine exit details from Binance.
        Uses current market price as the exit price.
        """
        user_id = position.user_id
        symbol = position.symbol
        
        logger.warning(f"🔧 [User {user_id}] Manually closing stale position for {symbol}")
        
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
        logger.debug(f"🔍 Attempting to update trade record {trade_id} with exit details")
        trade_query = select(Trade).where(Trade.trade_id == trade_id)
        trade_result = await db.execute(trade_query)
        trade_obj = trade_result.scalar_one_or_none()

        if trade_obj:
            logger.info(f"✅ Found trade {trade_id} in database - updating exit details")
            trade_obj.closed_at = datetime.now(timezone.utc)
            trade_obj.exit_price = exit_price
            trade_obj.exit_fee = exit_fee
            trade_obj.exit_reason = exit_reason
            trade_obj.profit_loss = profit_loss
            trade_obj.profit_loss_percentage = profit_loss_pct
            trade_obj.duration_seconds = int(duration_seconds)
            trade_obj.status = "closed"
            
            logger.debug(f"   📝 Exit Price: ${exit_price:.4f}, Exit Fee: ${exit_fee:.4f}")
            logger.debug(f"   📝 P&L: ${profit_loss:.2f} ({profit_loss_pct:+.2f}%)")
            logger.debug(f"   📝 Reason: {exit_reason}, Duration: {int(duration_seconds)}s")
            
            await db.commit()
            logger.info(f"💾 Trade record {trade_id} committed to database")
            
            # Verify the update
            await db.refresh(trade_obj)
            if trade_obj.closed_at:
                logger.info(f"✅ Verified: closed_at = {trade_obj.closed_at.isoformat()}")
            else:
                logger.error("❌ ERROR: closed_at is still NULL after commit!")
        else:
            logger.error(f"❌ CRITICAL: Trade record {trade_id} NOT FOUND in database!")
            logger.error("   Cannot update exit details for non-existent trade")
            logger.error("   This indicates a mismatch between OpenPosition and Trade tables")

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

    async def _manually_close_position_early(
        self, db: AsyncSession, position: OpenPosition, exit_reason: str, current_price: float
    ) -> None:
        """
        Manually close a position early due to time-based rules.
        Cancels the OCO order and places a market order to exit.
        
        Args:
            db: Database session
            position: The OpenPosition to close
            exit_reason: One of 'QUICK_PROFIT', 'BREAKEVEN_TIMEOUT', 'TIME_STOP_LOSS'
            current_price: Current market price for logging
        """
        user_id = position.user_id
        symbol = position.symbol
        trade_id = position.trade_id
        
        logger.info(
            f"🔧 [User {user_id}] Initiating early position close | "
            f"Reason: {exit_reason} | Current Price: ${current_price:.2f}"
        )
        
        try:
            # Step 1: Cancel the OCO order
            logger.debug(f"🔧 [User {user_id}] Canceling OCO order {position.oco_order_id}...")
            cancel_success = self.binance.cancel_oco_order(symbol, position.oco_order_id)
            
            if not cancel_success:
                logger.error(
                    f"❌ [User {user_id}] Failed to cancel OCO {position.oco_order_id} for early exit"
                )
                return
            
            # Step 2: Place market order to close position
            exit_side = "SELL" if position.side.upper() == "BUY" else "BUY"
            
            logger.debug(f"📤 [User {user_id}] Placing market {exit_side} order to close position...")
            exit_order = self.binance.place_market_order(
                symbol=symbol, side=exit_side, quantity=float(position.amount)
            )
            
            if not exit_order:
                logger.error(
                    f"❌ [User {user_id}] Failed to place market exit order for {trade_id}"
                )
                return
            
            # Step 3: Extract exit details
            exit_price = float(exit_order["fills"][0]["price"])
            exit_fee = float(exit_order["fills"][0]["commission"])
            amount = float(position.amount)
            entry_price = float(position.entry_price)
            entry_fees = float(position.fees_paid)
            
            # Step 4: Calculate P&L
            profit_loss = self._calculate_profit_loss(
                position.side, entry_price, exit_price, amount, entry_fees, exit_fee
            )
            profit_loss_pct = (profit_loss / float(position.entry_value)) * 100
            duration_seconds = (datetime.now(timezone.utc) - position.opened_at).total_seconds()
            
            # Step 5: Update trade record
            await self._update_trade_record_after_exit(
                db,
                trade_id,
                exit_price,
                exit_fee,
                exit_reason,
                profit_loss,
                profit_loss_pct,
                duration_seconds,
            )
            
            # Step 6: Delete position from database
            await self._delete_position_record(db, position)
            
            # Step 7: Remove from risk manager memory
            self._remove_position_from_risk_manager(user_id, trade_id)
            
            # Step 8: Get updated balance and log
            actual_usdt_balance = self._get_current_usdt_balance()
            
            emoji = "💰" if profit_loss > 0 else "❌" if profit_loss < 0 else "⚖️"
            logger.info(f"{emoji} [User {user_id}] POSITION CLOSED EARLY ({exit_reason}):")
            logger.info(f"   📊 {position.side.upper()} {amount:.6f} {symbol}")
            logger.info(f"   📈 Entry: ${entry_price:.4f} → Exit: ${exit_price:.4f}")
            logger.info(f"   💵 P&L: ${profit_loss:+.2f} ({profit_loss_pct:+.2f}%)")
            logger.info(f"   ⏱️  Duration: {duration_seconds/60:.1f} minutes")
            logger.info(f"   💰 New Balance: ${actual_usdt_balance:.2f}")
            
            metrics_logger.info(
                f"EARLY_EXIT | USER={user_id} | SYMBOL={symbol} | REASON={exit_reason} | "
                f"ENTRY=${entry_price:.4f} | EXIT=${exit_price:.4f} | PL=${profit_loss:+.2f} | "
                f"PL_PCT={profit_loss_pct:+.2f} | DURATION={duration_seconds:.0f}s | BALANCE=${actual_usdt_balance:.2f}"
            )
            
            # 🛡️ Record trade result in Circuit Breaker
            is_winner = profit_loss > 0
            self.circuit_breaker.record_trade_result(
                user_id=user_id,
                pnl=profit_loss,
                is_winner=is_winner
            )
            
            # Step 9: Notify user via WebSocket
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
            
        except Exception as e:
            logger.error(f"❌ [User {user_id}] Error in early position close: {e}")

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
        logger.info(
            "   💵 P&L: $%.2f (%+.2f%%)", profit_loss, profit_loss_pct
        )
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
        self, db: AsyncSession, config: TradingConfig, signal: Dict, params: Dict
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
            logger.info(f"   🌐 Binance Network: {'Testnet' if settings.binance_testnet else 'Mainnet (REAL MONEY)'}")
            logger.info(f"   🎯 AI Confidence: {signal.get('final_confidence', signal.get('confidence', 0)):.1f}%")

            # Calculate take profit and stop loss prices using ADAPTIVE approach
            current_price = self.binance.get_symbol_price(symbol)
            
            # Fetch market data for adaptive exit calculations
            kline_data = self.binance.get_kline_data(symbol, "1m", 100)
            df_with_indicators = self.ai_analyzer.calculate_technical_indicators(kline_data)
            
            # Get adaptive exit levels based on ATR and historical performance
            adaptive_exits = await self.risk_manager.get_adaptive_exit_levels(
                db, user_id, df_with_indicators, side.lower(), current_price
            )
            
            take_profit_price = adaptive_exits['take_profit_price']
            stop_loss_price = adaptive_exits['stop_loss_price']
            
            logger.info(
                f"🎯 [User {user_id}] Adaptive Exit Levels: "
                f"Entry=${current_price:.4f}, "
                f"TP=${take_profit_price:.4f} ({adaptive_exits['take_profit_pct']:.2f}%), "
                f"SL=${stop_loss_price:.4f} ({adaptive_exits['stop_loss_pct']:.2f}%), "
                f"R:R={adaptive_exits['risk_reward_ratio']:.2f}:1, "
                f"Source={adaptive_exits['source']}"
            )

            # Place order with OCO (entry + automatic TP/SL)
            logger.debug(f"📤 [User {user_id}] Placing {side} order with OCO on Binance...")
            oco_result = self.binance.place_order_with_oco(
                symbol=symbol,
                side=side,
                quantity=params["position_size"],
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
            )

            if not oco_result or 'entry_order' not in oco_result:
                logger.error(f"❌ [User {user_id}] TRADE FAILED: OCO order placement failed")
                metrics_logger.info(f"TRADE_FAILED | USER={user_id} | SYMBOL={symbol} | SIDE={side} | REASON=OCO_ORDER_PLACEMENT_FAILED")
                return

            # Extract entry order details
            entry_order = oco_result['entry_order']
            oco_order = oco_result['oco_order']
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
                strategy_used="scalping_ai",
                ai_signal_confidence=float(signal.get("final_confidence", signal.get("confidence", 0))),
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
                'trade_id': str(order_id),
                'symbol': symbol,
                'side': side.lower(),
                'amount': executed_qty,
                'entry_price': fill_price,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'entry_time': trade_start_time,
                'entry_value': executed_qty * fill_price,
                'fees_paid': commission,
                'oco_order_id': str(oco_order_id) if oco_order_id else None,
            }
            
            self.risk_manager.add_open_position(user_id, position_data)
            
            # Update risk manager
            self.risk_manager.record_trade(user_id)

            # Calculate trade metrics
            total_cost = executed_qty * fill_price + commission
            trade_duration = (datetime.now(timezone.utc) - trade_start_time).total_seconds()
            
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
            logger.info(f"   🧠 AI Confidence: {signal.get('final_confidence', signal.get('confidence', 0)):.1f}%")
            logger.info(f"   🤖 OCO Order: {oco_order_id}")

            # Log to metrics file for analysis
            metrics_logger.info(f"POSITION_OPENED | USER={user_id} | SYMBOL={symbol} | SIDE={side} | QTY={executed_qty:.6f} | ENTRY=${fill_price:.4f} | TP=${take_profit_price:.4f} | SL=${stop_loss_price:.4f} | VALUE=${total_cost:.2f} | FEE=${commission:.4f} | CONFIDENCE={signal.get('final_confidence', signal.get('confidence', 0)):.1f} | BALANCE=${current_usdt_balance:.2f} | NETWORK={'TESTNET' if settings.binance_testnet else 'MAINNET'} | IS_TEST_TRADE={config.is_test_mode} | OCO={oco_order_id}")

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
                        "confidence": signal.get("final_confidence", signal.get("confidence", 0)),
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
            logger.info(f"📋 [User {user_id}] Open Positions: {len(open_positions)} | Total Value: ${sum(p['entry_value'] for p in open_positions):.2f}")
            
            logger.info(f"✅ [User {user_id}] POSITION OPENED SUCCESSFULLY!")

        except Exception as e:
            trade_duration = (datetime.now(timezone.utc) - trade_start_time).total_seconds()
            logger.error(f"❌ [User {user_id}] TRADE EXECUTION ERROR after {trade_duration:.2f}s: {e}")
            metrics_logger.info(f"TRADE_ERROR | USER={user_id} | SYMBOL={symbol} | SIDE={side} | ERROR={str(e)} | DURATION={trade_duration:.2f}s")
            
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
                return current_minutes >= start_minutes or current_minutes <= end_minutes
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
                        'trade_id': position.trade_id,
                        'symbol': position.symbol,
                        'side': position.side,
                        'amount': float(position.amount),
                        'entry_price': float(position.entry_price),
                        'stop_loss': float(position.stop_loss) if position.stop_loss else None,
                        'take_profit': float(position.take_profit) if position.take_profit else None,
                        'entry_time': position.opened_at,
                        'entry_value': float(position.entry_value),
                        'fees_paid': float(position.fees_paid),
                        'oco_order_id': position.oco_order_id if position.oco_order_id else None,
                    }
                    self.risk_manager.add_open_position(position.user_id, position_data)
                
                logger.info(f"📊 Loaded {len(open_positions)} open positions from database")
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

                logger.debug(f"🕐 Next daily reset in {seconds_until_midnight/3600:.1f} hours at {tomorrow.strftime('%Y-%m-%d %H:%M:%S')}")
                await asyncio.sleep(seconds_until_midnight)

                if self.is_running:
                    logger.info("🔄 Performing daily reset...")
                    self.risk_manager.reset_daily_counters()
                    self.last_analysis_time.clear()  # Reset analysis timers
                    
                    # 🛡️ Log Circuit Breaker daily summaries before reset
                    for user_id in self.circuit_breaker.daily_pnl.keys():
                        prev_pnl = self.circuit_breaker.daily_pnl.get(user_id, 0)
                        prev_trades = self.circuit_breaker.daily_trade_count.get(user_id, 0)
                        
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
    
    def _log_cycle_summary(self, cycle_count: int, duration: float, processed_users: int):
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
                    user_position_value = sum(p['entry_value'] for p in open_positions)
                    total_positions += len(open_positions)
                    total_position_value += user_position_value
                    
                    user_summaries.append({
                        'user_id': user_id,
                        'positions': len(open_positions),
                        'value': user_position_value
                    })
            
            # Log cycle completion
            logger.info(f"✅ TRADING CYCLE #{cycle_count} COMPLETED:")
            logger.info(f"   ⏱️  Duration: {duration:.2f}s | Processed Users: {processed_users}")
            logger.info(f"   📊 System Status: {total_positions} open positions, ${total_position_value:.2f} total value")
            
            # Log individual user summaries if there are positions
            if user_summaries:
                logger.info("📋 USER POSITION SUMMARY:")
                for summary in user_summaries:
                    logger.info(f"   👤 User {summary['user_id']}: {summary['positions']} positions, ${summary['value']:.2f} invested")
            else:
                logger.info("📋 No open positions across all users")
                
            # Log to metrics
            metrics_logger.info(f"CYCLE_{cycle_count}_COMPLETED | DURATION={duration:.2f}s | USERS={processed_users} | POSITIONS={total_positions} | VALUE=${total_position_value:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error logging cycle summary: {e}")
            # Fallback to simple logging
            logger.info(f"✅ Trading cycle #{cycle_count} completed in {duration:.2f}s")
            metrics_logger.info(f"CYCLE_{cycle_count}_COMPLETED | DURATION={duration:.2f}s | PROCESSED_USERS={processed_users}")
