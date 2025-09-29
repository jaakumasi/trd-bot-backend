import asyncio
import logging
from datetime import datetime, time, timedelta
from typing import Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from ..database import get_db
from ..models.trade import TradingConfig, Trade
from ..models.user import User
from .binance_service import BinanceService
from .mock_binance_service import MockBinanceService
from .ai_analyzer import AIAnalyzer
from .risk_manager import RiskManager
from .websocket_manager import WebSocketManager
from ..logging_config import get_trading_metrics_logger
from ..config import settings
import json

logger = logging.getLogger(__name__)
metrics_logger = get_trading_metrics_logger()


class TradingBot:
    def __init__(self, ws_manager: WebSocketManager):
        # Choose Binance service based on configuration
        if settings.use_mock_binance:
            self.binance = MockBinanceService()
            logger.info("🧪 Using MOCK Binance service - All trades are simulated!")
        else:
            self.binance = BinanceService()
            logger.info("🔧 Using REAL Binance service")
            
        self.ai_analyzer = AIAnalyzer()
        self.risk_manager = RiskManager()
        self.ws_manager = ws_manager
        self.is_running = False
        self.active_users = {}
        self.last_analysis_time = {}

    async def initialize(self):
        """Initialize all services"""
        try:
            logger.info("🔧 Initializing Trading Bot services...")
            
            logger.info("📡 Connecting to Binance API...")
            binance_connected = await self.binance.initialize()
            if not binance_connected:
                logger.error("❌ Failed to initialize Binance service - Trading Bot will not start!")
                return False
            logger.info("✅ Binance service initialized successfully")

            logger.info("🧠 Initializing AI Analyzer...")
            logger.info("✅ AI Analyzer ready")
            
            logger.info("🛡️  Initializing Risk Manager...")
            logger.info("✅ Risk Manager ready")
            
            logger.info("🌐 WebSocket Manager ready")

            logger.info("🤖 Trading Bot initialized successfully")
            metrics_logger.info("BOT_INITIALIZED | STATUS=SUCCESS")
            return True

        except Exception as e:
            logger.error(f"❌ Bot initialization failed: {e}")
            logger.error(f"❌ This means the trading bot will NOT run!")
            metrics_logger.info(f"BOT_INITIALIZATION_FAILED | ERROR={str(e)}")
            return False

    async def start(self):
        """Start the trading bot"""
        if self.is_running:
            return True

        initialized = await self.initialize()
        if not initialized:
            return False

        # Log the trading hours configuration
        logger.info(f"⏰ Trading Hours: {settings.trading_active_hours_start} - {settings.trading_active_hours_end}")
        
        # Test current trading hours status
        current_time = datetime.now().time()
        is_trading_time = self._is_trading_hours(current_time)
        logger.info(f"🕐 Current time: {current_time.strftime('%H:%M:%S')} | Trading active: {'✅ Yes' if is_trading_time else '❌ No'}")

        self.is_running = True

        # Start main trading loop
        self.trading_task = asyncio.create_task(self.trading_loop())
        logger.info("🔄 Trading loop task created and started!")

        # Start daily reset task
        self.daily_reset_task_handle = asyncio.create_task(self.daily_reset_task())
        logger.info("⏰ Daily reset task created and started!")

        logger.info("🚀 Trading Bot started and running!")
        logger.info("⚡ You should see trading cycle logs every 60 seconds")
        metrics_logger.info("BOT_STARTED | STATUS=RUNNING")
        return True

    async def stop(self):
        """Stop the trading bot"""
        logger.info("🛑 Stopping Trading Bot...")
        self.is_running = False
        
        # Cancel running tasks
        if hasattr(self, 'trading_task') and not self.trading_task.done():
            self.trading_task.cancel()
            try:
                await self.trading_task
            except asyncio.CancelledError:
                logger.debug("Trading task cancelled")
                
        if hasattr(self, 'daily_reset_task_handle') and not self.daily_reset_task_handle.done():
            self.daily_reset_task_handle.cancel()
            try:
                await self.daily_reset_task_handle
            except asyncio.CancelledError:
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
                cycle_start_time = datetime.now()
                
                logger.info(f"🔄 Starting trading cycle #{cycle_count} at {cycle_start_time.strftime('%H:%M:%S')}")
                
                # Check if we're in trading hours
                current_time = datetime.now().time()
                if not self._is_trading_hours(current_time):
                    logger.info(f"⏰ Outside trading hours ({current_time.strftime('%H:%M:%S')}). Next check in 1 minute.")
                    await asyncio.sleep(60)  # Check again in 1 minute
                    continue

                # Load active trading configurations
                async for db in get_db():
                    active_configs = await self._get_active_trading_configs(db)
                    total_active_users = len(active_configs)
                    connected_users = self.ws_manager.get_connected_users()
                    total_connected = len(connected_users)
                    connection_count = self.ws_manager.get_connection_count()
                    
                    # Log user statistics
                    logger.info(f"👥 TRADING CYCLE #{cycle_count} STATS:")
                    logger.info(f"   📊 Active trading users: {total_active_users}")
                    logger.info(f"   🌐 Connected users: {total_connected}")
                    logger.info(f"   🔌 Total connections: {connection_count}")
                    
                    # Log metrics for analysis
                    metrics_logger.info(f"CYCLE_{cycle_count} | ACTIVE_USERS={total_active_users} | CONNECTED_USERS={total_connected} | CONNECTIONS={connection_count}")
                    
                    if total_active_users == 0:
                        logger.info("⚠️  No active trading users found")
                    else:
                        logger.info(f"🎯 Processing {total_active_users} active trading configurations...")

                    processed_users = 0
                    for config in active_configs:
                        try:
                            processed_users += 1
                            logger.debug(f"📈 Processing user {config.user_id} ({processed_users}/{total_active_users}) - {config.trading_pair}")
                            await self._process_user_trading(db, config)
                        except Exception as e:
                            logger.error(f"❌ Error processing user {config.user_id}: {e}")

                    cycle_duration = (datetime.now() - cycle_start_time).total_seconds()
                    logger.info(f"✅ Trading cycle #{cycle_count} completed in {cycle_duration:.2f}s")
                    metrics_logger.info(f"CYCLE_{cycle_count}_COMPLETED | DURATION={cycle_duration:.2f}s | PROCESSED_USERS={processed_users}")
                    
                    break  # Exit the db session loop

                # Wait before next analysis cycle (1 minute for scalping)
                logger.debug("⏳ Waiting 60 seconds before next cycle...")
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"❌ Trading loop error in cycle #{cycle_count}: {e}")
                await asyncio.sleep(10)  # Short delay before retry

    async def _process_user_trading(self, db: AsyncSession, config: TradingConfig):
        """Process trading for a specific user"""
        try:
            user_id = config.user_id
            symbol = config.trading_pair
            
            logger.debug(f"🔍 [User {user_id}] Starting trading analysis for {symbol}")

            # Rate limiting: Don't analyze too frequently per user
            last_analysis = self.last_analysis_time.get(user_id, 0)
            time_since_last = asyncio.get_event_loop().time() - last_analysis
            
            if time_since_last < 30:  # 30 seconds minimum
                logger.debug(f"⏱️  [User {user_id}] Rate limit active: {30 - time_since_last:.1f}s remaining")
                return

            logger.info(f"📊 [User {user_id}] ANALYZING {symbol} (Test: {'Yes' if config.is_test_mode else 'No'})")

            # Get market data
            logger.debug(f"📈 [User {user_id}] Fetching market data for {symbol}...")
            kline_data = self.binance.get_kline_data(symbol, "1m", 100)
            if kline_data.empty:
                logger.warning(f"⚠️  [User {user_id}] No market data available for {symbol}")
                return

            current_price = kline_data.iloc[-1]['close'] if not kline_data.empty else 0
            logger.debug(f"💰 [User {user_id}] Current {symbol} price: ${current_price:.4f}")

            # Get AI analysis
            logger.debug(f"🤖 [User {user_id}] Running AI analysis...")
            ai_signal = await self.ai_analyzer.analyze_market_data(symbol, kline_data)
            self.last_analysis_time[user_id] = asyncio.get_event_loop().time()

            # Validate AI signal response
            if ai_signal is None:
                logger.error(f"❌ [User {user_id}] AI analyzer returned None")
                return
            
            if not isinstance(ai_signal, dict):
                logger.error(f"❌ [User {user_id}] AI analyzer returned invalid type: {type(ai_signal)}")
                return

            # Ensure required fields exist with defaults
            signal = ai_signal.get("signal", "hold")
            confidence = ai_signal.get("confidence", 0)
            reasoning = ai_signal.get("reasoning", "No reasoning provided")

            # Log AI analysis results
            signal_emoji = {"buy": "🟢", "sell": "🔴", "hold": "🟡"}.get(signal, "❓")
            logger.info(f"{signal_emoji} [User {user_id}] AI SIGNAL: {signal.upper()} | Confidence: {confidence:.1f}% | {reasoning}")
            
            # Log detailed metrics
            metrics_logger.info(f"AI_ANALYSIS | USER={user_id} | SYMBOL={symbol} | SIGNAL={signal} | CONFIDENCE={confidence:.1f} | PRICE={current_price:.4f}")

            # Send analysis update via WebSocket
            await self.ws_manager.send_to_user(
                user_id,
                {
                    "type": "ai_analysis",
                    "symbol": symbol,
                    "signal": signal,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "timestamp": datetime.now().isoformat(),
                },
            )

            # Only proceed if we have a buy/sell signal
            if signal == "hold":
                logger.debug(f"⏸️  [User {user_id}] HOLD signal - no action taken")
                return

            logger.info(f"🎯 [User {user_id}] TRADE SIGNAL DETECTED: {signal.upper()}")

            # Get account balance
            logger.debug(f"💳 [User {user_id}] Checking account balance...")
            balances = self.binance.get_account_balance()
            usdt_balance = balances.get("USDT", {}).get("free", 0.0)
            logger.info(f"💰 [User {user_id}] Available USDT balance: ${usdt_balance:.2f}")

            # Validate trade with risk management
            logger.debug(f"🛡️  [User {user_id}] Validating trade with risk management...")
            is_valid, message, trade_params = self.risk_manager.validate_trade_signal(
                user_id, ai_signal, usdt_balance, config.__dict__
            )

            if not is_valid:
                logger.warning(f"❌ [User {user_id}] TRADE REJECTED: {message}")
                metrics_logger.info(f"TRADE_REJECTED | USER={user_id} | SYMBOL={symbol} | REASON={message}")
                await self.ws_manager.send_to_user(
                    user_id,
                    {
                        "type": "trade_rejected",
                        "reason": message,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
                return

            logger.info(f"✅ [User {user_id}] Trade validation PASSED - proceeding to execution")
            logger.info(f"📋 [User {user_id}] Trade params: Size=${trade_params.get('trade_value', 0):.2f}, Position=${trade_params.get('position_size', 0):.6f}")

            # Execute trade
            await self._execute_trade(db, config, ai_signal, trade_params)

        except Exception as e:
            logger.error(f"❌ [User {user_id}] Error in trading process: {e}")
            metrics_logger.info(f"TRADING_ERROR | USER={user_id} | ERROR={str(e)}")

    async def _execute_trade(
        self, db: AsyncSession, config: TradingConfig, signal: Dict, params: Dict
    ):
        """Execute a trade based on the signal"""
        trade_start_time = datetime.now()
        user_id = config.user_id
        symbol = config.trading_pair
        side = signal["signal"].upper()  # BUY or SELL
        
        try:
            logger.info(f"🚀 [User {user_id}] EXECUTING TRADE:")
            logger.info(f"   📊 Symbol: {symbol}")
            logger.info(f"   📈 Side: {side}")
            logger.info(f"   💰 Quantity: {params['position_size']:.6f}")
            logger.info(f"   💵 Trade Value: ${params.get('trade_value', 0):.2f}")
            logger.info(f"   🧪 Test Mode: {'Yes' if config.is_test_mode else 'No'}")
            logger.info(f"   🎯 AI Confidence: {signal.get('final_confidence', signal.get('confidence', 0)):.1f}%")

            # Place order on Binance
            logger.debug(f"📤 [User {user_id}] Placing {side} order on Binance...")
            order_result = self.binance.place_market_order(
                symbol=symbol,
                side=side,
                quantity=params["position_size"],
                test_mode=config.is_test_mode,
            )

            if not order_result:
                logger.error(f"❌ [User {user_id}] TRADE FAILED: Order placement failed")
                metrics_logger.info(f"TRADE_FAILED | USER={user_id} | SYMBOL={symbol} | SIDE={side} | REASON=ORDER_PLACEMENT_FAILED")
                return

            # Extract trade details
            executed_qty = float(order_result["executedQty"])
            fill_price = float(order_result["fills"][0]["price"])
            commission = float(order_result["fills"][0]["commission"])
            order_id = order_result["orderId"]
            
            logger.info(f"✅ [User {user_id}] ORDER FILLED:")
            logger.info(f"   🆔 Order ID: {order_id}")
            logger.info(f"   📊 Executed Qty: {executed_qty:.6f}")
            logger.info(f"   💰 Fill Price: ${fill_price:.4f}")
            logger.info(f"   💸 Commission: ${commission:.4f}")

            # Record trade in database
            logger.debug(f"💾 [User {user_id}] Recording trade in database...")
            trade = Trade(
                user_id=user_id,
                trade_id=order_id,
                symbol=symbol,
                side=side.lower(),
                amount=executed_qty,
                price=fill_price,
                total_value=params["trade_value"],
                fee=commission,
                status="filled",
                is_test_trade=config.is_test_mode,
                strategy_used="scalping_ai",
                ai_signal_confidence=signal.get("final_confidence", signal.get("confidence", 0)),
            )

            db.add(trade)
            await db.commit()
            logger.debug(f"✅ [User {user_id}] Trade recorded in database")

            # Update risk manager
            self.risk_manager.record_trade(user_id)
            logger.debug(f"📊 [User {user_id}] Risk manager updated")

            # Calculate trade metrics
            total_cost = executed_qty * fill_price + commission
            trade_duration = (datetime.now() - trade_start_time).total_seconds()
            
            # Log comprehensive trade metrics
            logger.info(f"📈 [User {user_id}] TRADE COMPLETED:")
            logger.info(f"   ⏱️  Execution Time: {trade_duration:.2f}s")
            logger.info(f"   💰 Total Cost: ${total_cost:.2f}")
            logger.info(f"   📊 Effective Price: ${fill_price:.4f}")
            logger.info(f"   💸 Fee %: {(commission/total_cost)*100:.3f}%")

            # Log to metrics file for analysis
            metrics_logger.info(f"TRADE_EXECUTED | USER={user_id} | SYMBOL={symbol} | SIDE={side} | QTY={executed_qty:.6f} | PRICE={fill_price:.4f} | VALUE=${total_cost:.2f} | FEE=${commission:.4f} | CONFIDENCE={signal.get('final_confidence', signal.get('confidence', 0)):.1f} | TEST={config.is_test_mode} | DURATION={trade_duration:.2f}s")

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
                        "test_mode": config.is_test_mode,
                        "order_id": order_id,
                        "commission": commission,
                    },
                    "timestamp": datetime.now().isoformat(),
                },
            )

            logger.info(f"🎉 [User {user_id}] TRADE SUCCESS: {side} {executed_qty:.6f} {symbol} @ ${fill_price:.4f}")

        except Exception as e:
            trade_duration = (datetime.now() - trade_start_time).total_seconds()
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
                    "timestamp": datetime.now().isoformat(),
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

    async def daily_reset_task(self):
        """Reset daily counters at midnight"""
        while self.is_running:
            try:
                now = datetime.now()
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
                    logger.info("🔄 Daily counters and analysis timers reset")
                    metrics_logger.info("DAILY_RESET | STATUS=COMPLETED")
                    
            except asyncio.CancelledError:
                logger.debug("Daily reset task cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Error in daily reset task: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry
