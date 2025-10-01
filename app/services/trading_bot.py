import asyncio
import logging
from datetime import datetime, time, timedelta, timezone
from typing import Dict, Optional
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
            
            # Load open positions from database
            await self._load_open_positions_from_db()
            
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
        current_time = datetime.now(timezone.utc).time()
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
                cycle_start_time = datetime.now(timezone.utc)
                
                logger.info(f"🔄 Starting trading cycle #{cycle_count} at {cycle_start_time.strftime('%H:%M:%S')}")
                
                # Check if we're in trading hours
                current_time = datetime.now(timezone.utc).time()
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
                            
                            # Check OCO orders first (automatic TP/SL from Binance)
                            await self._check_oco_orders(db)
                            
                            # Check for manual position exits (fallback if no OCO)
                            await self._check_position_exits(db, config)
                            
                            # Then process new trading signals
                            await self._process_user_trading(db, config)
                        except Exception as e:
                            logger.error(f"❌ Error processing user {config.user_id}: {e}")

                    cycle_duration = (datetime.now(timezone.utc) - cycle_start_time).total_seconds()
                    
                    # Log cycle summary with position overview
                    self._log_cycle_summary(cycle_count, cycle_duration, processed_users)
                    
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
                    "timestamp": datetime.now(timezone.utc).isoformat(),
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
                        "timestamp": datetime.now(timezone.utc).isoformat(),
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

    async def _check_position_exits(self, db: AsyncSession, config: TradingConfig):
        """Check if any open positions should be closed"""
        try:
            user_id = config.user_id
            symbol = config.trading_pair
            
            # Get current open positions for debugging
            current_positions = self.risk_manager.get_open_positions(user_id)
            if current_positions:
                logger.debug(f"🔍 [User {user_id}] Checking {len(current_positions)} open positions for exits")
            
            # Get current price
            current_price = self.binance.get_symbol_price(symbol)
            if not current_price or current_price <= 0.0:
                logger.debug(f"❌ [User {user_id}] Invalid current price: {current_price}")
                return
            
            logger.debug(f"💰 [User {user_id}] Current {symbol} price: ${current_price:.4f}")
            
            # Check for positions that should be closed
            positions_to_close = self.risk_manager.check_exit_conditions(user_id, current_price, symbol)
            
            if positions_to_close:
                logger.info(f"🚨 [User {user_id}] Found {len(positions_to_close)} positions ready to close!")
            else:
                logger.debug(f"✅ [User {user_id}] No positions ready to close at current price ${current_price:.4f}")
            
            for position_info in positions_to_close:
                position = position_info['position']
                exit_reason = position_info['exit_reason']
                
                logger.info(f"🚨 [User {user_id}] CLOSING POSITION: {exit_reason}")
                logger.info(f"   📊 Position: {position['side'].upper()} {position['amount']:.6f} {symbol}")
                logger.info(f"   💰 Entry: ${position['entry_price']:.4f} → Exit: ${current_price:.4f}")
                
                # Execute closing order (opposite side)
                close_side = "SELL" if position['side'].upper() == "BUY" else "BUY"
                
                close_order_result = self.binance.place_market_order(
                    symbol=symbol,
                    side=close_side,
                    quantity=position['amount'],
                    test_mode=config.is_test_mode,
                )
                
                if close_order_result:
                    # Extract closing order details
                    executed_qty = float(close_order_result["executedQty"])
                    exit_price = float(close_order_result["fills"][0]["price"])
                    exit_commission = float(close_order_result["fills"][0]["commission"])
                    
                    # Close the position and calculate P&L
                    closed_position = self.risk_manager.close_position(
                        user_id, position['trade_id'], exit_price, exit_reason, exit_commission
                    )
                    
                    if closed_position:
                        # Log comprehensive exit summary
                        duration = closed_position['duration_seconds']
                        duration_str = f"{int(duration//60)}m {int(duration%60)}s" if duration > 60 else f"{duration:.1f}s"
                        
                        logger.info(f"🎯 [User {user_id}] POSITION CLOSED SUMMARY:")
                        logger.info(f"   📈 Trade: {position['side'].upper()} → {close_side}")
                        logger.info(f"   💰 Entry: ${position['entry_price']:.4f} → Exit: ${exit_price:.4f}")
                        logger.info(f"   📊 Quantity: {executed_qty:.6f} {symbol}")
                        logger.info(f"   ⏱️  Duration: {duration_str}")
                        logger.info(f"   💸 Total Fees: ${closed_position['total_fees']:.4f}")
                        logger.info(f"   💵 P&L: ${closed_position['net_pnl']:+.2f} ({closed_position['pnl_percentage']:+.2f}%)")
                        
                        # Get actual Binance balance for accurate reporting
                        fresh_balances = self.binance.get_account_balance()
                        actual_usdt_balance = fresh_balances.get("USDT", {}).get("free", 0.0)
                        logger.info(f"   🏦 Binance USDT Balance: ${actual_usdt_balance:.2f}")
                        
                        # Log to metrics
                        profit_loss = "PROFIT" if closed_position['net_pnl'] > 0 else "LOSS"
                        metrics_logger.info(f"POSITION_CLOSED | USER={user_id} | SYMBOL={symbol} | REASON={exit_reason} | PNL=${closed_position['net_pnl']:+.2f} | PCT={closed_position['pnl_percentage']:+.2f}% | DURATION={duration:.0f}s | RESULT={profit_loss}")
                        
                        # Record closing trade in database
                        close_trade = Trade(
                            user_id=user_id,
                            trade_id=str(close_order_result["orderId"]),
                            symbol=symbol,
                            side=close_side.lower(),
                            amount=float(executed_qty),
                            price=float(exit_price),
                            total_value=float(executed_qty * exit_price),
                            fee=float(exit_commission),
                            status="filled",
                            is_test_trade=config.is_test_mode,
                            strategy_used="scalping_ai_exit",
                            ai_signal_confidence=0.0,  # This is a system-generated exit
                        )
                        
                        db.add(close_trade)
                        await db.commit()
                        
                        # Remove open position from database
                        position_query = select(OpenPosition).where(
                            OpenPosition.trade_id == position['trade_id']
                        )
                        db_position = await db.execute(position_query)
                        db_position_obj = db_position.scalar_one_or_none()
                        
                        if db_position_obj:
                            await db.delete(db_position_obj)
                            await db.commit()
                            logger.debug(f"✅ [User {user_id}] Open position removed from database")
                        
                        #Send notification with actual Binance balance
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
                
        except Exception as e:
            logger.error(f"❌ [User {user_id}] Error checking position exits: {e}")

    async def _check_oco_orders(self, db: AsyncSession):
        """
        Check status of all active OCO orders and update database when filled
        This runs periodically to detect when Binance executes TP or SL
        """
        try:
            # Query all open positions that have OCO orders
            result = await db.execute(
                select(OpenPosition).where(OpenPosition.oco_order_id.isnot(None))
            )
            positions_with_oco = result.scalars().all()
            
            if not positions_with_oco:
                logger.debug("🔍 No OCO orders to check")
                return
            
            logger.debug(f"🔍 Checking {len(positions_with_oco)} OCO orders...")
            
            for position in positions_with_oco:
                try:
                    user_id = position.user_id
                    oco_order_id = position.oco_order_id
                    symbol = position.symbol
                    
                    # Get OCO order status from Binance
                    oco_status = self.binance.get_oco_order_status(oco_order_id)
                    
                    if not oco_status:
                        logger.warning(f"⚠️  [User {user_id}] Could not get OCO status for {oco_order_id}")
                        continue
                    
                    order_status = oco_status.get('listOrderStatus', 'UNKNOWN')
                    
                    # Check if OCO has been executed (one leg filled, other cancelled)
                    if order_status == 'ALL_DONE':
                        logger.info(f"🎯 [User {user_id}] OCO ORDER EXECUTED: {oco_order_id}")
                        
                        # Determine which leg was executed (TP or SL)
                        orders = oco_status.get('orders', [])
                        exit_reason = None
                        exit_price = None
                        
                        for order in orders:
                            order_status_detail = order.get('status', '')
                            if order_status_detail == 'FILLED':
                                order_type = order.get('type', '')
                                exit_price = float(order.get('price', 0))
                                
                                # Determine if it was TP or SL
                                if 'STOP' in order_type.upper():
                                    exit_reason = 'STOP_LOSS'
                                    logger.info(f"🛑 [User {user_id}] Stop Loss triggered at ${exit_price:.4f}")
                                else:
                                    exit_reason = 'TAKE_PROFIT'
                                    logger.info(f"🎯 [User {user_id}] Take Profit triggered at ${exit_price:.4f}")
                                break
                        
                        if not exit_reason or not exit_price:
                            logger.warning(f"⚠️  [User {user_id}] Could not determine exit details for OCO {oco_order_id}")
                            continue
                        
                        # Calculate P&L
                        entry_price = float(position.entry_price)
                        amount = float(position.amount)
                        entry_fees = float(position.fees_paid)
                        
                        # Estimate exit fee (0.1% on Binance)
                        exit_value = amount * exit_price
                        exit_fee = exit_value * 0.001
                        
                        # Calculate P&L based on position side
                        if position.side.upper() == 'BUY':
                            profit_loss = (exit_price - entry_price) * amount - entry_fees - exit_fee
                        else:  # SELL
                            profit_loss = (entry_price - exit_price) * amount - entry_fees - exit_fee
                        
                        profit_loss_pct = (profit_loss / float(position.entry_value)) * 100
                        duration = (datetime.now(timezone.utc) - position.opened_at).total_seconds()
                        
                        # Update the original trade record
                        trade_query = select(Trade).where(Trade.trade_id == position.trade_id)
                        trade_result = await db.execute(trade_query)
                        trade_obj = trade_result.scalar_one_or_none()
                        
                        if trade_obj:
                            trade_obj.closed_at = datetime.now(timezone.utc)
                            trade_obj.exit_price = exit_price
                            trade_obj.exit_fee = exit_fee
                            trade_obj.exit_reason = exit_reason
                            trade_obj.profit_loss = profit_loss
                            trade_obj.profit_loss_percentage = profit_loss_pct
                            trade_obj.duration_seconds = int(duration)
                            trade_obj.status = 'closed'
                            
                            await db.commit()
                            logger.debug(f"✅ [User {user_id}] Trade record updated in database")
                        
                        # Remove open position from database
                        await db.delete(position)
                        await db.commit()
                        logger.debug(f"✅ [User {user_id}] Open position removed from database")
                        
                        # Remove from risk manager tracking
                        self.risk_manager.close_position(
                            user_id, 
                            position.trade_id, 
                            exit_price, 
                            exit_reason,
                            exit_fee
                        )
                        
                        # Get fresh balance
                        fresh_balances = self.binance.get_account_balance()
                        actual_usdt_balance = fresh_balances.get("USDT", {}).get("free", 0.0)
                        
                        # Log comprehensive exit
                        profit_emoji = "💰" if profit_loss > 0 else "📉"
                        logger.info(f"{profit_emoji} [User {user_id}] POSITION CLOSED VIA OCO:")
                        logger.info(f"   📊 {position.side.upper()} {amount:.6f} {symbol}")
                        logger.info(f"   📈 Entry: ${entry_price:.4f} → Exit: ${exit_price:.4f}")
                        logger.info(f"   🎯 Reason: {exit_reason}")
                        logger.info(f"   💵 P&L: ${profit_loss:.2f} ({profit_loss_pct:+.2f}%)")
                        logger.info(f"   💸 Total Fees: ${entry_fees + exit_fee:.4f}")
                        logger.info(f"   ⏱️  Duration: {duration/60:.1f} minutes")
                        logger.info(f"   🏦 Binance USDT Balance: ${actual_usdt_balance:.2f}")
                        
                        # Log to metrics
                        metrics_logger.info(f"OCO_POSITION_CLOSED | USER={user_id} | SYMBOL={symbol} | REASON={exit_reason} | ENTRY=${entry_price:.4f} | EXIT=${exit_price:.4f} | PL=${profit_loss:.2f} | PL_PCT={profit_loss_pct:+.2f} | DURATION={duration:.0f}s | BALANCE=${actual_usdt_balance:.2f}")
                        
                        # Send WebSocket notification
                        await self.ws_manager.send_to_user(
                            user_id,
                            {
                                "type": "oco_position_closed",
                                "position": {
                                    "symbol": symbol,
                                    "side": position.side,
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
                        
                    elif order_status == 'EXECUTING':
                        logger.debug(f"⏳ [User {user_id}] OCO {oco_order_id} still active for {symbol}")
                    else:
                        logger.warning(f"⚠️  [User {user_id}] OCO {oco_order_id} has unexpected status: {order_status}")
                        
                except Exception as e:
                    logger.error(f"❌ Error processing OCO order {position.oco_order_id}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Error in _check_oco_orders: {e}")

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
            logger.info(f"   🧪 Test Mode: {'Yes' if config.is_test_mode else 'No'}")
            logger.info(f"   🎯 AI Confidence: {signal.get('final_confidence', signal.get('confidence', 0)):.1f}%")

            # Calculate take profit and stop loss prices from signal
            current_price = self.binance.get_symbol_price(symbol)
            take_profit_price = float(signal.get('take_profit', 0))
            stop_loss_price = float(signal.get('stop_loss', 0))
            
            # Validate TP/SL prices
            if not take_profit_price or not stop_loss_price:
                logger.error(f"❌ [User {user_id}] Invalid TP/SL prices: TP={take_profit_price}, SL={stop_loss_price}")
                return
            
            logger.info(f"🎯 [User {user_id}] Order prices: Entry=${current_price:.4f}, TP=${take_profit_price:.4f}, SL=${stop_loss_price:.4f}")

            # Place order with OCO (entry + automatic TP/SL)
            logger.debug(f"📤 [User {user_id}] Placing {side} order with OCO on Binance...")
            oco_result = self.binance.place_order_with_oco(
                symbol=symbol,
                side=side,
                quantity=params["position_size"],
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                test_mode=config.is_test_mode,
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
            metrics_logger.info(f"POSITION_OPENED | USER={user_id} | SYMBOL={symbol} | SIDE={side} | QTY={executed_qty:.6f} | ENTRY=${fill_price:.4f} | TP=${take_profit_price:.4f} | SL=${stop_loss_price:.4f} | VALUE=${total_cost:.2f} | FEE=${commission:.4f} | CONFIDENCE={signal.get('final_confidence', signal.get('confidence', 0)):.1f} | BALANCE=${current_usdt_balance:.2f} | TEST={config.is_test_mode} | OCO={oco_order_id}")

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
