"""
Demo script to show the comprehensive logging system in action
This simulates trading bot activity with detailed logging
"""
import asyncio
import random
from datetime import datetime
from app.logging_config import setup_logging, get_trading_metrics_logger
import logging

# Initialize logging
metrics_logger = setup_logging()
logger = logging.getLogger('app.services.trading_bot')

async def simulate_trading_cycle():
    """Simulate a complete trading cycle with logging"""
    
    # Simulate cycle start
    cycle_num = random.randint(1, 100)
    logger.info(f"🔄 Starting trading cycle #{cycle_num} at {datetime.now().strftime('%H:%M:%S')}")
    
    # Simulate user stats
    active_users = random.randint(3, 8)
    connected_users = random.randint(2, active_users)
    connections = random.randint(connected_users, connected_users * 2)
    
    logger.info(f"👥 TRADING CYCLE #{cycle_num} STATS:")
    logger.info(f"   📊 Active trading users: {active_users}")
    logger.info(f"   🌐 Connected users: {connected_users}")
    logger.info(f"   🔌 Total connections: {connections}")
    
    metrics_logger.info(f"CYCLE_{cycle_num} | ACTIVE_USERS={active_users} | CONNECTED_USERS={connected_users} | CONNECTIONS={connections}")
    
    # Simulate processing each user
    for user_id in range(1, active_users + 1):
        await simulate_user_analysis(user_id, cycle_num)
        await asyncio.sleep(0.1)  # Small delay between users
    
    cycle_duration = random.uniform(45.0, 75.0)
    logger.info(f"✅ Trading cycle #{cycle_num} completed in {cycle_duration:.2f}s")
    metrics_logger.info(f"CYCLE_{cycle_num}_COMPLETED | DURATION={cycle_duration:.2f}s | PROCESSED_USERS={active_users}")

async def simulate_user_analysis(user_id: int, cycle_num: int):
    """Simulate analysis and potentially trade execution for a user"""
    
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "DOTUSDT"]
    symbol = random.choice(symbols)
    
    logger.debug(f"📈 Processing user {user_id} ({user_id}/{8}) - {symbol}")
    logger.info(f"📊 [User {user_id}] ANALYZING {symbol} (Test: {'Yes' if random.choice([True, False]) else 'No'})")
    
    # Simulate current price
    current_price = random.uniform(20000, 70000) if symbol == "BTCUSDT" else random.uniform(1000, 4000)
    logger.debug(f"💰 [User {user_id}] Current {symbol} price: ${current_price:.4f}")
    
    # Simulate AI analysis
    signals = ["buy", "sell", "hold"]
    signal = random.choice(signals)
    confidence = random.uniform(60, 95)
    reasoning = f"Technical indicators suggest {signal} signal with {confidence:.1f}% confidence"
    
    signal_emoji = {"buy": "🟢", "sell": "🔴", "hold": "🟡"}.get(signal, "❓")
    logger.info(f"{signal_emoji} [User {user_id}] AI SIGNAL: {signal.upper()} | Confidence: {confidence:.1f}% | {reasoning}")
    
    metrics_logger.info(f"AI_ANALYSIS | USER={user_id} | SYMBOL={symbol} | SIGNAL={signal} | CONFIDENCE={confidence:.1f} | PRICE={current_price:.4f}")
    
    # If not hold, simulate potential trade
    if signal != "hold":
        await simulate_trade_execution(user_id, symbol, signal, confidence, current_price)

async def simulate_trade_execution(user_id: int, symbol: str, signal: str, confidence: float, price: float):
    """Simulate trade execution with detailed logging"""
    
    # Simulate account balance check
    usdt_balance = random.uniform(100, 5000)
    logger.info(f"💰 [User {user_id}] Available USDT balance: ${usdt_balance:.2f}")
    
    # Simulate risk management validation
    if random.choice([True, True, True, False]):  # 75% chance of approval
        logger.info(f"✅ [User {user_id}] Trade validation PASSED - proceeding to execution")
        
        # Simulate trade parameters
        trade_value = random.uniform(50, min(500, usdt_balance * 0.1))
        position_size = trade_value / price
        
        logger.info(f"📋 [User {user_id}] Trade params: Size=${trade_value:.2f}, Position=${position_size:.6f}")
        
        # Simulate trade execution
        logger.info(f"🚀 [User {user_id}] EXECUTING TRADE:")
        logger.info(f"   📊 Symbol: {symbol}")
        logger.info(f"   📈 Side: {signal.upper()}")
        logger.info(f"   💰 Quantity: {position_size:.6f}")
        logger.info(f"   💵 Trade Value: ${trade_value:.2f}")
        logger.info(f"   🧪 Test Mode: {'Yes' if random.choice([True, False]) else 'No'}")
        logger.info(f"   🎯 AI Confidence: {confidence:.1f}%")
        
        # Simulate order execution
        await asyncio.sleep(0.2)  # Simulate execution time
        
        executed_qty = position_size * random.uniform(0.98, 1.0)  # Slight variation
        fill_price = price * random.uniform(0.9995, 1.0005)  # Small slippage
        commission = trade_value * 0.001  # 0.1% fee
        order_id = random.randint(100000, 999999)
        
        logger.info(f"✅ [User {user_id}] ORDER FILLED:")
        logger.info(f"   🆔 Order ID: {order_id}")
        logger.info(f"   📊 Executed Qty: {executed_qty:.6f}")
        logger.info(f"   💰 Fill Price: ${fill_price:.4f}")
        logger.info(f"   💸 Commission: ${commission:.4f}")
        
        total_cost = executed_qty * fill_price + commission
        execution_time = random.uniform(0.5, 2.0)
        
        logger.info(f"📈 [User {user_id}] TRADE COMPLETED:")
        logger.info(f"   ⏱️  Execution Time: {execution_time:.2f}s")
        logger.info(f"   💰 Total Cost: ${total_cost:.2f}")
        logger.info(f"   📊 Effective Price: ${fill_price:.4f}")
        logger.info(f"   💸 Fee %: {(commission/total_cost)*100:.3f}%")
        
        # Log comprehensive metrics
        metrics_logger.info(f"TRADE_EXECUTED | USER={user_id} | SYMBOL={symbol} | SIDE={signal} | QTY={executed_qty:.6f} | PRICE={fill_price:.4f} | VALUE=${total_cost:.2f} | FEE=${commission:.4f} | CONFIDENCE={confidence:.1f} | TEST={random.choice([True, False])} | DURATION={execution_time:.2f}s")
        
        logger.info(f"🎉 [User {user_id}] TRADE SUCCESS: {signal.upper()} {executed_qty:.6f} {symbol} @ ${fill_price:.4f}")
        
    else:
        # Simulate trade rejection
        reason = random.choice([
            "Insufficient balance", 
            "Daily limit exceeded", 
            "Risk threshold exceeded",
            "Market volatility too high"
        ])
        logger.warning(f"❌ [User {user_id}] TRADE REJECTED: {reason}")
        metrics_logger.info(f"TRADE_REJECTED | USER={user_id} | SYMBOL={symbol} | REASON={reason}")

async def main():
    """Run the trading simulation demo"""
    logger.info("🚀 TRADING BOT LOGGING DEMO STARTED")
    logger.info("🔧 Initializing Trading Bot services...")
    logger.info("✅ Binance service initialized")
    logger.info("✅ AI Analyzer ready")
    logger.info("✅ Risk Manager ready")
    logger.info("🌐 WebSocket Manager ready")
    logger.info("🤖 Trading Bot initialized successfully")
    metrics_logger.info("BOT_INITIALIZED | STATUS=SUCCESS")
    
    print("\n" + "="*80)
    print("🤖 TRADING BOT COMPREHENSIVE LOGGING DEMO")
    print("="*80)
    print("📝 Watch the console and check the logs/ directory for detailed output")
    print("📊 This demo shows what you'll see during actual trading")
    print("⏱️  Running 3 trading cycles...")
    print("="*80 + "\n")
    
    # Simulate WebSocket connections
    logger.info("🌐 WebSocket connected for user 1 | Total users: 1 | Total connections: 1")
    logger.info("🌐 WebSocket connected for user 2 | Total users: 2 | Total connections: 3")
    logger.info("🌐 WebSocket connected for user 3 | Total users: 3 | Total connections: 4")
    
    # Run a few trading cycles
    for i in range(3):
        await simulate_trading_cycle()
        if i < 2:  # Don't wait after the last cycle
            logger.debug("⏳ Waiting 60 seconds before next cycle...")
            await asyncio.sleep(1)  # Shortened for demo
    
    logger.info("🛑 Stopping Trading Bot...")
    logger.info("🛑 Trading Bot stopped")
    metrics_logger.info("BOT_STOPPED | STATUS=STOPPED")
    print("\n✅ Demo completed! Check the logs/ directory for detailed log files.")

if __name__ == "__main__":
    asyncio.run(main())