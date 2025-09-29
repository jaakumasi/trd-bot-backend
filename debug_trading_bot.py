#!/usr/bin/env python3
"""
Trading Bot Troubleshooting Script
Run this to diagnose why you're not seeing trading bot logs
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.logging_config import setup_logging
from app.config import settings
from app.services.binance_service import BinanceService
from app.services.trading_bot import TradingBot
from app.services.websocket_manager import WebSocketManager
from datetime import datetime, time
import logging

async def main():
    print("=" * 60)
    print("🔍 TRADING BOT TROUBLESHOOTING DIAGNOSTICS")
    print("=" * 60)
    
    # Setup logging
    print("\n1. 🔧 Setting up logging...")
    try:
        metrics_logger = setup_logging()
        logger = logging.getLogger(__name__)
        print("✅ Logging system initialized")
    except Exception as e:
        print(f"❌ Logging setup failed: {e}")
        return
    
    # Check environment variables
    print("\n2. 🔍 Checking environment variables...")
    print(f"   BINANCE_API_KEY: {'✅ Set' if settings.binance_api_key else '❌ Missing'}")
    print(f"   BINANCE_SECRET_KEY: {'✅ Set' if settings.binance_secret_key else '❌ Missing'}")
    print(f"   GEMINI_API_KEY: {'✅ Set' if settings.gemini_api_key else '❌ Missing'}")
    print(f"   BINANCE_TESTNET: {settings.binance_testnet}")
    print(f"   DATABASE_URL: {'✅ Set' if settings.database_url else '❌ Missing'}")
    
    # Test Binance connection
    print("\n3. 📡 Testing Binance connection...")
    try:
        binance = BinanceService()
        connected = await binance.initialize()
        if connected:
            print("✅ Binance connection successful")
            
            # Test account balance
            try:
                balances = binance.get_account_balance()
                usdt_balance = balances.get("USDT", {}).get("free", 0)
                print(f"   💰 USDT Balance: ${usdt_balance}")
            except Exception as e:
                print(f"   ⚠️  Could not get balance: {e}")
        else:
            print("❌ Binance connection failed")
            return
    except Exception as e:
        print(f"❌ Binance connection error: {e}")
        return
    
    # Check trading hours
    print("\n4. ⏰ Checking trading hours...")
    current_time = datetime.now().time()
    start_hour = 8
    end_hour = 16
    is_trading_hours = start_hour <= current_time.hour < end_hour
    
    print(f"   Current time: {current_time.strftime('%H:%M:%S')}")
    print(f"   Trading hours: {start_hour:02d}:00 - {end_hour:02d}:00 GMT")
    print(f"   In trading hours: {'✅ Yes' if is_trading_hours else '❌ No'}")
    
    if not is_trading_hours:
        print("   ⚠️  You're outside trading hours - bot won't process trades!")
        print(f"   ⏰ Next trading session starts at {start_hour:02d}:00 GMT")
    
    # Test trading bot initialization
    print("\n5. 🤖 Testing trading bot initialization...")
    try:
        ws_manager = WebSocketManager()
        trading_bot = TradingBot(ws_manager)
        
        initialized = await trading_bot.initialize()
        if initialized:
            print("✅ Trading bot initialized successfully")
            
            # Test if bot can start
            started = await trading_bot.start()
            if started:
                print("✅ Trading bot started successfully")
                print("   🔄 Bot should now be running the trading loop")
                
                # Wait a bit and check status
                print("\n6. 👀 Watching for trading loop activity...")
                print("   (Waiting 10 seconds to see if trading loop runs...)")
                await asyncio.sleep(10)
                
                if trading_bot.is_running:
                    print("✅ Trading bot is still running")
                else:
                    print("❌ Trading bot stopped unexpectedly")
                
                # Stop the bot
                await trading_bot.stop()
                print("🛑 Trading bot stopped for testing")
                
            else:
                print("❌ Trading bot failed to start")
        else:
            print("❌ Trading bot initialization failed")
            
    except Exception as e:
        print(f"❌ Trading bot test error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("📋 TROUBLESHOOTING CHECKLIST:")
    print("=" * 60)
    print("1. ✅ Check all environment variables are set correctly")
    print("2. ✅ Verify Binance API keys have SPOT trading permissions")
    print("3. ✅ Ensure you're in trading hours (8AM-4PM GMT)")
    print("4. ✅ Confirm bot initialization doesn't fail")
    print("5. ✅ Check if you have any active trading configurations")
    print("6. ✅ Verify you called POST /trading/start after login")
    print("\n💡 NEXT STEPS:")
    print("   1. Fix any ❌ issues shown above")
    print("   2. Restart your FastAPI server")
    print("   3. Check GET /debug/bot-status endpoint")
    print("   4. Look for 🔄 trading cycle logs every 60 seconds")

if __name__ == "__main__":
    asyncio.run(main())