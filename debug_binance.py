#!/usr/bin/env python3
"""
Binance Connection Diagnostic Script
This will help identify exactly why Binance connection is failing
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.config import settings
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
import logging

async def main():
    print("=" * 60)
    print("🔍 BINANCE CONNECTION DIAGNOSTIC")
    print("=" * 60)
    
    # Check environment variables
    print("\n1. 🔍 Environment Variables:")
    print(f"   API Key: {settings.binance_api_key[:10]}...{settings.binance_api_key[-10:] if len(settings.binance_api_key) > 20 else 'TOO_SHORT'}")
    print(f"   Secret Key: {settings.binance_secret_key[:10]}...{settings.binance_secret_key[-10:] if len(settings.binance_secret_key) > 20 else 'TOO_SHORT'}")
    print(f"   Testnet Mode: {settings.binance_testnet}")
    print(f"   API Key Length: {len(settings.binance_api_key)}")
    print(f"   Secret Key Length: {len(settings.binance_secret_key)}")
    
    # Test different Binance endpoints step by step
    try:
        print("\n2. 📡 Creating Binance Client...")
        client = Client(
            api_key=settings.binance_api_key,
            api_secret=settings.binance_secret_key,
            testnet=settings.binance_testnet,
        )
        print("✅ Client created successfully")
        
        # Test 1: System Status (No auth needed)
        print("\n3. 🌐 Testing System Status (No Auth)...")
        try:
            status = client.get_system_status()
            print(f"✅ System Status: {status}")
        except Exception as e:
            print(f"❌ System Status Failed: {e}")
            print("   This suggests network/connectivity issues")
        
        # Test 2: Server Time (No auth needed)
        print("\n4. ⏰ Testing Server Time (No Auth)...")
        try:
            server_time = client.get_server_time()
            print(f"✅ Server Time: {server_time}")
        except Exception as e:
            print(f"❌ Server Time Failed: {e}")
            print("   This suggests network/connectivity issues")
        
        # Test 3: Exchange Info (No auth needed)
        print("\n5. 📊 Testing Exchange Info (No Auth)...")
        try:
            # Just get a small part to avoid too much output
            info = client.get_exchange_info()
            print(f"✅ Exchange Info: Got {len(info.get('symbols', []))} trading pairs")
        except Exception as e:
            print(f"❌ Exchange Info Failed: {e}")
        
        # Test 4: Account Info (Requires auth)
        print("\n6. 🔐 Testing Account Info (Requires Auth)...")
        try:
            account = client.get_account()
            print("✅ Account Info Retrieved:")
            print(f"   Account Type: {account.get('accountType', 'Unknown')}")
            print(f"   Can Trade: {account.get('canTrade', False)}")
            print(f"   Can Withdraw: {account.get('canWithdraw', False)}")
            print(f"   Can Deposit: {account.get('canDeposit', False)}")
            print(f"   Balances: {len(account.get('balances', []))} assets")
            
            # Show non-zero balances
            balances = account.get('balances', [])
            non_zero = [b for b in balances if float(b['free']) > 0 or float(b['locked']) > 0]
            if non_zero:
                print("   Non-zero balances:")
                for balance in non_zero[:5]:  # Show first 5
                    print(f"     {balance['asset']}: {balance['free']} (free), {balance['locked']} (locked)")
            else:
                print("   ⚠️  No balances found (this might be normal for testnet)")
                
        except BinanceAPIException as e:
            print(f"❌ Account Info Failed (API Error): {e}")
            print(f"   Error Code: {e.code}")
            print(f"   Error Message: {e.message}")
            
            if e.code == -2015:
                print("\n🔧 SOLUTION FOR ERROR -2015:")
                print("   1. Check API key is correct (no extra spaces)")
                print("   2. Check secret key is correct (no extra spaces)")
                print("   3. Verify API key permissions include:")
                print("      - Enable Reading ✅")
                print("      - Enable Spot & Margin Trading ✅")
                print("   4. Check if IP restriction is enabled")
                print("   5. Make sure keys are for the right environment (testnet vs mainnet)")
                
        except BinanceRequestException as e:
            print(f"❌ Account Info Failed (Request Error): {e}")
            print("   This suggests network connectivity issues")
            
        except Exception as e:
            print(f"❌ Account Info Failed (Unknown Error): {e}")
            print(f"   Error Type: {type(e)}")
        
        # Test 5: Try to get ticker (No auth needed)
        print("\n7. 💰 Testing Price Ticker (No Auth)...")
        try:
            ticker = client.get_symbol_ticker(symbol="BTCUSDT")
            print(f"✅ BTCUSDT Price: ${float(ticker['price']):,.2f}")
        except Exception as e:
            print(f"❌ Price Ticker Failed: {e}")
        
        # Test 6: Test order (if account works)
        print("\n8. 🧪 Testing Order Placement (Test Mode)...")
        try:
            # Try to place a test order
            test_order = client.create_test_order(
                symbol='BTCUSDT',
                side='BUY',
                type='MARKET',
                quantity=0.001
            )
            print("✅ Test Order: Successful (no actual trade)")
        except BinanceAPIException as e:
            print(f"❌ Test Order Failed (API Error): {e}")
            print(f"   Error Code: {e.code}")
            if e.code == -2010:
                print("   This means insufficient balance (normal for test)")
            elif e.code == -1013:
                print("   This means invalid quantity (normal)")
        except Exception as e:
            print(f"❌ Test Order Failed: {e}")
    
    except Exception as e:
        print(f"❌ Failed to create Binance client: {e}")
        print(f"   Error Type: {type(e)}")
    
    print("\n" + "=" * 60)
    print("📋 DIAGNOSTIC SUMMARY:")
    print("=" * 60)
    print("1. If System Status/Server Time fail → Network issue")
    print("2. If Account Info fails with -2015 → API key/permission issue")
    print("3. If Account Info works but shows canTrade=False → Permission issue")
    print("4. If Test Order fails with -2010/-1013 → Normal (insufficient balance)")
    print("5. If everything works → Check your .env file is being loaded correctly")
    
    print("\n💡 COMMON FIXES:")
    print("   1. Regenerate API keys on Binance")
    print("   2. Double-check .env file format (no quotes around values)")
    print("   3. Restart your FastAPI server after changing .env")
    print("   4. Check for extra spaces/newlines in API keys")
    print("   5. Verify you're using testnet keys for testnet=true")

if __name__ == "__main__":
    asyncio.run(main())