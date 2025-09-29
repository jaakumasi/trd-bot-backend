# TRD Bot Trading System - Quick Reference Guide

## 🚀 MAIN ENTRY POINT
**File**: `app/main.py`
- `startup_event()` - Initializes database and starts trading bot
- `trading_bot = TradingBot(ws_manager)` - Creates bot instance
- `shutdown_event()` - Stops bot gracefully
- **WebSocket**: `/ws/{user_id}` - Real-time updates to users

## 🤖 CORE TRADING ENGINE
**File**: `app/services/trading_bot.py`
- `trading_loop()` - **MAIN LOOP**: Runs every 60 seconds, processes all active users
- `_process_user_trading()` - **PER-USER LOGIC**: Gets market data → AI analysis → risk check → execute
- `_execute_trade()` - **TRADE EXECUTION**: Places order on Binance + records in database
- `_get_active_trading_configs()` - Gets users with `is_active = True`
- `_is_trading_hours()` - Only trades 8AM-4PM GMT

## 🧠 AI ANALYSIS (GEMINI INTEGRATION)
**File**: `app/services/ai_analyzer.py`
- `analyze_market_data()` - **MAIN AI FUNCTION**: Sends market data to Gemini, gets buy/sell/hold signal
- `calculate_technical_indicators()` - **TECHNICAL ANALYSIS**: RSI, MACD, Bollinger Bands, Moving Averages
- `_calculate_technical_score()` - **VALIDATION**: Confirms AI signal with technical analysis
- **Gemini Model**: `genai.GenerativeModel("gemini-pro")` - Processes market prompt, returns JSON signal

## 💹 BINANCE INTEGRATION
**File**: `app/services/binance_service.py`
- `initialize()` - **CONNECTION**: Tests API keys and connection
- `get_kline_data()` - **MARKET DATA**: Gets 1-minute candlestick data for analysis
- `get_account_balance()` - **BALANCE CHECK**: Gets USDT/BTC balances
- `place_market_order()` - **TRADE EXECUTION**: Places real/test orders
- `get_symbol_price()` - **CURRENT PRICE**: Gets live price for calculations

## 🛡️ RISK MANAGEMENT
**File**: `app/services/risk_manager.py`
- `validate_trade_signal()` - **SAFETY GATE**: Validates every trade before execution
- `calculate_position_size()` - **POSITION SIZING**: Risk-based position calculation (1% account risk)
- `record_trade()` - **DAILY LIMITS**: Tracks daily trade count per user
- `reset_daily_counters()` - **MIDNIGHT RESET**: Resets daily counters at midnight

## 📡 API ENDPOINTS
**File**: `app/api/trading.py`
- `POST /trading/start` - **ACTIVATE BOT**: Sets user's `is_active = True`
- `POST /trading/stop` - **DEACTIVATE BOT**: Sets user's `is_active = False`
- `GET /trading/trades` - **TRADE HISTORY**: Returns user's past trades
- `PUT /trading/config` - **UPDATE SETTINGS**: Modify risk%, trading pair, etc.

## 🔐 AUTHENTICATION
**File**: `app/api/auth.py`
- `POST /auth/login` - **USER LOGIN**: JWT token generation
- `POST /auth/register` - **USER SIGNUP**: Creates user + default trading config
- `get_current_user()` - **AUTH GUARD**: Validates JWT tokens for protected routes

## ⚙️ CONFIGURATION
**File**: `app/config.py`
- **Binance API**: `binance_api_key`, `binance_secret_key`, `binance_testnet`
- **Gemini AI**: `gemini_api_key`
- **Trading**: `default_risk_percentage`, `trading_active_hours_start/end`
- **Security**: `jwt_secret`, `jwt_algorithm`

## 🗄️ DATABASE MODELS
**File**: `app/models/`
- `user.py` - **User**: User accounts and authentication
- `trade.py` - **Trade**: Individual trade records + **TradingConfig**: User trading settings
- `portfolio.py` - **Portfolio**: User portfolio tracking

## 📊 TRADING FLOW SEQUENCE
```
1. FastAPI starts → trading_bot.start() 
2. Every 60 seconds → trading_loop()
3. Get active users → _get_active_trading_configs()
4. For each user → _process_user_trading():
   a. Get market data → binance.get_kline_data()
   b. AI analysis → ai_analyzer.analyze_market_data()
   c. Risk validation → risk_manager.validate_trade_signal()
   d. Execute trade → _execute_trade() → binance.place_market_order()
   e. Record trade → database + WebSocket notification
```

## 🔑 KEY REQUIREMENTS FOR LIVE TRADING
- **Binance API Key**: Must have SPOT TRADING permissions (not just read-only)
- **Gemini API Key**: For AI analysis
- **Environment**: Set `BINANCE_TESTNET=false` for real trading
- **Minimum Balance**: $10 USDT to start trading
- **User Action**: Call `POST /trading/start` to activate bot for user

## ⏰ TIMING & LIMITS
- **Analysis Frequency**: Every 60 seconds
- **Rate Limiting**: Max 1 analysis per user every 30 seconds  
- **Trading Hours**: 8AM - 4PM GMT only
- **Daily Trade Limit**: Configurable per user (default: 10 trades/day)
- **Risk Per Trade**: 1% of account balance (configurable)

## 🚨 SAFETY FEATURES
- **Test Mode**: Default testnet mode (no real money)
- **Confidence Filter**: Minimum 60% AI confidence required
- **Position Limits**: Max 10% of balance per trade
- **Daily Limits**: Prevents overtrading
- **Stop Loss**: Built into position sizing calculation