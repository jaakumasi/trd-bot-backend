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

---

# 📋 COMPREHENSIVE TRADING WORKFLOW GUIDE

## 🎯 CRITICAL WORKFLOW DETAILS

### 🚀 **HOW TO START TRADING (STEP-BY-STEP)**

#### 1. **User Registration & Setup**
- User registers via `POST /auth/register`
- **AUTOMATIC**: Default `TradingConfig` is created with:
  - `is_active = False` (Trading is OFF by default)
  - `is_test_mode = True` (Safe testnet mode)
  - `risk_percentage = 1.0%` (1% risk per trade)
  - `trading_pair = "BTCUSDT"`
  - `max_daily_trades = 10`

#### 2. **MANDATORY: Manual Trading Activation**
⚠️ **IMPORTANT**: Trading does NOT start automatically after registration!

**Users MUST call the start trading endpoint:**
```bash
POST /trading/start
Authorization: Bearer <jwt_token>
```

This sets `is_active = True` in the user's `TradingConfig`.

#### 3. **Trading Bot Behavior**
- **The bot runs continuously** after server startup
- **Every 60 seconds**, it checks for users with `is_active = True`
- **Only active users** get processed in the trading loop
- **Inactive users are completely ignored**

### ⏰ **TRADING HOURS IMPACT**

#### **Fixed Trading Hours: 8AM - 4PM GMT**
```python
# Hard-coded in trading_bot.py
start_hour = 8   # 8:00 AM GMT
end_hour = 16    # 4:00 PM GMT (not inclusive)
```

#### **What Happens Outside Trading Hours:**
- ✅ **Bot keeps running** (doesn't stop)
- ✅ **WebSocket connections remain active**
- ❌ **NO market analysis is performed**
- ❌ **NO trades are executed**
- ⏳ **Bot checks every 60 seconds** if trading hours have resumed
- 📝 **Log message**: `"Outside trading hours (HH:MM:SS). Next check in 1 minute."`

#### **Trading Hours Resumption:**
- ✅ **Automatic**: When 8:00 AM GMT hits, trading resumes
- ✅ **All active users** are immediately processed
- ✅ **No manual intervention required**

### 📊 **COMPLETE TRADING CYCLE BREAKDOWN**

```
┌─────────────────────┐
│   SERVER STARTUP    │ 
│  trading_bot.start()│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    MAIN LOOP        │◄──┐
│   (Every 60 sec)    │   │
└──────────┬──────────┘   │
           │              │
           ▼              │
┌─────────────────────┐   │
│  Check Trading      │   │
│     Hours           │   │
│  (8AM-4PM GMT)      │   │
└──────────┬──────────┘   │
           │              │
    ┌──────┴──────┐       │
    │ Outside?    │──YES──┤
    └─────────────┘       │
           │NO             │
           ▼              │
┌─────────────────────┐   │
│ Query Active Users  │   │
│ is_active = True    │   │
└──────────┬──────────┘   │
           │              │
           ▼              │
┌─────────────────────┐   │
│ For Each Active     │   │
│      User:          │   │
│                     │   │
│ 1. Get Market Data  │   │
│ 2. AI Analysis      │   │
│ 3. Risk Validation  │   │
│ 4. Execute Trade    │   │
│ 5. WebSocket Update │   │
└──────────┬──────────┘   │
           │              │
           ▼              │
┌─────────────────────┐   │
│   Wait 60 seconds   │───┘
└─────────────────────┘
```

### 🔄 **USER CONTROL ENDPOINTS**

#### **Start Trading**
```bash
POST /trading/start
```
- Sets `is_active = True`
- User **immediately joins** the next trading cycle
- **Effect**: User will be processed in the next 60-second loop

#### **Stop Trading**  
```bash
POST /trading/stop
```
- Sets `is_active = False`
- User **immediately exits** trading cycles
- **Effect**: User is ignored in all subsequent loops until reactivated

#### **Update Configuration**
```bash
PUT /trading/config
{
  "risk_percentage": 2.0,
  "trading_pair": "ETHUSDT",
  "max_daily_trades": 15,
  "is_test_mode": false
}
```
- Changes take effect **immediately** in the next trading cycle
- **No restart required**

### 📈 **TRADING DECISION FLOW**

For each active user in each 60-second cycle:

```
1. ⏱️  RATE LIMITING CHECK
   └── Last analysis < 30 seconds ago? → SKIP USER

2. 📊 MARKET DATA FETCH  
   └── Get 1-minute candlestick data (100 periods)

3. 🤖 AI ANALYSIS
   └── Send to Gemini AI → Get BUY/SELL/HOLD signal

4. 📡 WEBSOCKET UPDATE
   └── Send AI analysis to user's frontend

5. 🎯 SIGNAL FILTERING
   └── Signal = HOLD? → END (no trade)

6. 💰 BALANCE CHECK
   └── Get user's USDT balance from Binance

7. 🛡️  RISK MANAGEMENT
   ├── Check daily trade limits
   ├── Validate position size  
   ├── Check AI confidence (>60%)
   └── PASS? → Continue | FAIL? → Reject trade

8. 🚀 TRADE EXECUTION
   ├── Place market order on Binance
   ├── Record trade in database
   ├── Update risk manager counters
   └── Send trade confirmation via WebSocket
```

### 🔄 **DAILY RESET BEHAVIOR**

**Midnight GMT Reset:**
- **Daily trade counters** reset to 0 for all users
- **Rate limiting timers** are cleared
- **Trading continues** automatically (no interruption)
- **Active users remain active** (no deactivation)

### ⚠️ **CRITICAL GOTCHAS**

1. **Trading is OFF by default** - Users must manually start it
2. **8AM-4PM GMT ONLY** - No trades outside these hours  
3. **60-second cycles** - Changes take up to 1 minute to take effect
4. **Rate limiting** - Max 1 analysis per user per 30 seconds
5. **Test mode default** - Set `is_test_mode = false` for real money
6. **WebSocket required** - Users need active connection for real-time updates
7. **Balance requirements** - Need sufficient USDT for calculated position sizes
8. **API permissions** - Binance API key must have SPOT trading enabled

### 🎯 **PRODUCTION CHECKLIST**

Before going live:
- [ ] Set `BINANCE_TESTNET=false` in environment
- [ ] Verify Binance API keys have SPOT trading permissions
- [ ] Set `is_test_mode=false` in user trading configs
- [ ] Ensure sufficient USDT balance in Binance account
- [ ] Test WebSocket connections from frontend
- [ ] Verify Gemini AI API key is working
- [ ] Check timezone settings (bot uses system time for GMT calculation)