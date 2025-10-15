# TRD Bot - AI-Powered Cryptocurrency Trading System
**Backend Documentation for New Developers**

---

## 🎯 System Overview

An automated cryptocurrency scalping bot that uses AI (Google Gemini) to analyze market data and execute trades on Binance. The system follows a conservative 1% risk-per-trade strategy with built-in safety mechanisms.

**Tech Stack:** FastAPI + SQLAlchemy + PostgreSQL + Binance API + Google Gemini AI + WebSockets

---

## 📊 Complete Trade Flow

### **Step-by-Step: How a Trade Happens**

```
1. User Registration (app/api/auth.py)
   ├── POST /auth/register → Creates User + TradingConfig (is_active=False by default)
   └── User receives JWT token

2. User Activates Trading (app/api/trading.py)
   ├── POST /trading/start → Sets is_active=True in TradingConfig
   └── User now joins the trading loop

3. Main Trading Loop (app/services/trading_bot.py)
   ├── Method: trading_loop() - Runs every 60 seconds
   ├── Method: _get_active_trading_configs() - Gets users with is_active=True
   └── For each active user:
       │
       ├─→ 4. Position Check
       │   ├── Method: _process_user_trading()
       │   ├── Check: Does user already have an open position?
       │   └── YES → Skip to next user | NO → Continue
       │
       ├─→ 5. Rate Limiting
       │   ├── Method: _remaining_rate_limit()
       │   ├── Check: Was user analyzed <30 seconds ago?
       │   └── YES → Skip to next user | NO → Continue
       │
       ├─→ 6. Market Data Fetch (app/services/binance_service.py)
       │   ├── Method: get_kline_data()
       │   └── Fetches 100 1-minute candlesticks from Binance
       │
       ├─→ 7. Historical Context Fetch (NEW - Phase 1)
       │   ├── File: app/services/ai_analyzer.py
       │   ├── Method: get_user_trade_history_context()
       │   └── Queries last 15 closed trades for performance metrics
       │
       ├─→ 8. AI Analysis (app/services/ai_analyzer.py)
       │   ├── Method: analyze_market_data()
       │   ├── Method: calculate_technical_indicators() - RSI, MACD, Bollinger Bands
       │   ├── Method: _generate_prompt() - Includes market data + user history
       │   ├── Sends to Gemini AI (gemini-2.5-flash model)
       │   └── Returns: {"signal": "buy/sell/hold", "confidence": 0-100, "entry_price", "stop_loss", "take_profit"}
       │
       ├─→ 9. Signal Decision
       │   ├── Signal = "hold"? → Skip to next user
       │   └── Signal = "buy" or "sell"? → Continue
       │
       ├─→ 10. Balance Check (app/services/binance_service.py)
       │   ├── Method: get_account_balance()
       │   └── Gets USDT available balance
       │
       ├─→ 11. Risk Validation (app/services/risk_manager.py)
       │   ├── Method: validate_trade_signal()
       │   ├── Checks:
       │   │   • User has no open positions (1 trade at a time rule)
       │   │   • Daily trade limit not exceeded
       │   │   • AI confidence ≥ 60%
       │   │   • Sufficient account balance (≥$10 USDT)
       │   │   • Position size valid
       │   │   • Trade value ≥$5 and ≤10% of balance
       │   └── PASS → Continue | FAIL → Reject + WebSocket notification
       │
       ├─→ 12. Trade Execution (app/services/trading_bot.py)
       │   ├── Method: _execute_trade()
       │   ├── Step 1: Place order on Binance
       │   │   ├── File: app/services/binance_service.py
       │   │   ├── Method: place_order_with_oco()
       │   │   ├── Creates market entry order (BUY/SELL)
       │   │   ├── Places OCO exit order (Take Profit + Stop Loss)
       │   │   └── OCO executes automatically on Binance servers
       │   ├── Step 2: Create Trade record (app/models/trade.py)
       │   │   └── Fields: symbol, side, amount, price, status='filled', ai_signal_confidence
       │   ├── Step 3: Create OpenPosition record (app/models/trade.py)
       │   │   └── Fields: trade_id, oco_order_id, entry_price, stop_loss, take_profit
       │   ├── Step 4: Add to RiskManager in-memory tracking
       │   │   └── Method: risk_manager.add_open_position()
       │   └── Step 5: Send WebSocket notification to user
       │
       └─→ 13. Wait 60 seconds → Loop back to Step 3
```

---

## 🔄 Order Placement & Tracking

### **OCO (One-Cancels-Other) Strategy**

Every trade uses OCO orders for automatic risk management:

**What Happens:**
1. **Entry Order** (Market Order)
   - Executes immediately at current price
   - File: `binance_service.py`, Method: `place_order_with_oco()`
   
2. **Exit Orders** (OCO - placed simultaneously)
   - **Take Profit**: Limit order at +0.3% from entry
   - **Stop Loss**: Stop-limit order at -0.5% from entry
   - **Automatic**: When one fills, the other cancels (handled by Binance)

**Tracking Lifecycle:**

```python
# app/models/trade.py - Trade model fields

# Entry Phase
status = 'pending'  → Order submitted
status = 'filled'   → Order executed

# Exit Phase (when OCO triggers)
status = 'closed'
exit_price = <fill price>
exit_reason = 'TAKE_PROFIT' | 'STOP_LOSS' | 'MANUAL' | 'TIMEOUT'
profit_loss = <net P&L after fees>
profit_loss_percentage = <P&L as % of entry value>
duration_seconds = <time position was open>
closed_at = <timestamp>
```

### **Position Monitoring**

**File:** `app/services/trading_bot.py`

**Method:** `_check_oco_orders()`
- Runs every 60 seconds
- Checks status of all open OCO orders via Binance API
- Method: `binance_service.get_oco_order_status()`
- When OCO status = 'ALL_DONE':
  - Extracts which leg filled (TP or SL)
  - Updates Trade record with exit details
  - Updates Portfolio with realized P&L
  - Removes position from RiskManager tracking
  - Deletes OpenPosition record

---

## 🛡️ Risk Management System

**File:** `app/services/risk_manager.py`

### **Core Rules**

1. **Single Trade Rule**
   - Method: `validate_trade_signal()` checks `get_open_positions(user_id)`
   - Only 1 open position per user at a time
   - New trades blocked until current position closes

2. **Position Sizing**
   - Method: `calculate_position_size()`
   - Formula: `position_size = (balance * risk_pct / 100) / |entry_price - stop_loss_price|`
   - Default: 1% account risk per trade
   - Max: 10% of balance per trade

3. **Daily Trade Limits**
   - Default: 10 trades per day per user
   - Tracked in-memory: `daily_trade_count[user_id]`
   - Method: `reset_daily_counters()` - Clears at midnight UTC

4. **Confidence Filter**
   - Minimum 60% AI confidence required
   - Constant: `MIN_SIGNAL_CONFIDENCE` in `service_constants.py`

5. **Balance Requirements**
   - Minimum: $10 USDT account balance
   - Minimum: $5 trade value
   - Constants: `MIN_ACCOUNT_BALANCE`, `MIN_TRADE_VALUE`

---

## ⏰ Manual Trade Closing & Timeouts

### **Automatic Position Timeout**

**File:** `app/services/trading_bot.py`

**Timeout Threshold:** 30 minutes (1800 seconds)
- Constant: `POSITION_TIMEOUT_MINUTES = 30`

**What Happens:**

```python
# Method: _check_oco_orders()

For each open position:
    1. Check: position open > 30 minutes?
    2. YES → Force close position:
       ├── Method: _force_close_stale_position()
       ├── Try to cancel OCO order:
       │   └── binance_service.cancel_oco_order() [uses _delete API]
       ├── Get current market price:
       │   └── binance_service.get_symbol_price()
       ├── Place market close order:
       │   └── binance_service.place_market_order()
       ├── Calculate P&L at current price
       ├── Update Trade record with exit details
       └── Update Portfolio statistics
```

### **Manual Close via API**

**File:** `app/api/trading.py`

**Endpoint:** `POST /trading/close/{trade_id}`
- User-initiated position close
- Same flow as timeout close
- Reason: `exit_reason = 'MANUAL'`

---

## 🕐 Trading Hours & Intervals

### **Operating Hours**
- **Active:** 8:00 AM - 4:00 PM GMT
- **Configuration:** `app/config.py` → `trading_active_hours_start` / `trading_active_hours_end`
- **Method:** `trading_bot._is_trading_hours()` - Checks before every cycle

**Outside Trading Hours:**
- ✅ Bot keeps running (doesn't stop)
- ❌ No market analysis performed
- ❌ No new trades executed
- ✅ Open positions continue to be monitored
- ✅ OCO orders still active on Binance

### **Timing Intervals**

| Action | Interval | File/Method |
|--------|----------|-------------|
| Main Trading Loop | **60 seconds** | `trading_bot.py` → `trading_loop()` |
| OCO Order Check | **60 seconds** | `trading_bot.py` → `_check_oco_orders()` |
| User Rate Limit | **30 seconds** | `trading_bot.py` → `_remaining_rate_limit()` |
| Position Timeout | **30 minutes** | `trading_bot.py` → `_check_oco_orders()` |
| Daily Counter Reset | **Midnight UTC** | `trading_bot.py` → `daily_reset_task()` |

---

## 🧠 AI Decision Making (Phase 1 Enhanced)

**File:** `app/services/ai_analyzer.py`

### **Historical Context Integration**

**Method:** `get_user_trade_history_context(db, user_id)`
- Queries last 15 closed trades from `trades` table
- Calculates:
  - Win rate (% profitable trades)
  - Average P&L per trade
  - Stop-loss hit rate
  - Take-profit hit rate
  - Recent trend (winning/losing/mixed based on last 5 trades)

### **AI Prompt Enhancement**

**Method:** `_generate_prompt(latest, market_summary, user_trade_history)`

**Includes:**
1. Market data (price, RSI, MACD, Bollinger Bands, volume)
2. **NEW:** User's historical performance summary
3. **NEW:** Critical instructions based on performance:
   - Low win rate (<40%) → Be very conservative, confidence >85% required
   - Losing streak → Reduce confidence by 15-20 points
   - High stop-loss rate (>60%) → Extra caution

**Result:** Personalized AI decisions that adapt to each user's trading success

---

## 📁 Key Files & Their Roles

### **Entry Point**
- `app/main.py` → FastAPI app initialization, WebSocket endpoint, bot startup/shutdown

### **Trading Core**
- `app/services/trading_bot.py` → Main trading loop, order execution, position monitoring
- `app/services/ai_analyzer.py` → Market analysis, AI signal generation, historical context
- `app/services/risk_manager.py` → Trade validation, position sizing, daily limits
- `app/services/binance_service.py` → Binance API integration, order placement, price fetching
- `app/services/mock_binance_service.py` → Simulated trading for testing (when `USE_MOCK_BINANCE=true`)
- `app/services/websocket_manager.py` → Real-time notifications to frontend

### **API Routes**
- `app/api/auth.py` → User registration, login, JWT tokens
- `app/api/trading.py` → Start/stop trading, config updates, trade history
- `app/api/portfolio.py` → Portfolio balance, performance stats

### **Database Models**
- `app/models/user.py` → User accounts
- `app/models/trade.py` → Trade records, TradingConfig, OpenPosition
- `app/models/portfolio.py` → Portfolio balances, performance metrics (win rate, total P&L)

### **Configuration**
- `app/config.py` → Environment variables, API keys, trading parameters
- `app/services/service_constants.py` → Hardcoded constants (timeouts, ratios, limits)

---

## 🚀 Quick Start for Development

### **1. Setup Environment**
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables in .env
BINANCE_API_KEY=your_key
BINANCE_SECRET_KEY=your_secret
BINANCE_TESTNET=true  # Safe mode
GEMINI_API_KEY=your_gemini_key
USE_MOCK_BINANCE=true  # For local testing without Binance
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/trd_bot
```

### **2. Run Backend**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### **3. Test Trade Flow**
```bash
# Register user
POST /auth/register {"username": "test", "email": "test@test.com", "password": "password"}

# Login
POST /auth/login {"username": "test", "password": "password"}

# Start trading (requires JWT token)
POST /trading/start
Authorization: Bearer <token>

# Watch logs for trading activity
tail -f logs/trading_bot_2025-10-15.log
```

---

## 🔍 Debugging Tips

### **Common Issues**

1. **"No active trading users found"**
   - Check: User called `POST /trading/start`?
   - Check: `TradingConfig.is_active = True` in database?

2. **"Outside trading hours"**
   - Check: Current time is 8AM-4PM GMT?
   - Verify: System timezone settings

3. **"User already has open position"**
   - Expected: Only 1 trade at a time allowed
   - Check: `OpenPosition` table for stuck positions

4. **"Trade rejected: insufficient balance"**
   - Check: USDT balance in Binance account
   - Check: Calculated position size vs. available balance

5. **"AI analyzer returned None"**
   - Check: Gemini API key valid?
   - Check: Gemini API rate limits

### **Useful Log Files**
- `logs/trading_bot_YYYY-MM-DD.log` → Main bot activity
- `logs/trades_YYYY-MM-DD.log` → Trade execution details
- `logs/metrics_YYYY-MM-DD.log` → Performance metrics

---

## 📊 Database Schema Quick Reference

```sql
-- Active user check
SELECT * FROM trading_configs WHERE is_active = true;

-- View open positions
SELECT * FROM open_positions;

-- Recent trade history
SELECT * FROM trades WHERE user_id = 1 ORDER BY executed_at DESC LIMIT 10;

-- User performance
SELECT 
    user_id,
    total_trades,
    winning_trades,
    losing_trades,
    win_rate,
    total_realized_pnl
FROM portfolio WHERE user_id = 1;
```

---

## 🎯 Architecture Highlights

**Design Patterns:**
- **Service Layer:** Business logic isolated in services (trading_bot, ai_analyzer, risk_manager)
- **Dependency Injection:** Services injected into TradingBot constructor
- **Mock Pattern:** `MockBinanceService` vs `BinanceService` for testing
- **Async/Await:** All database operations use SQLAlchemy AsyncSession

**Safety Features:**
- Default testnet mode (`BINANCE_TESTNET=true`)
- Single-trade-at-a-time constraint (Phase 1 implementation)
- Position timeout auto-close (30 minutes)
- Conservative risk management (1% account risk)
- Historical performance-aware AI (Phase 1 - reduces risk for losing users)

**Real-Time Communication:**
- WebSocket per user (`/ws/{user_id}`)
- Notifications: AI analysis, trade execution, errors
- Manager tracks connections: `WebSocketManager.active_connections`

---

## 📚 Further Reading

- `HISTORICAL_DATA_STRATEGY.md` - Original AI improvement plan
- `AI_IMPROVEMENT_STRATEGY.md` - Re-evaluated Phase 1 & 2 strategies
- `PHASE1_IMPLEMENTATION.md` - Detailed Phase 1 implementation notes
- `IMPLEMENTATION_SUMMARY.md` - Complete system analysis and changes
- `BUGFIX_GET_LATEST_PRICE.md` - Recent bug fix documentation

---

**Questions?** Check the inline code comments or trace the flow starting from `trading_bot.py → trading_loop()`