# TRD Bot - AI-Powered Cryptocurrency Trading System
**Backend Documentation for New Developers**

---

## 🎯 System Overview

An automated cryptocurrency **day trading bot** that uses AI (Google Gemini) to analyze market data and execute trades on Binance. The system follows a conservative **1% risk-per-trade strategy** with adaptive risk-reward based on support/resistance levels and multi-timeframe analysis.

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
   ├── Method: trading_loop() - Runs every 5 minutes (day trading cadence)
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
       │   ├── Check: Was user analyzed <5 minutes ago?
       │   └── YES → Skip to next user | NO → Continue
       │
       ├─→ 6. Multi-Timeframe Market Data Fetch (app/services/binance_service.py)
       │   ├── Method: _fetch_market_snapshot()
       │   ├── Fetches 3 timeframes for comprehensive analysis:
       │   │   • 15-minute (100 candles) - Primary entry signals
       │   │   • 1-hour (168 candles) - Trend context
       │   │   • 5-minute (288 candles) - Entry timing precision
       │   └── Returns DataFrames for each timeframe
       │
       ├─→ 7. Market Regime Analysis (app/services/market_regime_analyzer.py)
       │   ├── Method: classify_market_regime()
       │   ├── Analyzes: Trend strength (ADX), Volatility (ATR), Price action
       │   └── Returns: Regime classification + trading permission
       │
       ├─→ 8. Multi-Timeframe AI Analysis (app/services/ai_analyzer.py)
       │   ├── Method: analyze_market_data_mtf() - Multi-timeframe analysis
       │   ├── Method: calculate_technical_indicators() - RSI, MACD, ADX, ATR, Bollinger Bands
       │   ├── Method: _calculate_mtf_confluence() - Scores timeframe alignment (0-100)
       │   ├── Method: _generate_mtf_prompt() - Includes all timeframes + user history
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
       ├─→ 11. Risk Validation & Adaptive R:R (app/services/risk_manager.py)
       │   ├── Method: validate_trade_signal()
       │   ├── Method: detect_support_resistance_levels() - Identifies S/R zones
       │   ├── Method: calculate_adaptive_risk_reward() - Dynamic SL/TP placement
       │   ├── Checks:
       │   │   • User has no open positions (1 trade at a time rule)
       │   │   • Daily trade limit not exceeded (max 5/day)
       │   │   • AI confidence ≥ 75% (day trading quality bar)
       │   │   • Sufficient account balance (≥$10 USDT)
       │   │   • Position size follows 1% rule
       │   │   • Risk-reward ratio ≥ 1:1.5 (minimum acceptable)
       │   │   • Trade value ≥$10 and ≤3% of balance
       │   └── PASS → Continue | FAIL → Reject + WebSocket notification
       │
       ├─→ 12. Trade Execution (app/services/trading_bot.py)
       │   ├── Method: _execute_trade()
       │   ├── Step 1: Place order on Binance with Adaptive Exits
       │   │   ├── File: app/services/binance_service.py
       │   │   ├── Method: place_order_with_oco()
       │   │   ├── Creates market entry order (BUY/SELL)
       │   │   ├── Places OCO exit order with S/R-based levels:
       │   │   │   • Take Profit: Nearest resistance + ATR buffer (1:1.5 to 1:3 R:R)
       │   │   │   • Stop Loss: Nearest support - ATR buffer (0.8-2.5% range)
       │   │   └── OCO executes automatically on Binance servers
       │   ├── Step 2: Create Trade record (app/models/trade.py)
       │   │   └── Fields: symbol, side, amount, price, status='filled', ai_signal_confidence
       │   ├── Step 3: Create OpenPosition record (app/models/trade.py)
       │   │   └── Fields: trade_id, oco_order_id, entry_price, stop_loss, take_profit
       │   ├── Step 4: Add to RiskManager in-memory tracking
       │   │   └── Method: risk_manager.add_open_position()
       │   └── Step 5: Send WebSocket notification to user
       │
       └─→ 13. Wait 5 minutes → Loop back to Step 3
```

---

## 🔄 Order Placement & Tracking

### **OCO (One-Cancels-Other) Strategy**

Every trade uses OCO orders for automatic risk management:

**What Happens:**
1. **Entry Order** (Market Order)
   - Executes immediately at current price
   - File: `binance_service.py`, Method: `place_order_with_oco()`
   
2. **Exit Orders** (OCO - placed simultaneously with adaptive levels)
   - **Take Profit**: Limit order at nearest resistance + ATR buffer (1.5-5.0% range, 1:1.5-1:3 R:R)
   - **Stop Loss**: Stop-limit order at nearest support - ATR buffer (0.8-2.5% range)
   - **Adaptive**: Levels calculated based on actual market structure, not fixed percentages
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
- Runs every 5 minutes (aligned with trading cycle)
- Checks status of all open OCO orders via Binance API
- Method: `binance_service.get_oco_order_status()`
- Enforces 8-hour maximum hold time (day trading timeout)
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

1. **Multiple Position Management**
   - Method: `validate_trade_signal()` checks `get_open_positions(user_id)`
   - Maximum 3 concurrent open positions per user (configurable via `MAX_OPEN_POSITIONS`)
   - No duplicate positions on the same trading pair (symbol)
   - Each position can be on a different symbol (e.g., BTCUSDT, ETHUSDT, SOLUSDT)

2. **Position Sizing**
   - Method: `calculate_position_size()`
   - Formula: `position_size = (balance * risk_pct / 100) / |entry_price - stop_loss_price|`
   - Default: 1% account risk per trade
   - Max: 1% of balance per position (strict safety limit - see `MAX_BALANCE_TRADE_RATIO = 0.01`)

3. **Daily Trade Limits**
   - Default: 5 trades per day per user (quality over quantity)
   - Tracked in-memory: `daily_trade_count[user_id]`
   - Method: `reset_daily_counters()` - Clears at midnight UTC

4. **Confidence Filter (Adaptive)**
   - Base minimum: 60% AI confidence
   - **NEW:** Adaptive thresholds based on performance:
     - High win rate (>60%) → Threshold lowered by 10 (more aggressive)
     - Low win rate (<40%) → Threshold raised by 5 (more conservative)
     - Always randomized ±8% to prevent overfitting
   - Constant: `MIN_SIGNAL_CONFIDENCE = 60` in `service_constants.py`
   - Method: `risk_manager.get_adaptive_confidence_threshold()`

5. **Balance Requirements**
   - Minimum: $10 USDT account balance
   - Minimum: $5 trade value
   - Constants: `MIN_ACCOUNT_BALANCE`, `MIN_TRADE_VALUE`

6. **Anti-Overfitting Measures** 🆕
   - **Loosened thresholds:** S/R touch threshold: 0.5% (was 0.2%)
   - **Simplified confluence:** 3 core factors instead of 6 overlapping filters
   - **Parameter randomization:** All thresholds ±8-10% variance per trade
   - **Asset-specific profiles:** Different parameters for BTC/ETH/SOL/BNB
   - **Adaptive confidence:** Adjusts based on recent win rate
   - See: `OVERFITTING_ANALYSIS.md` and `ANTI_OVERFITTING_IMPLEMENTATION.md`


---

## ⏰ Manual Trade Closing & Timeouts

### **Automatic Position Timeout**

**File:** `app/services/trading_bot.py`

**Timeout Threshold:** 8 hours (480 minutes) - Day trading maximum hold
- Constant: `POSITION_TIMEOUT_MINUTES = 480`
- Prevents overnight exposure while allowing intraday moves to develop

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
- **Active:** 24/7 (00:00 - 23:59 GMT) - Crypto markets never close
- **Configuration:** `app/config.py` → `trading_active_hours_start` / `trading_active_hours_end`
- **Method:** `trading_bot._is_trading_hours()` - Checks before every cycle

**Day Trading Philosophy:**
- ✅ Continuous monitoring across all market sessions
- ✅ Captures opportunities in any timezone
- ✅ Open positions monitored round-the-clock
- ✅ 8-hour max hold prevents overnight exposure
- ✅ OCO orders always active on Binance

### **Timing Intervals**

| Action | Interval | File/Method |
|--------|----------|-------------|
| Main Trading Loop | **5 minutes** | `trading_bot.py` → `trading_loop()` |
| OCO Order Check | **5 minutes** | `trading_bot.py` → `_check_oco_orders()` |
| User Rate Limit | **5 minutes** | `trading_bot.py` → `_remaining_rate_limit()` |
| Position Timeout | **8 hours** | `trading_bot.py` → `_handle_oco_executing()` |
| Daily Counter Reset | **Midnight UTC** | `trading_bot.py` → `daily_reset_task()` |

---

## 🧠 Multi-Timeframe AI Decision Making

**File:** `app/services/ai_analyzer.py`

### **Multi-Timeframe Analysis**

**Method:** `analyze_market_data_mtf(symbol, mtf_data, user_trade_history, regime_analysis)`
- Analyzes 3 timeframes simultaneously:
  - **15-minute:** Primary entry signals and trade setup
  - **1-hour:** Overall trend direction and momentum
  - **5-minute:** Precise entry timing
- Calculates timeframe confluence score (0-100)
- Higher confluence = stronger, more reliable signals

### **Support/Resistance Detection**

**Method:** `risk_manager.detect_support_resistance_levels(df, current_price)`
- Identifies key price levels using:
  - Pivot points (swing highs/lows)
  - Price clustering analysis
  - Touch counting (validation of levels)
- Used for adaptive stop-loss and take-profit placement

### **Adaptive Risk-Reward Calculation**

**Method:** `risk_manager.calculate_adaptive_risk_reward(side, entry_price, sr_levels, atr)`
- Places stops at support levels (buys) or resistance levels (sells)
- Adds ATR-based buffer for volatility protection
- Targets 1:1.5 minimum, 1:2 preferred, up to 1:3 R:R
- Dynamically adjusts based on actual market structure

### **Day Trading Focus: Current Market Only**

**Historical Data: DISABLED**
- Day trading requires focus on **current market conditions**, not past performance
- Historical data can create psychological bias and hesitation
- Each trade is evaluated purely on technical setup quality
- Removes emotional baggage from previous wins/losses

**Why This Matters:**
- Prevents "revenge trading" after losses
- Prevents overconfidence after wins
- Ensures objective analysis of each setup
- AI focuses on market structure, not user psychology

### **Day Trading AI Prompt**

**Method:** `_generate_mtf_prompt(primary_latest, mtf_summary, user_history, regime_analysis)`

**Includes:**
1. Multi-timeframe market data (15m/1h/5m)
2. Technical indicators (RSI, MACD, ADX, ATR, Bollinger Bands, Stochastic, OBV)
3. Timeframe confluence score
4. Market regime classification (trending/ranging/volatile)
5. Day trading specific instructions:
   - Favor trending markets with ADX > 25
   - Accept range-bound trades if confluence is high
   - Respect 1:1.5 minimum risk-reward
   - Quality over quantity (1-5 trades/day target)

**Result:** Pure technical analysis focused on current market structure and multi-timeframe alignment, free from historical performance bias

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

#### **Per-Asset ATR Thresholds**
The system now supports **per-asset volatility calibration** for multi-symbol trading:

```python
# In service_constants.py
from app.services.service_constants import get_atr_thresholds

# Get thresholds for specific symbol
btc_thresholds = get_atr_thresholds("BTCUSDT")
# Returns: {"min": 0.40, "optimal_min": 0.50, "optimal_max": 1.20}

# Fallback to default for unconfigured assets
xrp_thresholds = get_atr_thresholds("XRPUSDT")
# Returns: BTCUSDT defaults
```

**Currently Configured Assets:**
- `BTCUSDT`: min=0.40%, optimal=0.50-1.20% (BTC volatility)
- `ETHUSDT`: min=0.50%, optimal=0.60-1.50% (Medium volatility)
- `BNBUSDT`: min=0.50%, optimal=0.60-1.40%
- `ADAUSDT`: min=0.60%, optimal=0.80-2.00% (Altcoin volatility)
- `SOLUSDT`: min=0.80%, optimal=1.00-2.50% (High volatility)
- `DOTUSDT`: min=0.60%, optimal=0.80-2.00%

**To Add New Assets:**
1. Edit `PER_ASSET_ATR_THRESHOLDS` dict in `service_constants.py`
2. Run `validate_constants()` at startup to verify ranges
3. Monitor logs for: `"Configured assets: BTCUSDT, ETHUSDT, ..."`

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

**Day Trading Safety Features:**
- Default testnet mode (`BINANCE_TESTNET=true`)
- Single-trade-at-a-time constraint
- 8-hour position timeout (prevents overnight exposure)
- 1% rule position sizing (risk-based, not capital-based)
- Adaptive stop-loss placement (support/resistance + ATR buffer)
- Minimum 1:1.5 risk-reward ratio enforcement
- Multi-timeframe confluence validation
- Pure technical analysis (no historical performance bias)
- Quality-focused trade selection (1-5 trades/day, not 10+)

**Real-Time Communication:**
- WebSocket per user (`/ws/{user_id}`)
- Notifications: AI analysis, trade execution, errors
- Manager tracks connections: `WebSocketManager.active_connections`

---

## 📚 Further Reading

- `DAY_TRADING_COMPLETE_TRANSFORMATION.md` - **Complete transformation from scalping to day trading**
- `HISTORICAL_DATA_STRATEGY.md` - Original AI improvement plan
- `AI_IMPROVEMENT_STRATEGY.md` - Re-evaluated Phase 1 & 2 strategies
- `IMPLEMENTATION_SUMMARY.md` - Complete system analysis and changes
- `BUGFIX_GET_LATEST_PRICE.md` - Recent bug fix documentation

---

**Questions?** Check the inline code comments or trace the flow starting from `trading_bot.py → trading_loop()`