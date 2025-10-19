"""Shared configuration values for trading services."""

from __future__ import annotations

# ============================================================================
# DAY TRADING CONFIGURATION
# ============================================================================
# This system is optimized for intraday trading with adaptive holding periods
# Risk management follows the 1% rule with dynamic R:R based on market structure
# ============================================================================

# AI analyzer ratios - DAY TRADING (adaptive, these are starting points)
# The system will dynamically adjust based on support/resistance levels
AI_BUY_STOP_LOSS_RATIO = 0.985   # ~1.5% below entry (day trading range)
AI_BUY_TAKE_PROFIT_RATIO = 1.030  # ~3.0% above entry (1:2 R:R baseline)
AI_SELL_STOP_LOSS_RATIO = 1.015   # ~1.5% above entry
AI_SELL_TAKE_PROFIT_RATIO = 0.970  # ~3.0% below entry
AI_HOLD_STOP_LOSS_RATIO = 0.990
AI_HOLD_TAKE_PROFIT_RATIO = 1.015

# Risk management thresholds - DAY TRADING
MIN_SIGNAL_CONFIDENCE = 75  # High-quality setups only (was 60)
MIN_ACCOUNT_BALANCE = 10.0
MAX_BALANCE_TRADE_RATIO = 0.03  # Max 3% of balance per position (conservative)
MIN_TRADE_VALUE = 10.0
DEFAULT_RISK_PERCENTAGE = 1.0  # 1% rule: max risk per trade
DEFAULT_MAX_DAILY_TRADES = 5  # Quality over quantity (was 10)

# Trading cadence - DAY TRADING
ANALYSIS_RATE_LIMIT_SECONDS = 300  # 5 minutes between analyses (was 30s)
TRADING_CYCLE_DELAY_SECONDS = 300  # 5-minute cycles (was 60s)
POSITION_TIMEOUT_MINUTES = 480  # 8 hours max hold (was 60 min)

# Multi-timeframe analysis windows
MTF_PRIMARY_INTERVAL = "15m"    # Entry signals
MTF_CONTEXT_INTERVAL = "1h"     # Trend context
MTF_PRECISION_INTERVAL = "5m"   # Entry timing
MTF_PRIMARY_CANDLES = 100       # ~25 hours of 15m data
MTF_CONTEXT_CANDLES = 168       # ~1 week of 1h data
MTF_PRECISION_CANDLES = 288     # ~24 hours of 5m data

# Adaptive Risk-Reward settings
MIN_RISK_REWARD_RATIO = 1.5  # Minimum acceptable R:R (1:1.5)
TARGET_RISK_REWARD_RATIO = 2.0  # Target R:R for optimal setups (1:2)
MAX_RISK_REWARD_RATIO = 4.0  # Maximum to consider (beyond this, too ambitious)

# Support/Resistance detection settings
SR_LOOKBACK_PERIODS = 50  # Candles to analyze for S/R levels
SR_TOUCH_THRESHOLD = 0.002  # 0.2% proximity to consider a "touch"
SR_MIN_TOUCHES = 2  # Minimum touches to validate a level
SR_STRENGTH_DECAY = 0.9  # Decay factor for older levels

# Regime filter settings - DAY TRADING
REGIME_ADX_TRENDING_THRESHOLD = 20  # ADX > 20 = trending (was 25 for high-frequency trading)
REGIME_ADX_STRONG_TREND_THRESHOLD = 30  # ADX > 30 = strong trend
REGIME_ALLOW_RANGING = True  # Allow trades in range-bound markets
REGIME_MIN_CONFLUENCE_TRENDING = 70  # Minimum confluence for trending markets
REGIME_MIN_CONFLUENCE_RANGING = 75  # Higher bar for ranging markets

# Numerical safety helpers
EIGHT_DECIMAL_PLACES = 100_000_000
OCO_SELL_STOP_LIMIT_BUFFER = 0.999
OCO_BUY_STOP_LIMIT_BUFFER = 1.001

__all__ = [
    "AI_BUY_STOP_LOSS_RATIO",
    "AI_BUY_TAKE_PROFIT_RATIO",
    "AI_SELL_STOP_LOSS_RATIO",
    "AI_SELL_TAKE_PROFIT_RATIO",
    "AI_HOLD_STOP_LOSS_RATIO",
    "AI_HOLD_TAKE_PROFIT_RATIO",
    "MIN_SIGNAL_CONFIDENCE",
    "MIN_ACCOUNT_BALANCE",
    "MAX_BALANCE_TRADE_RATIO",
    "MIN_TRADE_VALUE",
    "DEFAULT_RISK_PERCENTAGE",
    "DEFAULT_MAX_DAILY_TRADES",
    "ANALYSIS_RATE_LIMIT_SECONDS",
    "TRADING_CYCLE_DELAY_SECONDS",
    "POSITION_TIMEOUT_MINUTES",
    "MTF_PRIMARY_INTERVAL",
    "MTF_CONTEXT_INTERVAL",
    "MTF_PRECISION_INTERVAL",
    "MTF_PRIMARY_CANDLES",
    "MTF_CONTEXT_CANDLES",
    "MTF_PRECISION_CANDLES",
    "MIN_RISK_REWARD_RATIO",
    "TARGET_RISK_REWARD_RATIO",
    "MAX_RISK_REWARD_RATIO",
    "SR_LOOKBACK_PERIODS",
    "SR_TOUCH_THRESHOLD",
    "SR_MIN_TOUCHES",
    "SR_STRENGTH_DECAY",
    "REGIME_ADX_TRENDING_THRESHOLD",
    "REGIME_ADX_STRONG_TREND_THRESHOLD",
    "REGIME_ALLOW_RANGING",
    "REGIME_MIN_CONFLUENCE_TRENDING",
    "REGIME_MIN_CONFLUENCE_RANGING",
    "EIGHT_DECIMAL_PLACES",
    "OCO_SELL_STOP_LIMIT_BUFFER",
    "OCO_BUY_STOP_LIMIT_BUFFER",
]
