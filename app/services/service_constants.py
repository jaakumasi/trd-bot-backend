"""Shared configuration values for trading services."""

from __future__ import annotations
import random

# ============================================================================
# ANTI-OVERFITTING UTILITIES
# ============================================================================

def get_randomized_threshold(base_value: float, variance_pct: float = 0.10) -> float:
    """
    Add randomization to thresholds to prevent point-estimate overfitting.
    
    Args:
        base_value: The base threshold value
        variance_pct: Percentage variance (default 10%)
    
    Returns:
        Randomized value within ±variance_pct of base
    
    Example:
        get_randomized_threshold(75, 0.10) -> returns 67.5 to 82.5
    """
    lower = base_value * (1 - variance_pct)
    upper = base_value * (1 + variance_pct)
    return random.uniform(lower, upper)


# ============================================================================
# ASSET-SPECIFIC PARAMETERS (Prevents Bitcoin-only overfitting)
# ============================================================================

ASSET_PARAMS = {
    'BTCUSDT': {
        'name': 'Bitcoin',
        'atr_multiplier': 1.5,
        'min_confidence': 60,
        'volume_threshold_multiplier': 1.0,
        'sr_touch_threshold': 0.005,  # 0.5%
        'optimal_hold_minutes': (15, 30),
        'description': 'Lower volatility, most liquid'
    },
    'ETHUSDT': {
        'name': 'Ethereum',
        'atr_multiplier': 2.0,
        'min_confidence': 58,
        'volume_threshold_multiplier': 0.9,
        'sr_touch_threshold': 0.006,  # 0.6% (more volatile)
        'optimal_hold_minutes': (10, 20),
        'description': 'Higher volatility than BTC, fast moves'
    },
    'SOLUSDT': {
        'name': 'Solana',
        'atr_multiplier': 3.0,
        'min_confidence': 55,
        'volume_threshold_multiplier': 0.8,
        'sr_touch_threshold': 0.008,  # 0.8% (highly volatile)
        'optimal_hold_minutes': (5, 15),
        'description': 'High volatility altcoin, quick trades'
    },
    'BNBUSDT': {
        'name': 'Binance Coin',
        'atr_multiplier': 2.5,
        'min_confidence': 57,
        'volume_threshold_multiplier': 0.85,
        'sr_touch_threshold': 0.007,  # 0.7%
        'optimal_hold_minutes': (8, 18),
        'description': 'Exchange token, moderate volatility'
    },
    'DEFAULT': {
        'name': 'Generic Asset',
        'atr_multiplier': 2.0,
        'min_confidence': 60,
        'volume_threshold_multiplier': 1.0,
        'sr_touch_threshold': 0.006,
        'optimal_hold_minutes': (10, 25),
        'description': 'Fallback for unknown assets'
    }
}


def get_asset_params(symbol: str) -> dict:
    """
    Get asset-specific parameters to prevent overfitting to Bitcoin.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT', 'ETHUSDT')
    
    Returns:
        Dictionary of asset-specific parameters
    """
    return ASSET_PARAMS.get(symbol, ASSET_PARAMS['DEFAULT'])


# ============================================================================
# DAY TRADING CONFIGURATION
# ============================================================================
# This system is optimized for intraday trading with adaptive holding periods
# Risk management follows the 1% rule with dynamic R:R based on market structure
# ============================================================================

# AI analyzer ratios - DAY TRADING with 2:1 minimum R:R (mainnet ready)
# Updated for sustainable edge after fees and slippage
AI_BUY_STOP_LOSS_RATIO = 0.993   # ~0.7% below entry
AI_BUY_TAKE_PROFIT_RATIO = 1.014  # ~1.4% above entry (2:1 R:R baseline)
AI_SELL_STOP_LOSS_RATIO = 1.007   # ~0.7% above entry
AI_SELL_TAKE_PROFIT_RATIO = 0.986  # ~1.4% below entry
AI_HOLD_STOP_LOSS_RATIO = 0.993
AI_HOLD_TAKE_PROFIT_RATIO = 1.014

# Risk management thresholds - DAY TRADING
MIN_SIGNAL_CONFIDENCE = 70
MIN_ACCOUNT_BALANCE = 10.0
MAX_BALANCE_TRADE_RATIO = 0.03  # Max 3% of balance per position (conservative mainnet)
MIN_TRADE_VALUE = 10.0
DEFAULT_RISK_PERCENTAGE = 1.0  # 1% rule: max risk per trade
DEFAULT_MAX_DAILY_TRADES = 5  # Quality over quantity (was 10)
MAX_OPEN_POSITIONS = 3  # Maximum concurrent open positions per user

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

# Adaptive Risk-Reward settings (mainnet requirements)
MIN_RISK_REWARD_RATIO = 2.0  # Minimum 2:1 R:R for profitable trading after fees
TARGET_RISK_REWARD_RATIO = 2.5  # Target R:R for optimal setups (1:2.5)
MAX_RISK_REWARD_RATIO = 4.0  # Maximum to consider (beyond this, too ambitious)

# Support/Resistance detection settings
SR_LOOKBACK_PERIODS = 50  # Candles to analyze for S/R levels
SR_TOUCH_THRESHOLD = 0.005  # 0.5% proximity to consider a "touch" (loosened from 0.2% to reduce overfitting)
SR_MIN_TOUCHES = 2  # Minimum touches to validate a level
SR_STRENGTH_DECAY = 0.9  # Decay factor for older levels

# Regime filter settings - DAY TRADING
REGIME_ADX_TRENDING_THRESHOLD = 20  # ADX > 20 = trending (was 25 for high-frequency trading)
REGIME_ADX_STRONG_TREND_THRESHOLD = 30  # ADX > 30 = strong trend
REGIME_ALLOW_RANGING = True  # Allow trades in range-bound markets
REGIME_MIN_CONFLUENCE_TRENDING = 55  # Reduced from 60 for more opportunities
REGIME_MIN_CONFLUENCE_RANGING = 55  # Unified threshold for all regimes (reduced from 60)

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
    "MAX_OPEN_POSITIONS",
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
    "get_randomized_threshold",
    "ASSET_PARAMS",
    "get_asset_params",
]
