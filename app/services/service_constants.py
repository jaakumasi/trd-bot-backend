"""Shared configuration values for trading services."""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

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
# Updated for realistic stops based on ATR (1.5% stop, 3.0% target)
AI_BUY_STOP_LOSS_RATIO = 0.985   # ~1.5% below entry (1.5x ATR for BTC)
AI_BUY_TAKE_PROFIT_RATIO = 1.030  # ~3.0% above entry (2:1 R:R after fees)
AI_SELL_STOP_LOSS_RATIO = 1.015   # ~1.5% above entry
AI_SELL_TAKE_PROFIT_RATIO = 0.970  # ~3.0% below entry
AI_HOLD_STOP_LOSS_RATIO = 0.985
AI_HOLD_TAKE_PROFIT_RATIO = 1.030

# Risk management thresholds - DAY TRADING
MIN_SIGNAL_CONFIDENCE = 75
MIN_TRADING_QUALITY = 80  # Minimum combined setup quality score (0-100)
MIN_ACCOUNT_BALANCE = 10.0
MAX_BALANCE_TRADE_RATIO = 0.01  # Max 1% of balance per position
MIN_TRADE_VALUE = 10.0
DEFAULT_RISK_PERCENTAGE = 1.0  # 1% rule: max risk per trade
DEFAULT_MAX_DAILY_TRADES = 3  # Quality over quantity
MAX_OPEN_POSITIONS = 3  # Maximum concurrent open positions per user

# Per-asset ATR thresholds - DAY TRADING (calibrated for each symbol's volatility)
# Format: {symbol: {"min": min_atr_%, "optimal_min": optimal_low, "optimal_max": optimal_high}}
PER_ASSET_ATR_THRESHOLDS = {
    "BTCUSDT": {"min": 0.40, "optimal_min": 0.50, "optimal_max": 1.20},  # Current default
    "ETHUSDT": {"min": 0.50, "optimal_min": 0.60, "optimal_max": 1.50},  # Slightly more volatile
    "BNBUSDT": {"min": 0.50, "optimal_min": 0.60, "optimal_max": 1.40},
    "ADAUSDT": {"min": 0.60, "optimal_min": 0.80, "optimal_max": 2.00},  # Altcoin volatility
    "SOLUSDT": {"min": 0.80, "optimal_min": 1.00, "optimal_max": 2.50},  # High volatility
    "DOTUSDT": {"min": 0.60, "optimal_min": 0.80, "optimal_max": 2.00},
}

# Legacy fallback constants (used if symbol not in PER_ASSET_ATR_THRESHOLDS)
MIN_ATR_PERCENTAGE_FOR_ENTRY = 0.40  # Default minimum if asset not configured
OPTIMAL_ATR_PERCENTAGE_RANGE = (0.50, 1.20)  # Default optimal range

def get_atr_thresholds(symbol: str) -> dict:
    """
    Get ATR thresholds for a specific symbol.
    Falls back to default BTCUSDT thresholds if symbol not configured.
    
    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
    
    Returns:
        dict: {"min": float, "optimal_min": float, "optimal_max": float}
    """
    return PER_ASSET_ATR_THRESHOLDS.get(
        symbol,
        {
            "min": MIN_ATR_PERCENTAGE_FOR_ENTRY,
            "optimal_min": OPTIMAL_ATR_PERCENTAGE_RANGE[0],
            "optimal_max": OPTIMAL_ATR_PERCENTAGE_RANGE[1]
        }
    )

# Trading cadence - DAY TRADING
ANALYSIS_RATE_LIMIT_SECONDS = 300  # 5 minutes between analyses 
TRADING_CYCLE_DELAY_SECONDS = 300  # 5-minute cycles
POSITION_TIMEOUT_MINUTES = 480  # 8 hours max hold 

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

# Structure-based stop loss settings (Phase 3 enhancement)
MIN_STOP_DISTANCE_PCT = 0.005  # 0.5% minimum stop distance (prevents too-tight stops)
MAX_STOP_DISTANCE_PCT = 0.03   # 3.0% maximum stop distance (prevents insane stops)
STRUCTURE_STOP_ATR_BUFFER = 1.0  # 1.0x ATR buffer beyond S/R level (prevents premature stops on wicks)

# Support/Resistance detection settings
SR_LOOKBACK_PERIODS = 50  # Candles to analyze for S/R levels
SR_TOUCH_THRESHOLD = 0.005  # 0.5% proximity to consider a "touch" (loosened from 0.2% to reduce overfitting)
SR_MIN_TOUCHES = 2  # Minimum touches to validate a level
SR_STRENGTH_DECAY = 0.9  # Decay factor for older levels

# Regime filter settings - DAY TRADING
# ADJUSTED: Raised thresholds to reduce false RANGE_BOUND classifications
REGIME_ADX_TRENDING_THRESHOLD = 25  # ADX > 25 = trending
REGIME_ADX_STRONG_TREND_THRESHOLD = 45  # ADX > 45 = strong trend
REGIME_ALLOW_RANGING = True  # Allow trades in range-bound markets
REGIME_MIN_CONFLUENCE_TRENDING = 55  # Reduced from 60 for more opportunities
REGIME_MIN_CONFLUENCE_RANGING = 60  # Higher threshold to avoid false ranges

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
    "MIN_TRADING_QUALITY",
    "MIN_ACCOUNT_BALANCE",
    "MAX_BALANCE_TRADE_RATIO",
    "MIN_TRADE_VALUE",
    "DEFAULT_RISK_PERCENTAGE",
    "DEFAULT_MAX_DAILY_TRADES",
    "MAX_OPEN_POSITIONS",
    "PER_ASSET_ATR_THRESHOLDS",
    "MIN_ATR_PERCENTAGE_FOR_ENTRY",
    "OPTIMAL_ATR_PERCENTAGE_RANGE",
    "get_atr_thresholds",
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
    "MIN_STOP_DISTANCE_PCT",
    "MAX_STOP_DISTANCE_PCT",
    "STRUCTURE_STOP_ATR_BUFFER",
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
    "ASSET_PARAMS",
    "get_asset_params",
    "validate_constants",
]


def validate_constants() -> bool:
    """
    Validate trading constants for unit consistency and sensible ranges.
    Called at app startup to catch configuration errors early.
    
    Returns:
        bool: True if all validations pass
    """
    issues = []
    
    # Check fractional thresholds (should be 0-1 range)
    if not (0 < SR_TOUCH_THRESHOLD < 1):
        issues.append(f"SR_TOUCH_THRESHOLD={SR_TOUCH_THRESHOLD} should be fractional (0-1), not percentage")
    
    # Check percentage thresholds (should be 0-100 range)
    if not (0 < MIN_ATR_PERCENTAGE_FOR_ENTRY < 10):
        issues.append(f"MIN_ATR_PERCENTAGE_FOR_ENTRY={MIN_ATR_PERCENTAGE_FOR_ENTRY} should be small percentage (0-10)")
    
    # Validate per-asset ATR thresholds
    for symbol, thresholds in PER_ASSET_ATR_THRESHOLDS.items():
        if not all(key in thresholds for key in ["min", "optimal_min", "optimal_max"]):
            issues.append(f"Asset {symbol} missing required ATR threshold keys")
        elif not (0 < thresholds["min"] < thresholds["optimal_min"] < thresholds["optimal_max"] < 10):
            issues.append(
                f"Asset {symbol} ATR thresholds invalid: "
                f"min={thresholds['min']}, optimal=({thresholds['optimal_min']}, {thresholds['optimal_max']})"
            )
    
    # Check R:R ratios are sensible
    if not (1.0 <= MIN_RISK_REWARD_RATIO <= MAX_RISK_REWARD_RATIO):
        issues.append(f"R:R ratios invalid: MIN={MIN_RISK_REWARD_RATIO}, MAX={MAX_RISK_REWARD_RATIO}")
    
    # Check position sizing cap is reasonable
    if not (0 < MAX_BALANCE_TRADE_RATIO <= 0.05):
        issues.append(f"MAX_BALANCE_TRADE_RATIO={MAX_BALANCE_TRADE_RATIO} should be 0-5% (0-0.05)")
    
    # Check risk percentage is reasonable
    if not (0 < DEFAULT_RISK_PERCENTAGE <= 5.0):
        issues.append(f"DEFAULT_RISK_PERCENTAGE={DEFAULT_RISK_PERCENTAGE} should be 0-5%")
    
    # Verify AI ratios produce intended stop/take percentages
    buy_sl_pct = (1 - AI_BUY_STOP_LOSS_RATIO) * 100
    buy_tp_pct = (AI_BUY_TAKE_PROFIT_RATIO - 1) * 100
    effective_rr = buy_tp_pct / buy_sl_pct if buy_sl_pct > 0 else 0
    
    if effective_rr < MIN_RISK_REWARD_RATIO * 0.9:
        issues.append(
            f"AI ratios produce R:R={effective_rr:.2f} which is below MIN_RISK_REWARD_RATIO={MIN_RISK_REWARD_RATIO}"
        )
    
    # Log results
    if issues:
        logger.error("❌ CONSTANT VALIDATION FAILED:")
        for issue in issues:
            logger.error(f"   - {issue}")
        return False
    
    logger.info("✅ Trading constants validation passed")
    logger.info(f"   📊 Effective R:R from AI ratios: {effective_rr:.2f}:1")
    logger.info(f"   📏 MAX_BALANCE_TRADE_RATIO: {MAX_BALANCE_TRADE_RATIO*100:.1f}%")
    logger.info(f"   🎯 MIN_ATR_PERCENTAGE_FOR_ENTRY: {MIN_ATR_PERCENTAGE_FOR_ENTRY:.2f}%")
    logger.info(f"   🎲 Configured assets: {', '.join(PER_ASSET_ATR_THRESHOLDS.keys())}")
    return True
