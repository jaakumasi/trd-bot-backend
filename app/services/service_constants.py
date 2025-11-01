"""BTCUSDT Day Trading - Adaptive Structure-Based Configuration

Professional crypto trading with context-aware validation.
NO fixed thresholds. NO multi-asset code. ONLY adaptive functions.
"""

from __future__ import annotations
import logging
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# CORE TRADING PARAMETERS
# ============================================================================

# Risk Management
MIN_ACCOUNT_BALANCE = 10.0
MAX_BALANCE_TRADE_RATIO = 0.03  # 3% max position size
MIN_TRADE_VALUE = 10.0
DEFAULT_RISK_PERCENTAGE = 1.0  # 1% rule
DEFAULT_MAX_DAILY_TRADES = 5
MAX_OPEN_POSITIONS = 2

# Risk:Reward Requirements
MIN_RISK_REWARD_RATIO = 2.0
TARGET_RISK_REWARD_RATIO = 2.5
MAX_RISK_REWARD_RATIO = 4.0

# Trading Cadence
ANALYSIS_RATE_LIMIT_SECONDS = 300
TRADING_CYCLE_DELAY_SECONDS = 300
POSITION_TIMEOUT_MINUTES = 480

# Multi-Timeframe Analysis
MTF_PRIMARY_INTERVAL = "15m"
MTF_CONTEXT_INTERVAL = "1h"
MTF_PRECISION_INTERVAL = "5m"
MTF_PRIMARY_CANDLES = 100
MTF_CONTEXT_CANDLES = 168
MTF_PRECISION_CANDLES = 288

# Numerical Safety
EIGHT_DECIMAL_PLACES = 100_000_000
OCO_SELL_STOP_LIMIT_BUFFER = 0.999
OCO_BUY_STOP_LIMIT_BUFFER = 1.001

# Quality & Confidence
MIN_SETUP_QUALITY = 65  # Composite quality score threshold
MIN_CONFIDENCE_FLOOR = 45  # Absolute minimum AI confidence

# Market Structure Detection
SR_LOOKBACK_PERIODS = 75
SR_TOUCH_THRESHOLD = 0.008
SR_MIN_TOUCHES = 2
SR_STRENGTH_DECAY = 0.9

# Stop Loss Boundaries
ABSOLUTE_MIN_STOP_PCT = 0.003
ABSOLUTE_MAX_STOP_PCT = 0.04

# Regime Detection
REGIME_ADX_TRENDING_THRESHOLD = 22
REGIME_ADX_STRONG_TREND_THRESHOLD = 40
REGIME_ALLOW_RANGING = True
REGIME_MIN_CONFLUENCE_TRENDING = 50
REGIME_MIN_CONFLUENCE_RANGING = 55


# ============================================================================
# ADAPTIVE VALIDATION FUNCTIONS
# ============================================================================

def is_volume_expanding(
    current_volume: float,
    volume_sma: float,
    regime: str,
    recent_volumes: pd.Series
) -> Tuple[bool, str, float]:
    """Context-aware volume validation using percentile ranking."""
    if volume_sma <= 0 or current_volume <= 0:
        return True, "Insufficient volume data", 1.0
    
    volume_ratio = current_volume / volume_sma
    
    if len(recent_volumes) >= 20:
        volume_percentile = (recent_volumes < current_volume).sum() / len(recent_volumes) * 100
    else:
        volume_percentile = 50
    
    if regime in ['BULL_TREND', 'BEAR_TREND']:
        if volume_percentile < 20:
            return False, f"Volume too low: {volume_percentile:.0f}th percentile", volume_ratio
        return True, f"Trend volume OK: {volume_percentile:.0f}th percentile", volume_ratio
    
    elif regime == 'RANGE_BOUND':
        if volume_percentile < 40:
            return False, f"Range needs volume: {volume_percentile:.0f}th percentile", volume_ratio
        return True, f"Range volume OK: {volume_percentile:.0f}th percentile", volume_ratio
    
    else:
        if volume_percentile < 30:
            return False, f"Volume below avg: {volume_percentile:.0f}th percentile", volume_ratio
        return True, f"Volume acceptable: {volume_percentile:.0f}th percentile", volume_ratio


def find_structural_stop(
    df: pd.DataFrame,
    side: str,
    entry_price: float,
    atr: float,
    lookback: int = 20
) -> Tuple[float, str]:
    """Place stop beyond market structure, not arbitrary percentages."""
    try:
        recent = df.tail(lookback)
        atr_buffer = atr * 0.5
        
        if side.lower() == 'buy':
            swing_low = recent['low'].min()
            stop_price = swing_low - atr_buffer
            stop_distance_pct = abs(entry_price - stop_price) / entry_price * 100
            
            if stop_distance_pct < (ABSOLUTE_MIN_STOP_PCT * 100):
                stop_price = entry_price * (1 - ABSOLUTE_MIN_STOP_PCT)
                return stop_price, f"Minimum {ABSOLUTE_MIN_STOP_PCT*100:.1f}% stop"
            
            if stop_distance_pct > (ABSOLUTE_MAX_STOP_PCT * 100):
                stop_price = entry_price * (1 - ABSOLUTE_MAX_STOP_PCT)
                return stop_price, f"Capped at {ABSOLUTE_MAX_STOP_PCT*100:.1f}% stop"
            
            return stop_price, f"Below ${swing_low:.2f} swing low ({stop_distance_pct:.2f}%)"
        
        else:
            swing_high = recent['high'].max()
            stop_price = swing_high + atr_buffer
            stop_distance_pct = abs(stop_price - entry_price) / entry_price * 100
            
            if stop_distance_pct < (ABSOLUTE_MIN_STOP_PCT * 100):
                stop_price = entry_price * (1 + ABSOLUTE_MIN_STOP_PCT)
                return stop_price, f"Minimum {ABSOLUTE_MIN_STOP_PCT*100:.1f}% stop"
            
            if stop_distance_pct > (ABSOLUTE_MAX_STOP_PCT * 100):
                stop_price = entry_price * (1 + ABSOLUTE_MAX_STOP_PCT)
                return stop_price, f"Capped at {ABSOLUTE_MAX_STOP_PCT*100:.1f}% stop"
            
            return stop_price, f"Above ${swing_high:.2f} swing high ({stop_distance_pct:.2f}%)"
    
    except Exception as e:
        logger.warning(f"Structural stop failed: {e}")
        if side.lower() == 'buy':
            stop_price = entry_price - (atr * 1.5)
        else:
            stop_price = entry_price + (atr * 1.5)
        return stop_price, "Fallback: 1.5x ATR stop"


def assess_setup_quality(
    signal: Dict,
    regime_analysis: Dict,
    market_df: pd.DataFrame,
    recent_win_rate: Optional[float] = None
) -> Tuple[int, str]:
    """Calculate composite setup quality score (0-100)."""
    score = 0
    factors = []
    
    # AI Confidence (0-30 pts)
    confidence = signal.get('final_confidence', signal.get('confidence', 50))
    confidence_score = min(30, int(confidence * 0.5))
    score += confidence_score
    factors.append(f"AI: {confidence:.0f}% -> {confidence_score}pts")
    
    # Regime Alignment (0-25 pts)
    regime = regime_analysis.get('regime', 'UNKNOWN')
    regime_confluence = regime_analysis.get('confluence_score', 50)
    
    if regime in ['BULL_TREND', 'BEAR_TREND']:
        alignment_score = min(25, int(regime_confluence * 0.4))
    elif regime == 'RANGE_BOUND':
        alignment_score = min(20, int(regime_confluence * 0.35))
    else:
        alignment_score = 10
    
    score += alignment_score
    factors.append(f"Regime: {regime} -> {alignment_score}pts")
    
    # Technical Confluence (0-25 pts)
    try:
        latest = market_df.iloc[-1]
        rsi = latest.get('rsi', 50)
        macd = latest.get('macd', 0)
        macd_signal = latest.get('macd_signal', 0)
        close = latest.get('close', 0)
        sma_20 = latest.get('sma_20', close)
        sma_50 = latest.get('sma_50', close)
        
        confluence_count = 0
        signal_action = signal.get('signal', 'hold').lower()
        
        if signal_action == 'buy':
            if rsi < 70: confluence_count += 1
            if macd > macd_signal: confluence_count += 1
            if close > sma_20: confluence_count += 1
            if sma_20 > sma_50: confluence_count += 1
        elif signal_action == 'sell':
            if rsi > 30: confluence_count += 1
            if macd < macd_signal: confluence_count += 1
            if close < sma_20: confluence_count += 1
            if sma_20 < sma_50: confluence_count += 1
        
        confluence_score = int(confluence_count * 6.25)
        score += confluence_score
        factors.append(f"Technical: {confluence_count}/4 -> {confluence_score}pts")
    except Exception:
        factors.append("Technical: 0pts")
    
    # Performance (0-20 pts)
    if recent_win_rate is not None:
        if recent_win_rate >= 60:
            perf_score = 20
        elif recent_win_rate >= 50:
            perf_score = 15
        elif recent_win_rate >= 40:
            perf_score = 10
        else:
            perf_score = 0
        score += perf_score
        factors.append(f"WR: {recent_win_rate:.0f}% -> {perf_score}pts")
    else:
        score += 10
        factors.append("WR: none -> 10pts")
    
    explanation = f"Quality: {score}/100 | " + " | ".join(factors)
    return score, explanation


def calculate_dynamic_target(
    df: pd.DataFrame,
    side: str,
    entry_price: float,
    stop_price: float,
    min_rr: float = 2.0
) -> Tuple[float, str]:
    """Calculate target based on next structural level."""
    try:
        risk = abs(entry_price - stop_price)
        min_reward = risk * min_rr
        recent = df.tail(SR_LOOKBACK_PERIODS)
        
        if side.lower() == 'buy':
            resistances = recent[recent['high'] > entry_price]['high'].values
            if len(resistances) > 0:
                target = np.percentile(resistances, 75)
                target_price = max(target, entry_price + min_reward)
                rr = (target_price - entry_price) / risk
                return target_price, f"Target ${target_price:.2f} ({rr:.2f}:1)"
            else:
                target_price = entry_price + min_reward
                return target_price, f"Min {min_rr}:1 target"
        else:
            supports = recent[recent['low'] < entry_price]['low'].values
            if len(supports) > 0:
                target = np.percentile(supports, 25)
                target_price = min(target, entry_price - min_reward)
                rr = (entry_price - target_price) / risk
                return target_price, f"Target ${target_price:.2f} ({rr:.2f}:1)"
            else:
                target_price = entry_price - min_reward
                return target_price, f"Min {min_rr}:1 target"
    except Exception as e:
        logger.warning(f"Dynamic target failed: {e}")
        risk = abs(entry_price - stop_price)
        if side.lower() == 'buy':
            return entry_price + (risk * min_rr), f"{min_rr}:1 fallback"
        else:
            return entry_price - (risk * min_rr), f"{min_rr}:1 fallback"


def validate_constants() -> bool:
    """Validate configuration at startup."""
    issues = []
    
    if not (0 < SR_TOUCH_THRESHOLD < 1):
        issues.append("SR_TOUCH_THRESHOLD invalid")
    if not (1.0 <= MIN_RISK_REWARD_RATIO <= MAX_RISK_REWARD_RATIO):
        issues.append("R:R ratios invalid")
    if not (0 < MAX_BALANCE_TRADE_RATIO <= 0.05):
        issues.append("Position size invalid")
    
    if issues:
        logger.error("Config validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    
    logger.info("Config validated")
    logger.info("  Mode: BTCUSDT adaptive structure-based")
    logger.info(f"  Quality: {MIN_SETUP_QUALITY}/100 min")
    logger.info(f"  Position: {MAX_BALANCE_TRADE_RATIO*100:.0f}% max")
    return True


__all__ = [
    "MIN_ACCOUNT_BALANCE",
    "MAX_BALANCE_TRADE_RATIO",
    "MIN_TRADE_VALUE",
    "DEFAULT_RISK_PERCENTAGE",
    "DEFAULT_MAX_DAILY_TRADES",
    "MAX_OPEN_POSITIONS",
    "MIN_RISK_REWARD_RATIO",
    "TARGET_RISK_REWARD_RATIO",
    "MAX_RISK_REWARD_RATIO",
    "ANALYSIS_RATE_LIMIT_SECONDS",
    "TRADING_CYCLE_DELAY_SECONDS",
    "POSITION_TIMEOUT_MINUTES",
    "MTF_PRIMARY_INTERVAL",
    "MTF_CONTEXT_INTERVAL",
    "MTF_PRECISION_INTERVAL",
    "MTF_PRIMARY_CANDLES",
    "MTF_CONTEXT_CANDLES",
    "MTF_PRECISION_CANDLES",
    "EIGHT_DECIMAL_PLACES",
    "OCO_SELL_STOP_LIMIT_BUFFER",
    "OCO_BUY_STOP_LIMIT_BUFFER",
    "MIN_SETUP_QUALITY",
    "MIN_CONFIDENCE_FLOOR",
    "SR_LOOKBACK_PERIODS",
    "SR_TOUCH_THRESHOLD",
    "SR_MIN_TOUCHES",
    "SR_STRENGTH_DECAY",
    "ABSOLUTE_MIN_STOP_PCT",
    "ABSOLUTE_MAX_STOP_PCT",
    "REGIME_ADX_TRENDING_THRESHOLD",
    "REGIME_ADX_STRONG_TREND_THRESHOLD",
    "REGIME_ALLOW_RANGING",
    "REGIME_MIN_CONFLUENCE_TRENDING",
    "REGIME_MIN_CONFLUENCE_RANGING",
    "is_volume_expanding",
    "find_structural_stop",
    "assess_setup_quality",
    "calculate_dynamic_target",
    "validate_constants",
]
