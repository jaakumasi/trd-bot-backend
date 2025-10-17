"""
Market Regime Analyzer
Classifies market conditions to filter trading opportunities.
Supports configurable strictness levels for different trading environments.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from scipy.stats import percentileofscore
import logging

logger = logging.getLogger(__name__)


class MarketRegimeAnalyzer:
    """
    Classifies market into distinct regimes to filter trade opportunities.
    Supports three filtering modes:
    - STRICT: Only trending markets with ADX > 25 (production safeguard)
    - BALANCED: Trending + favorable range-bound with ADX > 20 (recommended)
    - PERMISSIVE: Most conditions except extreme volatility (testnet/development)
    Uses adaptive thresholds based on recent market behavior.
    """
    
    def __init__(self, filter_mode: str = "balanced"):
        self.regime_history = []  # Last 20 regime classifications
        self.atr_history_window = 200  # Track longer history for better percentiles
        self.filter_mode = filter_mode.lower()
        
        # Validate filter mode
        if self.filter_mode not in ["strict", "balanced", "permissive"]:
            logger.warning(f"⚠️  Invalid filter mode '{filter_mode}', defaulting to 'balanced'")
            self.filter_mode = "balanced"
        
        logger.info(f"🔧 Market Regime Analyzer initialized in {self.filter_mode.upper()} mode")
        
        # These are dynamic and will be recalculated based on recent data
        self.volatility_threshold_low = 0.05  # Will be updated to 20th percentile
        self.volatility_threshold_high = 0.12  # Will be updated to 80th percentile
        
    async def classify_market_regime(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        Primary regime classifier using multi-factor analysis with adaptive thresholds.
        
        Args:
            df: DataFrame with OHLCV data and technical indicators
            symbol: Trading pair symbol (e.g., "BTCUSDT")
        
        Returns:
            {
                "regime": "BULL_TREND" | "BEAR_TREND" | "RANGE_BOUND" | "HIGH_VOLATILITY" | "LOW_VOLATILITY",
                "confidence": 0-100,
                "volatility_percentile": 0-100,
                "trend_strength": 0-100,
                "atr_percentage": float,
                "allow_scalping": True/False,
                "sma_alignment": "BULL" | "BEAR" | "NEUTRAL",
                "volume_trend": "INCREASING" | "DECREASING" | "STABLE",
                "dynamic_vol_low": float,
                "dynamic_vol_high": float
            }
        """
        
        try:
            # Ensure we have required indicators
            if 'atr' not in df.columns or 'close' not in df.columns:
                logger.error("❌ Missing required indicators for regime analysis")
                return self._default_regime()
            
            # 1. Volatility Analysis (ATR-based) with Dynamic Thresholds
            atr = df['atr'].iloc[-1]
            price = df['close'].iloc[-1]
            atr_percentage = (atr / price) * 100
            
            # Calculate ATR percentile over available history (prefer 200, minimum 100)
            lookback = min(self.atr_history_window, len(df))
            atr_history = df['atr'].tail(lookback)
            volatility_percentile = percentileofscore(atr_history, atr)
            
            # Update dynamic thresholds based on recent market behavior
            # Convert ATR to percentages for threshold comparison
            atr_pct_history = (atr_history / df['close'].tail(lookback)) * 100
            self.volatility_threshold_low = float(atr_pct_history.quantile(0.20))  # 20th percentile
            self.volatility_threshold_high = float(atr_pct_history.quantile(0.80))  # 80th percentile
            # Update dynamic thresholds based on recent market behavior
            # Convert ATR to percentages for threshold comparison
            atr_pct_history = (atr_history / df['close'].tail(lookback)) * 100
            self.volatility_threshold_low = float(atr_pct_history.quantile(0.20))  # 20th percentile
            self.volatility_threshold_high = float(atr_pct_history.quantile(0.80))  # 80th percentile
            
            # 2. Trend Strength (ADX + Price vs MAs)
            adx = df['adx'].iloc[-1] if 'adx' in df.columns else 0.0
            sma_20 = df['sma_20'].iloc[-1] if 'sma_20' in df.columns else price
            sma_50 = df['sma_50'].iloc[-1] if 'sma_50' in df.columns else price
            
            # ADX: >25 = strong trend, <20 = weak/ranging
            trend_strength = min(adx, 100)
            
            # Price position relative to moving averages
            price_vs_sma20 = ((price - sma_20) / sma_20) * 100
            sma_alignment = self._get_sma_alignment(price, sma_20, sma_50)
            
            # 3. Market Breadth (Volume Trend)
            volume_trend = self._get_volume_trend(df)
            
            # 4. Regime Classification Logic
            regime = self._determine_regime(
                atr_percentage=atr_percentage,
                volatility_percentile=volatility_percentile,
                trend_strength=trend_strength,
                price_vs_sma20=price_vs_sma20,
                sma_alignment=sma_alignment,
                volume_trend=volume_trend
            )
            
            # 5. Scalping Permission Logic (now considers dynamic thresholds)
            allow_scalping = self._should_allow_scalping(regime, trend_strength, atr_percentage)
            
            # 6. Calculate confidence
            confidence = self._calculate_regime_confidence(df, regime, trend_strength, volatility_percentile)
            
            result = {
                "regime": regime,
                "confidence": confidence,
                "volatility_percentile": volatility_percentile,
                "trend_strength": trend_strength,
                "atr_percentage": atr_percentage,
                "allow_scalping": allow_scalping,
                "sma_alignment": sma_alignment,
                "volume_trend": volume_trend,
                "adx": adx,
                "price_vs_sma20": price_vs_sma20,
                "dynamic_vol_low": self.volatility_threshold_low,
                "dynamic_vol_high": self.volatility_threshold_high
            }
            
            # Store in history
            self.regime_history.append(result)
            if len(self.regime_history) > 20:
                self.regime_history.pop(0)
            
            # Log regime classification with dynamic thresholds
            emoji = "✅" if allow_scalping else "🚫"
            logger.info(
                f"{emoji} [REGIME-{self.filter_mode.upper()}] {symbol} | {regime} | "
                f"ADX={adx:.1f} | ATR%={atr_percentage:.3f}% | "
                f"Vol_Pct={volatility_percentile:.0f} | "
                f"Dynamic_Thresholds=[{self.volatility_threshold_low:.3f}%-{self.volatility_threshold_high:.3f}%] | "
                f"Scalping={'ALLOWED' if allow_scalping else 'BLOCKED'}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in regime classification: {e}")
            return self._default_regime()
    
    def _get_sma_alignment(self, price: float, sma_20: float, sma_50: float) -> str:
        """Determine SMA alignment (bullish/bearish/neutral)"""
        if pd.isna(sma_20) or pd.isna(sma_50):
            return "NEUTRAL"
        
        if sma_20 > sma_50 and price > sma_20:
            return "BULL"
        elif sma_20 < sma_50 and price < sma_20:
            return "BEAR"
        else:
            return "NEUTRAL"
    
    def _get_volume_trend(self, df: pd.DataFrame) -> str:
        """Analyze volume trend"""
        if 'volume' not in df.columns or 'volume_sma' not in df.columns:
            return "STABLE"
        
        volume_ratio = df['volume'].iloc[-1] / df['volume_sma'].iloc[-1]
        
        if volume_ratio > 1.2:
            return "INCREASING"
        elif volume_ratio < 0.8:
            return "DECREASING"
        else:
            return "STABLE"
    
    def _determine_regime(self, atr_percentage: float, volatility_percentile: float, 
                          trend_strength: float, price_vs_sma20: float, 
                          sma_alignment: str, volume_trend: str) -> str:
        """
        Decision tree for regime classification with dynamic volatility thresholds.
        
        Priority Order:
        1. High Volatility Check (safety first) - uses dynamic threshold
        2. Low Volatility Check - uses dynamic threshold
        3. Trend Detection (strong trends + volume confirmation)
        4. Default to Range-Bound
        """
        
        # High Volatility Check (First Priority - Safety)
        # Use BOTH percentile and dynamic threshold for robustness
        if atr_percentage > self.volatility_threshold_high or volatility_percentile > 85:
            return "HIGH_VOLATILITY"
        
        # Low Volatility Check - very tight conditions using dynamic threshold
        if atr_percentage < self.volatility_threshold_low and volatility_percentile < 15:
            return "LOW_VOLATILITY"
        
        # Trend Detection (ADX > 25 indicates strong trend)
        if trend_strength > 25:
            return self._classify_trend_direction(
                sma_alignment, price_vs_sma20, volume_trend
            )
        
        # Default to range-bound if no clear trend
        return "RANGE_BOUND"
    
    def _classify_trend_direction(self, sma_alignment: str, 
                                   price_vs_sma20: float, 
                                   volume_trend: str) -> str:
        """Helper to classify trend direction with volume confirmation"""
        # Volume confirmation strengthens trend classification
        
        if sma_alignment == "BULL" and price_vs_sma20 > 0.3:
            # Require higher price_vs_sma20 if volume is decreasing
            if volume_trend == "DECREASING" and price_vs_sma20 < 0.5:
                return "RANGE_BOUND"  # Weak volume = questionable trend
            return "BULL_TREND"
        
        if sma_alignment == "BEAR" and price_vs_sma20 < -0.3:
            # Same logic for bearish trends
            if volume_trend == "DECREASING" and price_vs_sma20 > -0.5:
                return "RANGE_BOUND"
            return "BEAR_TREND"
        
        return "RANGE_BOUND"
    
    def _should_allow_scalping(self, regime: str, trend_strength: float, atr_percentage: float) -> bool:
        """
        Determines if scalping is permitted in current regime based on filter mode.
        
        STRICT MODE (Production):
        - Only BULL_TREND, BEAR_TREND with ADX > 25
        - Blocks: HIGH_VOLATILITY, RANGE_BOUND, LOW_VOLATILITY
        
        BALANCED MODE (Recommended):
        - Trending markets: ADX > 20
        - Favorable range-bound: ADX 15-25 with moderate volatility
        - Blocks: HIGH_VOLATILITY, LOW_VOLATILITY with ADX < 15
        
        PERMISSIVE MODE (Testing/Development):
        - Allows most conditions except HIGH_VOLATILITY
        - Only requires ADX > 12 for basic trend validation
        - Ideal for testing AI signals in various conditions
        """
        
        # Always block extreme high volatility (safety first)
        if regime == "HIGH_VOLATILITY":
            return False
        
        if self.filter_mode == "strict":
            # Only strong trends
            if regime in ["BULL_TREND", "BEAR_TREND"]:
                return (
                    trend_strength > 25 and 
                    self.volatility_threshold_low < atr_percentage < self.volatility_threshold_high
                )
            return False
        
        elif self.filter_mode == "balanced":
            # Allow trending markets with slightly lower ADX
            if regime in ["BULL_TREND", "BEAR_TREND"]:
                return (
                    trend_strength > 20 and 
                    self.volatility_threshold_low < atr_percentage < self.volatility_threshold_high
                )
            
            # Allow range-bound markets with decent momentum
            # More forgiving than strict - just needs some ADX and reasonable volatility
            if regime == "RANGE_BOUND":
                # Must have some momentum (ADX 15-30) and not be in extreme zones
                has_momentum = 15 <= trend_strength <= 35
                reasonable_volatility = atr_percentage > 0.02  # At least 0.02% ATR
                return has_momentum and reasonable_volatility
            
            # Allow LOW_VOLATILITY if there's trend strength
            if regime == "LOW_VOLATILITY":
                return trend_strength > 18
            
            # Block HIGH_VOLATILITY only
            return False
        
        else:  # permissive mode
            # Allow almost everything except HIGH_VOLATILITY and dead markets
            if regime == "LOW_VOLATILITY":
                # Even low volatility is ok if there's some trend
                return trend_strength > 12
            
            # All other regimes allowed with minimal trend validation
            return trend_strength > 12
    
    def _calculate_regime_confidence(self, df: pd.DataFrame, regime: str, 
                                     trend_strength: float, volatility_percentile: float) -> float:
        """
        Calculate confidence in regime classification (0-100).
        
        Higher confidence when:
        - Strong ADX readings
        - Clear price momentum
        - Consistent recent regime history
        - Strong directional price movement
        """
        
        confidence = 50.0  # Base confidence
        
        # Factor 1: Trend Strength (0-25 points)
        if trend_strength > 40:
            confidence += 25
        elif trend_strength > 25:
            confidence += 15
        elif trend_strength < 15:
            confidence -= 10
        
        # Factor 2: Volatility Consistency (0-15 points)
        if 30 < volatility_percentile < 70:  # Mid-range volatility is more predictable
            confidence += 15
        elif volatility_percentile > 90 or volatility_percentile < 10:
            confidence -= 10
        
        # Factor 3: Recent Regime Consistency (0-10 points)
        if len(self.regime_history) >= 3:
            recent_regimes = [r['regime'] for r in self.regime_history[-3:]]
            if recent_regimes.count(regime) >= 2:  # Consistent regime
                confidence += 10
        
        # Factor 4: Price Momentum Confirmation (0-10 points)
        # Use the dataframe to check recent price direction
        if len(df) >= 5:
            recent_closes = df['close'].tail(5)
            price_change = (recent_closes.iloc[-1] - recent_closes.iloc[0]) / recent_closes.iloc[0] * 100
            
            # Strong directional movement increases confidence
            if abs(price_change) > 1.0:  # >1% movement in 5 periods
                confidence += 10
            elif abs(price_change) > 0.5:
                confidence += 5
        
        return max(0, min(100, confidence))
    
    def _default_regime(self) -> Dict:
        """Return safe default regime when analysis fails"""
        return {
            "regime": "RANGE_BOUND",
            "confidence": 0,
            "volatility_percentile": 50,
            "trend_strength": 0,
            "atr_percentage": 0,
            "allow_scalping": False,
            "sma_alignment": "NEUTRAL",
            "volume_trend": "STABLE",
            "adx": 0,
            "price_vs_sma20": 0
        }
    
    def get_regime_statistics(self) -> Dict:
        """Get statistics about recent regime classifications"""
        if not self.regime_history:
            return {"error": "No regime history available"}
        
        regimes = [r['regime'] for r in self.regime_history]
        scalping_allowed_count = sum(1 for r in self.regime_history if r['allow_scalping'])
        
        return {
            "total_classifications": len(self.regime_history),
            "scalping_allowed_rate": scalping_allowed_count / len(self.regime_history) * 100,
            "regime_distribution": {
                "BULL_TREND": regimes.count("BULL_TREND"),
                "BEAR_TREND": regimes.count("BEAR_TREND"),
                "RANGE_BOUND": regimes.count("RANGE_BOUND"),
                "HIGH_VOLATILITY": regimes.count("HIGH_VOLATILITY"),
                "LOW_VOLATILITY": regimes.count("LOW_VOLATILITY")
            },
            "avg_trend_strength": np.mean([r['trend_strength'] for r in self.regime_history]),
            "avg_volatility_percentile": np.mean([r['volatility_percentile'] for r in self.regime_history])
        }
