"""
Market Regime Analyzer
Classifies market conditions to filter trading opportunities.
Only allows scalping in favorable trending conditions.
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
    Only trades when conditions are optimal for scalping.
    """
    
    def __init__(self):
        self.regime_history = []  # Last 20 regime classifications
        self.volatility_threshold_low = 0.3  # ATR% threshold
        self.volatility_threshold_high = 1.5
        
    async def classify_market_regime(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        Primary regime classifier using multi-factor analysis.
        
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
                "volume_trend": "INCREASING" | "DECREASING" | "STABLE"
            }
        """
        
        try:
            # Ensure we have required indicators
            if 'atr' not in df.columns or 'close' not in df.columns:
                logger.error("❌ Missing required indicators for regime analysis")
                return self._default_regime()
            
            # 1. Volatility Analysis (ATR-based)
            atr = df['atr'].iloc[-1]
            price = df['close'].iloc[-1]
            atr_percentage = (atr / price) * 100
            
            # Calculate ATR percentile over last 100 periods
            atr_history = df['atr'].tail(100)
            volatility_percentile = percentileofscore(atr_history, atr)
            
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
            
            # 5. Scalping Permission Logic
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
                "price_vs_sma20": price_vs_sma20
            }
            
            # Store in history
            self.regime_history.append(result)
            if len(self.regime_history) > 20:
                self.regime_history.pop(0)
            
            # Log regime classification
            emoji = "✅" if allow_scalping else "🚫"
            logger.info(
                f"{emoji} [REGIME] {symbol} | {regime} | "
                f"ADX={adx:.1f} | ATR%={atr_percentage:.2f}% | "
                f"Vol_Percentile={volatility_percentile:.0f} | "
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
        Decision tree for regime classification.
        
        Priority Order:
        1. High Volatility Check (safety first)
        2. Low Volatility Check
        3. Trend Detection (strong trends + volume confirmation)
        4. Default to Range-Bound
        """
        
        # High Volatility Check (First Priority - Safety)
        if atr_percentage > self.volatility_threshold_high or volatility_percentile > 90:
            return "HIGH_VOLATILITY"
        
        # Low Volatility Check
        if atr_percentage < self.volatility_threshold_low and volatility_percentile < 20:
            return "LOW_VOLATILITY"
        
        # Trend Detection (ADX > 25 indicates strong trend)
        if trend_strength > 25:
            # Volume confirmation strengthens trend classification
            # INCREASING volume confirms trend, DECREASING weakens it
            
            if sma_alignment == "BULL" and price_vs_sma20 > 0.3:
                # Require higher price_vs_sma20 if volume is decreasing
                if volume_trend == "DECREASING" and price_vs_sma20 < 0.5:
                    return "RANGE_BOUND"  # Weak volume = questionable trend
                return "BULL_TREND"
            elif sma_alignment == "BEAR" and price_vs_sma20 < -0.3:
                # Same logic for bearish trends
                if volume_trend == "DECREASING" and price_vs_sma20 > -0.5:
                    return "RANGE_BOUND"
                return "BEAR_TREND"
        
        # Default to range-bound if no clear trend
        return "RANGE_BOUND"
    
    def _should_allow_scalping(self, regime: str, trend_strength: float, atr_percentage: float) -> bool:
        """
        Determines if scalping is permitted in current regime.
        
        Scalping Rules:
        - ALLOWED: BULL_TREND, BEAR_TREND (with ADX > 25 and reasonable volatility)
        - FORBIDDEN: HIGH_VOLATILITY, RANGE_BOUND, LOW_VOLATILITY
        
        Philosophy: Only trade when there's a clear directional bias with momentum.
        """
        
        if regime in ["BULL_TREND", "BEAR_TREND"]:
            # Only scalp in direction of trend with sufficient momentum
            # and volatility within acceptable range
            return (
                trend_strength > 25 and 
                self.volatility_threshold_low < atr_percentage < self.volatility_threshold_high
            )
        
        # Block all other conditions
        return False
    
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
