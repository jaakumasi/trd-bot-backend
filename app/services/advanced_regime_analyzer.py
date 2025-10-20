"""
Advanced Market Regime Analyzer with Microstructure Analysis
Advanced day trading strategy using momentum persistence, order flow, and volatility clustering.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from scipy.stats import percentileofscore
import logging

logger = logging.getLogger(__name__)


class AdvancedRegimeAnalyzer:
    """
    Next-generation market regime analyzer optimized for day trading with:
    - Momentum Persistence Analysis
    - Order Flow Imbalance Detection
    - Volatility Clustering (GARCH-style)
    - Mean Reversion Zone Identification
    - Dynamic Confluence Thresholds
    """
    
    def __init__(self, filter_mode: str = "day_trading"):
        self.regime_history = []
        self.atr_history_window = 200
        self.filter_mode = filter_mode.lower()
        
        # Validate filter mode
        if self.filter_mode not in ["strict", "balanced", "permissive", "day_trading"]:
            logger.warning(f"⚠️  Invalid filter mode '{filter_mode}', defaulting to 'day_trading'")
            self.filter_mode = "day_trading"
        
        logger.info(f"🚀 Advanced Market Regime Analyzer initialized in {self.filter_mode.upper()} mode")
        
        # Dynamic thresholds
        self.volatility_threshold_low = 0.05
        self.volatility_threshold_high = 0.12
        
        # Microstructure tracking
        self.momentum_persistence_history = []
        self.order_flow_history = []
        
    async def classify_market_regime(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        Advanced regime classification with microstructure analysis.
        
        Returns comprehensive market state including:
        - Traditional regime (BULL/BEAR/RANGE/HIGH_VOL/LOW_VOL)
        - Momentum persistence score (0-100)
        - Order flow imbalance (-100 to +100)
        - Volatility clustering factor
        - Mean reversion opportunity score
        - Dynamic trading quality score (0-100)
        """
        
        try:
            # Ensure we have required indicators
            if 'atr' not in df.columns or 'close' not in df.columns:
                logger.error("❌ Missing required indicators for regime analysis")
                return self._default_regime()
            
            # === TRADITIONAL ANALYSIS ===
            atr = df['atr'].iloc[-1]
            price = df['close'].iloc[-1]
            atr_percentage = (atr / price) * 100
            
            # Calculate ATR percentile
            lookback = min(self.atr_history_window, len(df))
            atr_history = df['atr'].tail(lookback)
            volatility_percentile = percentileofscore(atr_history, atr)
            
            # Update dynamic thresholds
            atr_pct_history = (atr_history / df['close'].tail(lookback)) * 100
            self.volatility_threshold_low = float(atr_pct_history.quantile(0.20))
            self.volatility_threshold_high = float(atr_pct_history.quantile(0.80))
            
            # Trend Strength
            adx = df['adx'].iloc[-1] if 'adx' in df.columns else 0.0
            sma_20 = df['sma_20'].iloc[-1] if 'sma_20' in df.columns else price
            sma_50 = df['sma_50'].iloc[-1] if 'sma_50' in df.columns else price
            
            trend_strength = min(adx, 100)
            price_vs_sma20 = ((price - sma_20) / sma_20) * 100
            sma_alignment = self._get_sma_alignment(price, sma_20, sma_50)
            volume_trend = self._get_volume_trend(df)
            
            # Base regime classification
            regime = self._determine_regime(
                atr_percentage, volatility_percentile, trend_strength,
                price_vs_sma20, sma_alignment, volume_trend
            )
            
            # === ADVANCED MICROSTRUCTURE ANALYSIS ===
            
            # 1. Momentum Persistence Score (0-100)
            momentum_persistence = self._calculate_momentum_persistence(df)
            
            # 2. Order Flow Imbalance (-100 to +100)
            order_flow_imbalance = self._calculate_order_flow_imbalance(df)
            
            # 3. Volatility Clustering Factor (0-100)
            volatility_clustering = self._calculate_volatility_clustering(df)
            
            # 4. Mean Reversion Opportunity Score (0-100)
            mean_reversion_score = self._calculate_mean_reversion_opportunity(df)
            
            # 5. Dynamic Trading Quality Score (0-100)
            trading_quality = self._calculate_trading_quality(
                regime, trend_strength, atr_percentage, momentum_persistence,
                order_flow_imbalance, volatility_clustering, mean_reversion_score
            )
            
            # 6. Adaptive Confluence Threshold
            dynamic_threshold = self._calculate_dynamic_confluence_threshold(
                trading_quality, volatility_clustering
            )
            
            # === TRADING PERMISSION (now based on quality score) ===
            allow_trading = trading_quality >= 50  # Quality-based, not binary regime check
            
            # Calculate confidence
            confidence = self._calculate_advanced_confidence(
                df, regime, trend_strength, volatility_percentile,
                trading_quality, momentum_persistence
            )
            
            result = {
                # Traditional metrics
                "regime": regime,
                "confidence": confidence,
                "volatility_percentile": volatility_percentile,
                "trend_strength": trend_strength,
                "atr_percentage": atr_percentage,
                "allow_trading": allow_trading,
                "sma_alignment": sma_alignment,
                "volume_trend": volume_trend,
                "adx": adx,
                "price_vs_sma20": price_vs_sma20,
                "dynamic_vol_low": self.volatility_threshold_low,
                "dynamic_vol_high": self.volatility_threshold_high,
                
                # Advanced microstructure metrics
                "momentum_persistence": momentum_persistence,
                "order_flow_imbalance": order_flow_imbalance,
                "volatility_clustering": volatility_clustering,
                "mean_reversion_score": mean_reversion_score,
                "trading_quality_score": trading_quality,
                "dynamic_confluence_threshold": dynamic_threshold,
                
                # Trading recommendations
                "trading_edge": self._determine_trading_edge(
                    order_flow_imbalance, mean_reversion_score, momentum_persistence
                ),
                "optimal_hold_time": self._estimate_optimal_hold_time(
                    momentum_persistence, volatility_clustering
                )
            }
            
            # Store in history
            self.regime_history.append(result)
            if len(self.regime_history) > 20:
                self.regime_history.pop(0)
            
            # Enhanced logging
            emoji = "✅" if allow_trading else "🚫"
            quality_emoji = "🔥" if trading_quality > 70 else "⚡" if trading_quality > 50 else "❄️"
            
            logger.info(
                f"{emoji}{quality_emoji} [ADVANCED-{self.filter_mode.upper()}] {symbol} | {regime} | "
                f"Quality={trading_quality:.0f} | MomPersist={momentum_persistence:.0f} | "
                f"OrderFlow={order_flow_imbalance:+.0f} | VolCluster={volatility_clustering:.0f} | "
                f"MeanRev={mean_reversion_score:.0f} | Threshold={dynamic_threshold:.0f} | "
                f"Edge={result['trading_edge']} | HoldTime={result['optimal_hold_time']}min"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in advanced regime classification: {e}", exc_info=True)
            return self._default_regime()
    
    def _calculate_momentum_persistence(self, df: pd.DataFrame, lookback: int = 10) -> float:
        """
        Calculate how long momentum tends to persist (key for day trading exits).
        
        Analyzes recent price movements to predict how long the current momentum will last.
        High persistence = trends continue, good for trend-following trades
        Low persistence = choppy reversals, better for mean reversion trades
        
        Returns: 0-100 score
        """
        if len(df) < lookback + 5:
            return 50.0
        
        closes = df['close'].tail(lookback + 5).values
        score = 50.0  # Base score
        
        # Calculate directional consistency
        price_changes = np.diff(closes)
        positive_moves = np.sum(price_changes > 0)
        negative_moves = np.sum(price_changes < 0)
        
        # High directional consistency = high persistence
        consistency_ratio = max(positive_moves, negative_moves) / len(price_changes)
        score += (consistency_ratio - 0.5) * 100  # -50 to +50 adjustment
        
        # Check momentum acceleration (is it building or fading?)
        recent_momentum = np.mean(np.abs(price_changes[-3:]))
        earlier_momentum = np.mean(np.abs(price_changes[:3]))
        
        if recent_momentum > earlier_momentum * 1.2:
            score += 20  # Accelerating momentum = higher persistence
        elif recent_momentum < earlier_momentum * 0.8:
            score -= 20  # Fading momentum = lower persistence
        
        # Track in history for adaptive learning
        final_score = max(0, min(100, score))
        self.momentum_persistence_history.append(final_score)
        if len(self.momentum_persistence_history) > 50:
            self.momentum_persistence_history.pop(0)
        
        return final_score
    
    def _calculate_order_flow_imbalance(self, df: pd.DataFrame, lookback: int = 20) -> float:
        """
        Detect institutional buying/selling pressure using volume-weighted price changes.
        
        Positive imbalance = buying pressure (institutional accumulation)
        Negative imbalance = selling pressure (institutional distribution)
        
        Returns: -100 to +100
        """
        if len(df) < lookback or 'volume' not in df.columns:
            return 0.0
        
        recent_df = df.tail(lookback)
        
        # Calculate volume-weighted price changes
        price_changes = recent_df['close'].diff()
        volumes = recent_df['volume']
        
        # Separate buying vs selling pressure
        buying_volume = np.sum(volumes[price_changes > 0])
        selling_volume = np.sum(volumes[price_changes < 0])
        
        total_volume = buying_volume + selling_volume
        
        if total_volume == 0:
            return 0.0
        
        # Calculate imbalance (-1 to +1)
        imbalance_ratio = (buying_volume - selling_volume) / total_volume
        
        # Scale to -100 to +100
        imbalance_score = imbalance_ratio * 100
        
        # Track in history
        self.order_flow_history.append(imbalance_score)
        if len(self.order_flow_history) > 50:
            self.order_flow_history.pop(0)
        
        return imbalance_score
    
    def _calculate_volatility_clustering(self, df: pd.DataFrame, lookback: int = 30) -> float:
        """
        Volatility clustering: high volatility tends to follow high volatility.
        
        This GARCH-like concept helps predict near-term volatility from recent volatility.
        High clustering = current volatility likely to persist
        Low clustering = volatility likely to change
        
        Returns: 0-100 score
        """
        if len(df) < lookback or 'atr' not in df.columns:
            return 50.0
        
        atr_values = df['atr'].tail(lookback).values
        
        # Calculate volatility of volatility (how clustered is it?)
        atr_std = np.std(atr_values)
        atr_mean = np.mean(atr_values)
        
        if atr_mean == 0:
            return 50.0
        
        # Low coefficient of variation = high clustering
        cv = atr_std / atr_mean
        
        # Convert to 0-100 score (lower CV = higher score)
        clustering_score = max(0, min(100, 100 - (cv * 100)))
        
        return clustering_score
    
    def _calculate_mean_reversion_opportunity(self, df: pd.DataFrame) -> float:
        """
        Identify when price has stretched too far from mean (day trading opportunity!).
        
        Uses Bollinger Bands + RSI + price vs SMA deviation to find extremes.
        High score = excellent mean reversion day trading opportunity
        
        Returns: 0-100 score
        """
        if len(df) < 20:
            return 0.0
        
        latest = df.iloc[-1]
        score = 0.0
        
        # Factor 1: Bollinger Band position (40 points)
        if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'bb_middle', 'close']):
            bb_upper = latest['bb_upper']
            bb_lower = latest['bb_lower']
            bb_middle = latest['bb_middle']
            price = latest['close']
            
            if not pd.isna(bb_upper) and not pd.isna(bb_lower):
                bb_range = bb_upper - bb_lower
                if bb_range > 0:
                    # How far from middle? (0 = at middle, 1 = at band)
                    deviation = abs(price - bb_middle) / (bb_range / 2)
                    
                    if deviation > 0.9:  # Very stretched
                        score += 40
                    elif deviation > 0.7:
                        score += 30
                    elif deviation > 0.5:
                        score += 15
        
        # Factor 2: RSI extremes (30 points)
        if 'rsi' in df.columns:
            rsi = latest['rsi']
            if not pd.isna(rsi):
                if rsi < 30 or rsi > 70:  # Oversold/overbought
                    score += 30
                elif rsi < 40 or rsi > 60:
                    score += 15
        
        # Factor 3: Distance from SMA (30 points)
        if 'sma_20' in df.columns and 'close' in df.columns:
            sma_20 = latest['sma_20']
            price = latest['close']
            
            if not pd.isna(sma_20) and sma_20 > 0:
                price_deviation_pct = abs((price - sma_20) / sma_20) * 100
                
                if price_deviation_pct > 2.0:  # >2% from mean
                    score += 30
                elif price_deviation_pct > 1.0:
                    score += 20
                elif price_deviation_pct > 0.5:
                    score += 10
        
        return min(100, score)
    
    def _calculate_trading_quality(
        self, regime: str, trend_strength: float, atr_percentage: float,
        momentum_persistence: float, order_flow_imbalance: float,
        volatility_clustering: float, mean_reversion_score: float
) -> float:
        """
        REVOLUTIONARY: Calculate overall trading quality based on microstructure.
        
        This replaces the binary allow/block with a nuanced 0-100 quality score.
        Different regimes can have high quality if microstructure is favorable!
        
        Returns: 0-100 quality score
        """
        quality = 0.0
        
        # === Regime Base Quality (0-30 points) ===
        if regime in ["BULL_TREND", "BEAR_TREND"]:
            quality += 30  # Trends are great
            
            # Bonus for strong trends
            if trend_strength > 40:
                quality += 10
                
        elif regime == "RANGE_BOUND":
            # Range can be EXCELLENT for day trading with proper risk management!
            quality += 15  # Base points for range
            
            # Bonus if mean reversion opportunity is high
            if mean_reversion_score > 60:
                quality += 15  # Range + mean reversion = gold
                
        elif regime == "LOW_VOLATILITY":
            # Low vol can work if momentum persists
            quality += 10
            
            if momentum_persistence > 60:
                quality += 15
                
        elif regime == "HIGH_VOLATILITY":
            # High vol is risky but can work with strong order flow
            quality += 5
            
            if abs(order_flow_imbalance) > 50:
                quality += 10  # Strong institutional flow helps
        
        # === Microstructure Bonuses (up to +70 points) ===
        
        # Momentum persistence (0-20 points)
        if momentum_persistence > 70:
            quality += 20
        elif momentum_persistence > 50:
            quality += 10
        
        # Order flow imbalance (0-20 points)
        if abs(order_flow_imbalance) > 60:  # Strong directional flow
            quality += 20
        elif abs(order_flow_imbalance) > 40:
            quality += 10
        
        # Volatility clustering (0-15 points)
        if volatility_clustering > 70:  # Predictable volatility
            quality += 15
        elif volatility_clustering > 50:
            quality += 8
        
        # Mean reversion (0-15 points)
        if mean_reversion_score > 70:  # Prime mean reversion setup
            quality += 15
        elif mean_reversion_score > 50:
            quality += 8
        
        # === ATR Sweet Spot Bonus (+5 points) ===
        if 0.1 < atr_percentage < 0.8:  # Optimal volatility for day trading
            quality += 5
        
        return max(0, min(100, quality))
    
    def _calculate_dynamic_confluence_threshold(
        self, trading_quality: float, volatility_clustering: float
) -> float:
        """
        Adaptive confluence threshold based on market conditions.
        
        More aggressive thresholds for day trading (35-60 range instead of 40-70).
        When quality is high, we can accept lower confluence (be more aggressive).
        When volatility is unpredictable, require higher confluence (be cautious).
        
        Returns: Dynamic threshold (35-60)
        """
        base_threshold = 50.0
        
        # Adjust down for high-quality setups (can be more aggressive)
        if trading_quality > 80:
            base_threshold -= 15  # Can go as low as 35
        elif trading_quality > 70:
            base_threshold -= 10  # Can go as low as 40
        elif trading_quality > 60:
            base_threshold -= 5   # Can go as low as 45
        
        # Adjust up for unpredictable volatility (need more confirmation)
        if volatility_clustering < 30:
            base_threshold += 10  # Max 60 even in chaotic conditions
        elif volatility_clustering < 50:
            base_threshold += 5   # Max 55
        
        final_threshold = max(35, min(60, base_threshold))
        logger.debug(f"🎯 Dynamic confluence threshold: {final_threshold} (quality: {trading_quality:.0f}, vol_cluster: {volatility_clustering:.0f})")
        return final_threshold
    
    def _determine_trading_edge(
        self, order_flow: float, mean_reversion: float, momentum: float
    ) -> str:
        """
        Identify the primary edge in current market conditions.
        
        Returns: "TREND_FOLLOWING" | "MEAN_REVERSION" | "BREAKOUT" | "NEUTRAL"
        """
        if abs(order_flow) > 50 and momentum > 60:
            return "TREND_FOLLOWING"
        elif mean_reversion > 70:
            return "MEAN_REVERSION"
        elif abs(order_flow) > 60 and mean_reversion > 50:
            return "BREAKOUT"
        else:
            return "NEUTRAL"
    
    def _estimate_optimal_hold_time(
        self, momentum_persistence: float, volatility_clustering: float
    ) -> str:
        """
        Estimate optimal position hold time based on microstructure.
        
        Returns: Time range in minutes
        """
        if momentum_persistence > 70 and volatility_clustering > 60:
            return "15-30"  # Strong persistent momentum
        elif momentum_persistence > 50:
            return "10-20"  # Moderate momentum
        else:
            return "5-15"  # Quick trades
    
    # === HELPER METHODS (from original implementation) ===
    
    def _get_sma_alignment(self, price: float, sma_20: float, sma_50: float) -> str:
        if pd.isna(sma_20) or pd.isna(sma_50):
            return "NEUTRAL"
        if sma_20 > sma_50 and price > sma_20:
            return "BULL"
        elif sma_20 < sma_50 and price < sma_20:
            return "BEAR"
        else:
            return "NEUTRAL"
    
    def _get_volume_trend(self, df: pd.DataFrame) -> str:
        if 'volume' not in df.columns or 'volume_sma' not in df.columns:
            return "STABLE"
        volume_ratio = df['volume'].iloc[-1] / df['volume_sma'].iloc[-1]
        if volume_ratio > 1.2:
            return "INCREASING"
        elif volume_ratio < 0.8:
            return "DECREASING"
        else:
            return "STABLE"
    
    def _determine_regime(
        self, atr_percentage: float, volatility_percentile: float,
        trend_strength: float, price_vs_sma20: float,
        sma_alignment: str, volume_trend: str
    ) -> str:
        # Strong trend detection first
        if trend_strength > 35:
            return self._classify_trend_direction(sma_alignment, price_vs_sma20, volume_trend)
        
        # High volatility check
        if volatility_percentile > 95 and trend_strength < 25:
            return "HIGH_VOLATILITY"
        
        # Low volatility check
        if atr_percentage < self.volatility_threshold_low and volatility_percentile < 20:
            return "LOW_VOLATILITY"
        
        # Moderate trend detection
        if trend_strength > 20:
            return self._classify_trend_direction(sma_alignment, price_vs_sma20, volume_trend)
        
        return "RANGE_BOUND"
    
    def _classify_trend_direction(
        self, sma_alignment: str, price_vs_sma20: float, volume_trend: str
    ) -> str:
        if sma_alignment == "BULL" and price_vs_sma20 > 0.3:
            if volume_trend == "DECREASING" and price_vs_sma20 < 0.5:
                return "RANGE_BOUND"
            return "BULL_TREND"
        if sma_alignment == "BEAR" and price_vs_sma20 < -0.3:
            if volume_trend == "DECREASING" and price_vs_sma20 > -0.5:
                return "RANGE_BOUND"
            return "BEAR_TREND"
        return "RANGE_BOUND"
    
    def _calculate_advanced_confidence(
        self, df: pd.DataFrame, regime: str, trend_strength: float,
        volatility_percentile: float, trading_quality: float,
        momentum_persistence: float
) -> float:
        confidence = 50.0
        
        # Trend strength factor
        if trend_strength > 40:
            confidence += 25
        elif trend_strength > 25:
            confidence += 15
        elif trend_strength < 15:
            confidence -= 10
        
        # Volatility consistency factor
        if 30 < volatility_percentile < 70:
            confidence += 15
        elif volatility_percentile > 90 or volatility_percentile < 10:
            confidence -= 10
        
        # Trading quality boost
        if trading_quality > 75:
            confidence += 10
        elif trading_quality > 60:
            confidence += 5
        
        # Momentum persistence factor
        if momentum_persistence > 70:
            confidence += 10
        
        return max(0, min(100, confidence))
    
    def _default_regime(self) -> Dict:
        return {
            "regime": "RANGE_BOUND",
            "confidence": 0,
            "volatility_percentile": 50,
            "trend_strength": 0,
            "atr_percentage": 0,
            "allow_trading": False,
            "sma_alignment": "NEUTRAL",
            "volume_trend": "STABLE",
            "adx": 0,
            "price_vs_sma20": 0,
            "momentum_persistence": 50,
            "order_flow_imbalance": 0,
            "volatility_clustering": 50,
            "mean_reversion_score": 0,
            "trading_quality_score": 0,
            "dynamic_confluence_threshold": 55,
            "trading_edge": "NEUTRAL",
            "optimal_hold_time": "10-20"
        }
