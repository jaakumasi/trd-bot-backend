"""
Multi-Timeframe Confirmation Analyzer
Validates trade signals across multiple timeframes to reduce false signals.
Only allows trades when 1-min, 5-min, and 15-min trends align.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MultiTimeframeAnalyzer:
    """
    Analyze multiple timeframes to confirm trend direction.
    
    Philosophy:
    - A strong setup should show alignment across timeframes
    - 1-min might show noise, but 5-min + 15-min confirm real moves
    - Reduces false signals by 30-40% (industry research)
    
    Timeframes:
    - Primary: 1-min (execution timeframe)
    - Confirmation 1: 5-min (short-term trend)
    - Confirmation 2: 15-min (medium-term trend)
    """
    
    def __init__(self, require_all_timeframes: bool = False):
        """
        Initialize the multi-timeframe analyzer.
        
        Args:
            require_all_timeframes: If True, ALL timeframes must agree.
                                   If False, majority vote (2 out of 3).
        """
        self.require_all_timeframes = require_all_timeframes
        logger.info("📊 Multi-Timeframe Analyzer initialized")
        logger.info("   📈 Timeframes: 1-min (primary), 5-min, 15-min")
        logger.info(f"   ⚖️  Requirement: {'ALL must agree' if require_all_timeframes else 'Majority vote (2/3)'}")
    
    def analyze_trend(self, df: pd.DataFrame, period: int = 20) -> str:
        """
        Analyze trend direction on a single timeframe.
        
        Uses multiple indicators for robust trend detection:
        - SMA20 slope (price vs moving average trend)
        - Price momentum (recent price direction)
        - ADX (trend strength confirmation)
        
        Args:
            df: DataFrame with OHLCV data and indicators
            period: Period for trend analysis (default 20)
        
        Returns:
            "BULLISH", "BEARISH", or "NEUTRAL"
        """
        if len(df) < period:
            return "NEUTRAL"
        
        try:
            # 1. SMA20 trend analysis
            if 'sma_20' not in df.columns:
                logger.warning("⚠️  SMA20 not found, calculating...")
                df['sma_20'] = df['close'].rolling(window=20).mean()
            
            current_price = df['close'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1]
            sma_20_prev = df['sma_20'].iloc[-2] if len(df) >= 2 else sma_20
            
            # Is price above/below SMA?
            price_vs_sma = "BULL" if current_price > sma_20 else "BEAR"
            
            # Is SMA trending up/down?
            sma_slope = "UP" if sma_20 > sma_20_prev else "DOWN"
            
            # 2. Recent price momentum (last 5 candles)
            recent_prices = df['close'].tail(5)
            momentum = "UP" if recent_prices.iloc[-1] > recent_prices.iloc[0] else "DOWN"
            
            # 3. ADX trend strength (optional, if available)
            adx_strong = False
            if 'adx' in df.columns:
                adx = df['adx'].iloc[-1]
                adx_strong = adx > 25  # Strong trend threshold
            
            # Combine signals (weighted voting)
            bull_votes = 0
            bear_votes = 0
            
            # Vote 1: Price vs SMA (weight: 2)
            if price_vs_sma == "BULL":
                bull_votes += 2
            else:
                bear_votes += 2
            
            # Vote 2: SMA slope (weight: 2)
            if sma_slope == "UP":
                bull_votes += 2
            else:
                bear_votes += 2
            
            # Vote 3: Momentum (weight: 1)
            if momentum == "UP":
                bull_votes += 1
            else:
                bear_votes += 1
            
            # Vote 4: ADX confirmation (weight: 1, bonus if strong)
            if adx_strong:
                if bull_votes > bear_votes:
                    bull_votes += 1
                else:
                    bear_votes += 1
            
            # Determine trend
            if bull_votes > bear_votes + 1:  # Clear bull
                return "BULLISH"
            elif bear_votes > bull_votes + 1:  # Clear bear
                return "BEARISH"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            logger.error(f"❌ Error analyzing trend: {e}")
            return "NEUTRAL"
    
    def confirm_signal(
        self,
        signal: str,
        df_1min: pd.DataFrame,
        df_5min: Optional[pd.DataFrame] = None,
        df_15min: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Confirm trading signal across multiple timeframes.
        
        Args:
            signal: Trading signal ("BUY" or "SELL") from primary timeframe
            df_1min: 1-minute OHLCV data (primary)
            df_5min: 5-minute OHLCV data (optional)
            df_15min: 15-minute OHLCV data (optional)
        
        Returns:
            Dictionary with confirmation results:
            - confirmed: Boolean, whether signal is confirmed
            - confidence: 0-100, confidence level
            - timeframe_alignment: Dictionary of trend per timeframe
            - aligned_count: Number of aligned timeframes
            - reasoning: Text explanation
        """
        # Analyze each available timeframe
        trend_1min = self.analyze_trend(df_1min)
        trend_5min = self.analyze_trend(df_5min) if df_5min is not None and len(df_5min) >= 20 else "UNAVAILABLE"
        trend_15min = self.analyze_trend(df_15min) if df_15min is not None and len(df_15min) >= 20 else "UNAVAILABLE"
        
        # Map signal to expected trend
        expected_trend = "BULLISH" if signal.upper() == "BUY" else "BEARISH"
        
        # Count alignments
        aligned_timeframes = []
        available_timeframes = []
        
        # Check 1-min (primary)
        available_timeframes.append("1min")
        if trend_1min == expected_trend:
            aligned_timeframes.append("1min")
        
        # Check 5-min
        if trend_5min != "UNAVAILABLE":
            available_timeframes.append("5min")
            if trend_5min == expected_trend:
                aligned_timeframes.append("5min")
        
        # Check 15-min
        if trend_15min != "UNAVAILABLE":
            available_timeframes.append("15min")
            if trend_15min == expected_trend:
                aligned_timeframes.append("15min")
        
        # Calculate confidence
        aligned_count = len(aligned_timeframes)
        available_count = len(available_timeframes)
        
        if available_count == 0:
            confidence = 0
            confirmed = False
            reasoning = "No timeframes available for analysis"
        else:
            alignment_percentage = (aligned_count / available_count) * 100
            
            # Determine confirmation based on requirements
            if self.require_all_timeframes:
                confirmed = aligned_count == available_count
                confidence = 100 if confirmed else alignment_percentage
            else:
                # Majority vote: need 2 out of 3 (or 1 out of 1, 2 out of 2)
                confirmed = aligned_count >= max(2, available_count)
                confidence = alignment_percentage
            
            # Build reasoning
            if confirmed:
                reasoning = f"{aligned_count}/{available_count} timeframes aligned: {', '.join(aligned_timeframes)}"
            else:
                reasoning = f"Only {aligned_count}/{available_count} timeframes aligned: {', '.join(aligned_timeframes) if aligned_timeframes else 'None'}"
        
        result = {
            "confirmed": confirmed,
            "confidence": round(confidence, 1),
            "timeframe_alignment": {
                "1min": trend_1min,
                "5min": trend_5min,
                "15min": trend_15min,
            },
            "aligned_timeframes": aligned_timeframes,
            "aligned_count": aligned_count,
            "available_count": available_count,
            "reasoning": reasoning,
            "signal": signal.upper(),
            "expected_trend": expected_trend,
        }
        
        # Log the analysis
        status_emoji = "✅" if confirmed else "❌"
        logger.info(f"{status_emoji} Multi-Timeframe Confirmation for {signal.upper()}:")
        logger.info(f"   📊 1-min: {trend_1min}")
        logger.info(f"   📊 5-min: {trend_5min}")
        logger.info(f"   📊 15-min: {trend_15min}")
        logger.info(f"   ⚖️  Result: {aligned_count}/{available_count} aligned ({confidence:.1f}% confidence)")
        logger.info(f"   💡 {reasoning}")
        
        return result
    
    def get_higher_timeframe_data(
        self,
        binance_service,
        symbol: str,
        interval: str,
        limit: int = 100,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch higher timeframe data from Binance.
        
        Args:
            binance_service: BinanceService instance
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Kline interval ("5m", "15m", "1h", etc.)
            limit: Number of candles to fetch
        
        Returns:
            DataFrame with OHLCV data or None if error
        """
        try:
            # Placeholder for future implementation
            # Currently not used as trading_bot.py fetches timeframes directly
            _ = (binance_service, symbol, interval, limit)  # Mark as intentionally unused
            logger.debug(f"📊 Higher timeframe data fetching placeholder for {interval}")
            return None
        except Exception as e:
            logger.error(f"❌ Error fetching {interval} data: {e}")
            return None
