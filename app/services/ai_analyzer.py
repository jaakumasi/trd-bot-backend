import google.generativeai as genai
from typing import Dict, Optional
import pandas as pd
import numpy as np
import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..config import settings
from .trade_outcome_tracker import TradeOutcomeTracker
import logging

logger = logging.getLogger(__name__)


class AIAnalyzer:
    def __init__(self):
        try:
            genai.configure(api_key=settings.gemini_api_key)
            # Updated to use the latest available Gemini model
            self.model = genai.GenerativeModel("models/gemini-2.5-flash")
            # Initialize trade outcome tracker for historical learning
            self.trade_tracker = TradeOutcomeTracker()
            logger.info("✅ AI Analyzer initialized with Gemini 2.5 Flash model + TradeOutcomeTracker")
        except Exception as e:
            logger.error(f"❌ Failed to initialize AI Analyzer: {e}")
            self.model = None
            self.trade_tracker = None

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for analysis"""
        if df.empty or len(df) < 20:
            logger.warning(f"⚠️ DataFrame too small for indicators: {len(df)} rows")
            return df

        # Work on a copy to avoid modifying original
        df = df.copy()

        # RSI calculation
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # Moving averages
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = (
            df["close"].rolling(window=50).mean() if len(df) >= 50 else np.nan
        )
        df["ema_12"] = df["close"].ewm(span=12).mean()
        df["ema_26"] = df["close"].ewm(span=26).mean()

        # MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9).mean()

        # Bollinger Bands
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        bb_std = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
        df["bb_lower"] = df["bb_middle"] - (bb_std * 2)

        # Volume indicators
        df["volume_sma"] = df["volume"].rolling(window=20).mean()

        # On-Balance Volume (OBV)
        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

        # Stochastic Oscillator
        low_14 = df["low"].rolling(window=14).min()
        high_14 = df["high"].rolling(window=14).max()
        df["stoch_k"] = 100 * ((df["close"] - low_14) / (high_14 - low_14))
        df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

        # Average True Range (ATR)
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

        # Average Directional Index (ADX) for trend strength
        df = self._calculate_adx(df, period=14)

        return df

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average Directional Index (ADX) for trend strength.
        ADX > 25 indicates strong trend, < 20 indicates weak/ranging market.
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_dm_smooth = pd.Series(plus_dm).rolling(window=period).mean()
        minus_dm_smooth = pd.Series(minus_dm).rolling(window=period).mean()

        # Directional Indicators
        df["plus_di"] = 100 * (plus_dm_smooth / atr)
        df["minus_di"] = 100 * (minus_dm_smooth / atr)

        # ADX Calculation
        dx = (
            100 * abs(df["plus_di"] - df["minus_di"]) / (df["plus_di"] + df["minus_di"])
        )
        df["adx"] = dx.rolling(window=period).mean()

        # Fill NaN values with 0
        df["adx"] = df["adx"].fillna(0)
        df["plus_di"] = df["plus_di"].fillna(0)
        df["minus_di"] = df["minus_di"].fillna(0)

        return df

    async def get_user_trade_history_context(
        self, db: AsyncSession, user_id: int, limit: int = 15
    ) -> Optional[Dict]:
        """
        Retrieve and summarize recent trade performance for a user.
        This provides personalized context to help the AI make better decisions.
        """
        try:
            from ..models.trade import Trade

            # Query last N closed trades for this user
            query = (
                select(Trade)
                .where(Trade.user_id == user_id, Trade.status == "closed")
                .order_by(Trade.closed_at.desc())
                .limit(limit)
            )

            result = await db.execute(query)
            trades = result.scalars().all()

            if not trades or len(trades) == 0:
                logger.debug(f"📊 [User {user_id}] No historical trades found")
                return None

            # Calculate performance metrics
            total_trades = len(trades)
            winning_trades = sum(
                1 for t in trades if t.profit_loss and float(t.profit_loss) > 0
            )
            losing_trades = sum(
                1 for t in trades if t.profit_loss and float(t.profit_loss) < 0
            )

            total_pnl = sum(float(t.profit_loss) for t in trades if t.profit_loss)
            avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

            # Analyze exit reasons
            stop_loss_count = sum(1 for t in trades if t.exit_reason == "STOP_LOSS")
            take_profit_count = sum(1 for t in trades if t.exit_reason == "TAKE_PROFIT")

            # Determine recent trend (last 5 trades)
            recent_trades = trades[:5]
            recent_wins = sum(
                1 for t in recent_trades if t.profit_loss and float(t.profit_loss) > 0
            )
            recent_trend = (
                "winning"
                if recent_wins >= 3
                else "losing" if recent_wins <= 1 else "mixed"
            )

            # Calculate average trade duration
            avg_duration = (
                sum(t.duration_seconds for t in trades if t.duration_seconds)
                / total_trades
                if total_trades > 0
                else 0
            )

            context = {
                "has_history": True,
                "user_id": user_id,  # Include user_id for pattern matching
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "avg_pnl_per_trade": avg_pnl,
                "avg_duration_minutes": avg_duration / 60 if avg_duration > 0 else 0,
                "stop_loss_rate": (
                    (stop_loss_count / total_trades * 100) if total_trades > 0 else 0
                ),
                "take_profit_rate": (
                    (take_profit_count / total_trades * 100) if total_trades > 0 else 0
                ),
                "recent_trend": recent_trend,
                "recent_wins": recent_wins,
                "recent_total": len(recent_trades),
            }

            logger.info(
                f"📊 [User {user_id}] Historical Context: {total_trades} trades, {win_rate:.1f}% win rate, ${avg_pnl:+.2f} avg P&L, trend: {recent_trend}"
            )

            return context

        except Exception as e:
            logger.error(f"❌ Error fetching user trade history: {e}")
            return None

    async def analyze_market_data(
        self,
        symbol: str,
        df: pd.DataFrame,
        user_trade_history: Optional[Dict] = None,
        regime_analysis: Optional[Dict] = None,
    ) -> Dict:
        """Use Gemini AI to analyze market data and provide trading signals."""
        try:
            if self.model is None:
                logger.warning("🤖 AI model not available, using fallback analysis")
                return self._fallback_analysis(
                    "AI model unavailable - using safe default"
                )

            if df.empty or len(df) < 20:
                return self._fallback_analysis("Insufficient data")

            df_with_indicators = self.calculate_technical_indicators(df)
            if df_with_indicators.empty or len(df_with_indicators) < 2:
                return self._fallback_analysis("Insufficient indicator data")

            latest = df_with_indicators.iloc[-1]
            prev = df_with_indicators.iloc[-2]
            self._latest_data = latest

            # Step 0: Check minimum volatility requirement (prevent trading in dead markets)
            if 'atr' in latest.index and 'close' in latest.index:
                atr = latest['atr']
                price = latest['close']
                atr_percentage = (atr / price) * 100 if price > 0 else 0
                logger.debug(f"📊 Current ATR: {atr_percentage:.3f}% - regime analyzer will validate tradability")

            # Step 1: Check trading quality
            if regime_analysis:
                trading_quality = regime_analysis.get("trading_quality_score", 0)
                if trading_quality < 30:
                    logger.info(
                        f"🚫 Low trading quality: {trading_quality}/100 - Market regime: {regime_analysis.get('regime')}"
                    )
                    return self._fallback_analysis(
                        f"Trading quality too low ({trading_quality}/100). "
                        f"Regime: {regime_analysis.get('regime')}, "
                        f"ADX: {regime_analysis.get('trend_strength', 0):.1f}, "
                        f"ATR%: {regime_analysis.get('atr_percentage', 0):.2f}%",
                        float(latest["close"]),
                    )
                
                # Step 1.5: Block mean reversion in strong trends (CRITICAL SAFETY)
                regime = regime_analysis.get("regime", "")
                trading_edge = regime_analysis.get("trading_edge", "")
                trend_strength = regime_analysis.get("trend_strength", 0)
                confidence = regime_analysis.get("confidence", 0)
                
                # DO NOT attempt mean reversion counter-trend trades in strong trending markets
                if trading_edge == "MEAN_REVERSION" and (
                    (regime in ["BEAR_TREND", "BULL_TREND"] and trend_strength > 40) or
                    (regime in ["BEAR_TREND", "BULL_TREND"] and confidence > 70)
                ):
                    logger.warning(
                        "⛔ BLOCKED: Mean reversion in strong trend - "
                        f"Regime: {regime} ({confidence}% confidence), "
                        f"ADX: {trend_strength:.1f}, Edge: {trading_edge}. "
                        "This is a strong directional move, NOT a bounce opportunity!"
                    )
                    return self._fallback_analysis(
                        f"Mean reversion blocked in strong {regime} (ADX: {trend_strength:.1f}). "
                        f"Do not catch falling knives or fight strong trends.",
                        float(latest["close"]),
                    )

            # Step 2: Signal Confluence Check (now with dynamic thresholds!)
            confluence_score = self._calculate_signal_confluence(
                df_with_indicators, regime_analysis
            )

            # Get dynamic threshold from settings (environment-aware)
            # Mainnet uses stricter thresholds for safety
            dynamic_threshold = (
                regime_analysis.get("dynamic_confluence_threshold", settings.get_confluence_threshold())
                if regime_analysis
                else settings.get_confluence_threshold()
            )

            if confluence_score < dynamic_threshold:
                logger.info(
                    f"⚠️ Low signal confluence: {confluence_score}/{dynamic_threshold} "
                    f"(Mode: {settings.get_environment_name()}) - Recommending HOLD"
                )
                return self._fallback_analysis(
                    f"Insufficient signal confluence ({confluence_score}/{dynamic_threshold}). "
                    f"Trend alignment, momentum, volume, or regime not aligned.",
                    float(latest["close"]),
                )

            market_summary = self._build_market_summary(
                symbol, df_with_indicators, latest, prev
            )
            
            # Build enhanced AI context with price action, volatility, and volume analysis
            enhanced_context = self._build_enhanced_ai_context(
                df_with_indicators, sr_levels=None, regime_analysis=regime_analysis
            )
            
            prompt = self._generate_prompt(
                latest, market_summary, user_trade_history, regime_analysis, enhanced_context
            )

            try:
                response = await self.model.generate_content_async(prompt)
                response_text = (response.text or "").strip()
                logger.info(f"🤖 Gemini Full Response: {response.text}")
                ai_analysis = self._parse_model_response(response_text)
            except ValueError as parse_error:
                logger.error(f"❌ JSON Parse Error: {parse_error}")
                logger.error(f"❌ Raw response text: {response_text}")
                return self._fallback_analysis(
                    "AI response parsing failed - invalid JSON format",
                    float(latest.get("close", 0.0) or 0.0),
                )
            except Exception as gemini_error:
                logger.error(f"❌ Gemini API Error: {gemini_error}")
                return self._fallback_analysis(
                    f"AI analysis failed: {str(gemini_error)[:100]}",
                    float(latest.get("close", 0.0) or 0.0),
                )

            sanitized = self._sanitize_analysis(ai_analysis, latest)
            enriched = self._enforce_price_targets(sanitized)

            technical_score = self._calculate_technical_score(latest)
            enriched["technical_score"] = technical_score
            enriched["final_confidence"] = min(enriched["confidence"], technical_score)
            enriched["signal_confluence"] = confluence_score
            enriched["regime_analysis"] = regime_analysis  # Include for risk validation
            
            # Require momentum confirmation for counter-trend entries
            signal = enriched.get("signal", "hold").lower()
            rsi = float(latest.get("rsi", 50))
            
            # For BUY signals, RSI must be turning UP (not just oversold)
            if signal == "buy" and rsi < 30:
                # Check if RSI is starting to turn up
                if len(df) >= 3:
                    rsi_values = df['rsi'].tail(3).values
                    rsi_current = rsi_values[-1]
                    rsi_prev = rsi_values[-2]
                    
                    # RSI must be turning UP, not just oversold
                    if rsi_current <= rsi_prev:
                        logger.warning(
                            f"⛔ BLOCKED BUY: RSI={rsi:.1f} still falling (no momentum confirmation). "
                            f"Wait for RSI to turn back UP above 30 before buying oversold conditions."
                        )
                        return self._fallback_analysis(
                            f"BUY blocked - RSI={rsi:.1f} still declining, no momentum reversal confirmed",
                            float(latest["close"]),
                        )
                    else:
                        logger.info(
                            f"✅ BUY momentum confirmed: RSI turning up {rsi_prev:.1f}→{rsi_current:.1f}"
                        )
            
            # For SELL signals, RSI must be turning DOWN (not just overbought)
            if signal == "sell" and rsi > 70:
                if len(df) >= 3:
                    rsi_values = df['rsi'].tail(3).values
                    rsi_current = rsi_values[-1]
                    rsi_prev = rsi_values[-2]
                    
                    if rsi_current >= rsi_prev:
                        logger.warning(
                            f"⛔ BLOCKED SELL: RSI={rsi:.1f} still rising (no momentum confirmation). "
                            f"Wait for RSI to turn back DOWN below 70 before shorting overbought conditions."
                        )
                        return self._fallback_analysis(
                            f"SELL blocked - RSI={rsi:.1f} still rising, no momentum reversal confirmed",
                            float(latest["close"]),
                        )
                    else:
                        logger.info(
                            f"✅ SELL momentum confirmed: RSI turning down {rsi_prev:.1f}→{rsi_current:.1f}"
                        )

            # CRITICAL: Enforce directional bias in strong trends
            if regime_analysis:
                signal = enriched.get("action", "hold").lower()
                regime = regime_analysis.get("regime", "")
                trend_strength = regime_analysis.get("trend_strength", 0)
                confidence = regime_analysis.get("confidence", 0)
                
                # In strong bearish trends, BLOCK buy signals (don't catch falling knives)
                if signal == "buy" and regime == "BEAR_TREND" and (
                    trend_strength > 50 or confidence > 75
                ):
                    logger.warning(
                        f"⛔ BLOCKED: BUY signal in strong BEAR_TREND - "
                        f"ADX: {trend_strength:.1f}, Confidence: {confidence}%. "
                        f"Do NOT buy into strong downtrends!"
                    )
                    return self._fallback_analysis(
                        f"BUY blocked in strong bearish market (ADX: {trend_strength:.1f})",
                        float(latest["close"]),
                    )
                
                # In strong bullish trends, BLOCK sell signals
                if signal == "sell" and regime == "BULL_TREND" and (
                    trend_strength > 50 or confidence > 75
                ):
                    logger.warning(
                        f"⛔ BLOCKED: SELL signal in strong BULL_TREND - "
                        f"ADX: {trend_strength:.1f}, Confidence: {confidence}%. "
                        f"Do NOT short into strong uptrends!"
                    )
                    return self._fallback_analysis(
                        f"SELL blocked in strong bullish market (ADX: {trend_strength:.1f})",
                        float(latest["close"]),
                    )

            logger.info(f"🎯 Final AI Analysis: {enriched}")
            return enriched

        except Exception as error:
            logger.error(f"AI analysis error: {error}")
            return self._fallback_analysis(f"Analysis error: {str(error)[:50]}")

    def _describe_last_5_candles(self, df: pd.DataFrame) -> str:
        """
        Analyze last 5 candles for price action quality.
        Returns natural language description for AI prompt.
        """
        if len(df) < 5:
            return "Insufficient data for candle analysis"
        
        last_5 = df.tail(5)
        descriptions = []
        
        for idx, candle in last_5.iterrows():
            body_size = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            body_pct = (body_size / total_range * 100) if total_range > 0 else 0
            
            direction = "bullish" if candle['close'] > candle['open'] else "bearish"
            
            # Detect wicks
            if candle['close'] > candle['open']:  # Bull candle
                upper_wick = candle['high'] - candle['close']
                lower_wick = candle['open'] - candle['low']
            else:  # Bear candle
                upper_wick = candle['high'] - candle['open']
                lower_wick = candle['close'] - candle['low']
            
            upper_wick_pct = (upper_wick / total_range * 100) if total_range > 0 else 0
            lower_wick_pct = (lower_wick / total_range * 100) if total_range > 0 else 0
            
            # Classify candle
            if body_pct > 70:
                strength = "strong"
            elif body_pct > 40:
                strength = "moderate"
            elif body_pct > 20:
                strength = "weak"
            else:
                strength = "doji-like"
            
            # Check for rejection
            rejection = ""
            if upper_wick_pct > 40:
                rejection = " with upper rejection"
            elif lower_wick_pct > 40:
                rejection = " with lower rejection"
            
            descriptions.append(f"{strength} {direction}{rejection}")
        
        return " | ".join(descriptions)

    def _score_candle_quality(self, df: pd.DataFrame) -> Dict:
        """
        Score overall candle quality for trend strength assessment.
        """
        if len(df) < 5:
            return {"quality_score": 50, "description": "insufficient_data"}
        
        last_5 = df.tail(5)
        total_quality = 0
        bullish_count = 0
        bearish_count = 0
        
        for idx, candle in last_5.iterrows():
            body_size = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            body_pct = (body_size / total_range * 100) if total_range > 0 else 0
            
            # Strong body = higher quality
            if body_pct > 70:
                total_quality += 20
            elif body_pct > 50:
                total_quality += 15
            elif body_pct > 30:
                total_quality += 10
            else:
                total_quality += 5
            
            if candle['close'] > candle['open']:
                bullish_count += 1
            else:
                bearish_count += 1
        
        # Directional consistency bonus
        consistency = max(bullish_count, bearish_count)
        if consistency >= 4:
            total_quality += 10  # Very consistent
        elif consistency == 3:
            total_quality += 5  # Somewhat consistent
        
        direction = "bullish" if bullish_count > bearish_count else "bearish" if bearish_count > bullish_count else "mixed"
        
        return {
            "quality_score": min(100, total_quality),
            "direction": direction,
            "consistency": f"{consistency}/5"
        }

    def _detect_atr_trend(self, df: pd.DataFrame) -> str:
        """
        Detect if ATR is expanding (volatility increasing) or contracting.
        """
        if len(df) < 20 or 'atr' not in df.columns:
            return "unknown"
        
        atr_values = df['atr'].tail(20)
        recent_atr = atr_values.tail(5).mean()
        earlier_atr = atr_values.iloc[:10].mean()
        
        if pd.isna(recent_atr) or pd.isna(earlier_atr) or earlier_atr == 0:
            return "stable"
        
        change_pct = (recent_atr - earlier_atr) / earlier_atr * 100
        
        if change_pct > 15:
            return "expanding_rapidly"
        elif change_pct > 5:
            return "expanding"
        elif change_pct < -15:
            return "contracting_rapidly"
        elif change_pct < -5:
            return "contracting"
        else:
            return "stable"

    def _analyze_volume_profile(self, df: pd.DataFrame) -> Dict:
        """
        Analyze volume profile for buying vs selling pressure.
        """
        if len(df) < 10 or 'volume' not in df.columns:
            return {"pressure": "unknown", "ratio": 1.0}
        
        last_10 = df.tail(10)
        
        buying_volume = 0
        selling_volume = 0
        
        for idx, candle in last_10.iterrows():
            if candle['close'] > candle['open']:  # Bull candle = buying
                buying_volume += candle['volume']
            else:  # Bear candle = selling
                selling_volume += candle['volume']
        
        total_volume = buying_volume + selling_volume
        if total_volume == 0:
            return {"pressure": "neutral", "ratio": 1.0}
        
        buy_ratio = buying_volume / total_volume
        
        if buy_ratio > 0.65:
            pressure = "strong_buying"
        elif buy_ratio > 0.55:
            pressure = "moderate_buying"
        elif buy_ratio < 0.35:
            pressure = "strong_selling"
        elif buy_ratio < 0.45:
            pressure = "moderate_selling"
        else:
            pressure = "neutral"
        
        return {
            "pressure": pressure,
            "buy_ratio": buy_ratio,
            "sell_ratio": 1 - buy_ratio
        }

    def _build_enhanced_ai_context(
        self,
        df: pd.DataFrame,
        sr_levels: Optional[Dict] = None,
        regime_analysis: Optional[Dict] = None
    ) -> str:
        """
        Build comprehensive market structure context for AI prompt.
        Includes: price action, S/R proximity, ATR trend, volume profile, candle quality, regime state.
        """
        context_parts = []
        
        # Price Action Analysis
        candle_desc = self._describe_last_5_candles(df)
        context_parts.append(f"Recent Price Action (Last 5 Candles): {candle_desc}")
        
        # Candle Quality
        quality = self._score_candle_quality(df)
        context_parts.append(
            f"Candle Quality: {quality['quality_score']}/100 "
            f"(Direction: {quality['direction']}, Consistency: {quality['consistency']})"
        )
        
        # ATR Trend
        atr_trend = self._detect_atr_trend(df)
        context_parts.append(f"Volatility Trend (ATR): {atr_trend.replace('_', ' ').title()}")
        
        # Volume Profile
        volume = self._analyze_volume_profile(df)
        if volume['pressure'] != "unknown":
            context_parts.append(
                f"Volume Profile: {volume['pressure'].replace('_', ' ').title()} "
                f"(Buy: {volume['buy_ratio']*100:.0f}%, Sell: {volume['sell_ratio']*100:.0f}%)"
            )
        
        # Market Regime Context (using regime_analysis parameter)
        if regime_analysis:
            regime = regime_analysis.get('regime', 'UNKNOWN')
            trend_strength = regime_analysis.get('trend_strength', 0)
            trading_edge = regime_analysis.get('trading_edge', 'NEUTRAL')
            trading_quality = regime_analysis.get('trading_quality_score', 0)
            
            context_parts.append(
                f"Market Regime: {regime} (ADX: {trend_strength:.1f}, "
                f"Edge: {trading_edge}, Quality: {trading_quality}/100)"
            )
            
            # Add regime-specific guidance
            if regime in ['BULL_TREND', 'BEAR_TREND'] and trend_strength > 25:
                context_parts.append(
                    f"⚠️ Strong {regime.split('_')[0].lower()} trend detected - "
                    f"favor {('long' if regime == 'BULL_TREND' else 'short')} positions"
                )
            elif regime == 'RANGE_BOUND':
                context_parts.append(
                    "⚠️ Range-bound market - consider mean reversion strategies at S/R levels"
                )
            elif regime in ['HIGH_VOL', 'LOW_VOL']:
                context_parts.append(
                    f"⚠️ {regime.replace('_', ' ')} environment - adjust position sizing accordingly"
                )
        
        # Support/Resistance Proximity
        if sr_levels and 'support' in sr_levels and 'resistance' in sr_levels:
            current_price = float(df.iloc[-1]['close'])
            support = sr_levels['support']
            resistance = sr_levels['resistance']
            
            dist_to_support = (current_price - support) / current_price * 100
            dist_to_resistance = (resistance - current_price) / current_price * 100
            
            context_parts.append(
                f"S/R Proximity: Support {dist_to_support:.2f}% below, "
                f"Resistance {dist_to_resistance:.2f}% above"
            )
        
        return "\n".join(context_parts)

    def _build_market_summary(
        self,
        symbol: str,
        df: pd.DataFrame,
        latest: pd.Series,
        prev: pd.Series,
    ) -> Dict:
        price_change = float(
            (latest["close"] - df.iloc[-12]["close"]) / df.iloc[-12]["close"] * 100
        )
        volume_ratio = (
            float(latest["volume"] / latest["volume_sma"])
            if not pd.isna(latest["volume_sma"])
            else 1.0
        )

        summary = {
            "symbol": symbol,
            "current_price": float(latest["close"]),
            "price_change_1h": price_change,
            "volume_ratio": volume_ratio,
            "rsi": float(latest["rsi"]) if not pd.isna(latest["rsi"]) else 50.0,
            "macd": self._extract_macd(latest),
            "bollinger_bands": {
                "position": self._determine_bollinger_position(latest),
                "squeeze": self._calculate_bollinger_squeeze(latest),
            },
            "moving_averages": {
                "price_vs_sma20": self._relative_to_sma(latest),
                "sma20_trend": self._sma_trend(latest, prev),
            },
        }
        return summary

    def _calculate_signal_confluence(
        self, df: pd.DataFrame, regime_analysis: Optional[Dict]
    ) -> int:
        """
        Enhanced multi-factor signal confluence scoring with pullback quality reward.

        Uses these CORE factors:
        - Trend Alignment (25 points): Is price action aligned with the broader trend?
        - Pullback Quality (15 points): Rewards optimal entries at value zones
        - Momentum Confirmation (25 points): Is momentum supporting the desired direction?
        - Volume Support (15 points): Is volume confirming the move?
        - Regime Appropriateness (20 points): Is the current market regime suitable for this trade?
        
        Returns: 0-100 score (>60 required for trade)
        """
        score = 0
        latest = df.iloc[-1]

        # Factor 1: Trend Alignment (25 points)
        sma_20 = latest.get("sma_20", np.nan)
        sma_50 = latest.get("sma_50", np.nan)
        price = latest.get("close", 0)

        if not pd.isna(sma_20) and not pd.isna(sma_50):
            # Identify trend direction
            bullish_trend = sma_20 > sma_50
            bearish_trend = sma_20 < sma_50
            
            if bullish_trend and price > sma_50:  # Bullish structure (price above both MAs or in pullback zone)
                score += 25
            elif bearish_trend and price < sma_50:  # Bearish structure
                score += 25
            elif bullish_trend and price > sma_20:  # Weaker bullish (only above 20 SMA)
                score += 15
            elif bearish_trend and price < sma_20:  # Weaker bearish
                score += 15

        # Factor 2: Pullback Quality (15 points)
        # Reward optimal entries when price pulls back to value in a trend
        if not pd.isna(sma_20) and not pd.isna(sma_50):
            bullish_trend = sma_20 > sma_50
            bearish_trend = sma_20 < sma_50
            
            # Calculate distance from price to MAs
            distance_to_20 = abs(price - sma_20) / price if price > 0 else 1
            
            # BULLISH PULLBACK: Price between 20 and 50 SMA in uptrend (optimal buy zone)
            if bullish_trend and sma_50 <= price <= sma_20:
                pullback_quality = 15  # Maximum points for perfect pullback
                score += pullback_quality
                logger.debug(
                    f"✅ BULLISH PULLBACK DETECTED: Price ${price:.2f} in value zone "
                    f"(${sma_50:.2f} to ${sma_20:.2f}) - BONUS +{pullback_quality} points"
                )
            # BULLISH NEAR PULLBACK: Price slightly below 20 SMA (still good entry)
            elif bullish_trend and price < sma_20 and distance_to_20 < 0.01:  # Within 1% of 20 SMA
                pullback_quality = 10
                score += pullback_quality
                logger.debug(f"✅ Near 20 SMA in uptrend - BONUS +{pullback_quality} points")
            
            # BEARISH PULLBACK: Price between 50 and 20 SMA in downtrend (optimal short zone)
            elif bearish_trend and sma_20 <= price <= sma_50:
                pullback_quality = 15
                score += pullback_quality
                logger.debug(
                    f"✅ BEARISH PULLBACK DETECTED: Price ${price:.2f} in value zone "
                    f"(${sma_20:.2f} to ${sma_50:.2f}) - BONUS +{pullback_quality} points"
                )
            # BEARISH NEAR PULLBACK: Price slightly above 20 SMA
            elif bearish_trend and price > sma_20 and distance_to_20 < 0.01:
                pullback_quality = 10
                score += pullback_quality
                logger.debug(f"✅ Near 20 SMA in downtrend - BONUS +{pullback_quality} points")
            
            # PENALTY: Price too extended from trend (chase entry)
            elif bullish_trend and price > sma_20:
                extension = (price - sma_20) / sma_20
                if extension > 0.02:  # More than 2% extended above 20 SMA
                    score -= 5  # Small penalty for chasing
                    logger.debug(
                        "⚠️ Price extended %.1f%% above 20 SMA - chase entry",
                        extension * 100
                    )
            elif bearish_trend and price < sma_20:
                extension = (sma_20 - price) / sma_20
                if extension > 0.02:
                    score -= 5
                    logger.debug(
                        "⚠️ Price extended %.1f%% below 20 SMA - chase entry",
                        extension * 100
                    )

        # Factor 3: Momentum Confirmation (25 points)
        rsi = latest.get("rsi", 50)
        macd = latest.get("macd", 0)
        macd_signal = latest.get("macd_signal", 0)

        if not pd.isna(rsi) and not pd.isna(macd):
            if macd > macd_signal and rsi > 50:  # Bullish Momentum
                score += 25
            elif macd < macd_signal and rsi < 50:  # Bearish Momentum
                score += 25
            elif macd > macd_signal or rsi > 50:  # Partial Bullish
                score += 12
            elif macd < macd_signal or rsi < 50:  # Partial Bearish
                score += 12

        # Factor 4: Volume Support (15 points)
        volume = latest.get("volume", 0)
        volume_sma = latest.get("volume_sma", 0)
        if not pd.isna(volume) and not pd.isna(volume_sma) and volume_sma > 0:
            volume_ratio = volume / volume_sma
            if volume_ratio > 1.5:  # High volume confirmation
                score += 15
            elif volume_ratio > 1.1:  # Moderate volume confirmation
                score += 8

        # Factor 5: Regime Appropriateness (20 points)
        if regime_analysis:
            trading_quality = regime_analysis.get("trading_quality_score")
            if trading_quality is not None:
                # Quality 0-100 maps to 0-20 points
                score += int(trading_quality * 0.2)
            else:
                # Fallback for older regime analyzer
                regime = regime_analysis.get("regime")
                if regime in ["BULL_TREND", "BEAR_TREND"]:
                    score += 20
                elif regime == "RANGE_BOUND":
                    score += 10

        final_score = max(0, min(100, score))

        logger.info(f"📊 Signal Confluence Score: {final_score}/100 (5-factor model with pullback quality)")

        return final_score

    def _build_pattern_aware_history_context(
        self,
        user_id: int,
        current_setup: Dict,
        user_trade_history: Optional[Dict] = None
    ) -> str:
        """
        Build pattern-aware historical context using TradeOutcomeTracker.
        This replaces basic win/loss stats with actionable pattern insights.
        
        Args:
            user_id: User ID for filtering trades to this user only
            current_setup: Current market setup dictionary
            user_trade_history: Optional basic trade history dict
        
        Returns natural language context for AI prompt with:
        - Calibration data (AI confidence vs actual win rate)
        - Failing pattern warnings
        - Regime-specific performance
        - Historical context from tracker
        """
        if not self.trade_tracker:
            logger.warning("⚠️ TradeOutcomeTracker not available, using basic history")
            return self._build_basic_history_context(user_trade_history)
        
        context_parts = []
        
        # Get calibration curve for this user
        calibration = self.trade_tracker.get_calibration_curve(user_id=user_id)
        if calibration:
            context_parts.append("AI CONFIDENCE CALIBRATION:")
            context_parts.append(
                "The AI's confidence scores are calibrated against actual outcomes:"
            )
            for conf_bucket, actual_wr in sorted(calibration.items()):
                context_parts.append(f"  - {conf_bucket}% confidence → {actual_wr:.1f}% actual win rate")
            
            # Warning if AI is overconfident
            high_conf_buckets = [k for k in calibration.keys() if k >= 80]
            if high_conf_buckets:
                avg_high_conf_wr = sum(calibration[k] for k in high_conf_buckets) / len(high_conf_buckets)
                if avg_high_conf_wr < 70:
                    context_parts.append(
                        f"⚠️ WARNING: High confidence predictions (80%+) only win {avg_high_conf_wr:.1f}% of the time. "
                        "Be more conservative with confidence scores."
                    )
        
        # Get failing patterns for this user
        failing_patterns = self.trade_tracker.identify_failing_patterns(user_id=user_id, min_occurrences=3)
        if failing_patterns:
            context_parts.append("\nRECURRING FAILURE PATTERNS (AVOID THESE):")
            for pattern in failing_patterns[:3]:  # Top 3 patterns
                context_parts.append(
                    f"  - {pattern['pattern_type']}: {pattern['occurrences']} trades, "
                    f"{pattern['win_rate']:.1f}% win rate, "
                    f"avg loss ${pattern['avg_pnl']:+.2f}"
                )
                context_parts.append(f"    Description: {pattern['description']}")
        
        # Get regime-specific performance for this user
        regime_perf = self.trade_tracker.get_regime_performance(user_id=user_id)
        if regime_perf:
            context_parts.append("\nPERFORMANCE BY MARKET REGIME:")
            for regime, stats in regime_perf.items():
                if stats['trade_count'] >= 5:  # Only show regimes with sufficient data
                    context_parts.append(
                        f"  - {regime}: {stats['win_rate']:.1f}% win rate "
                        f"({stats['trade_count']} trades, avg P&L ${stats['avg_pnl']:+.2f})"
                    )
        
        # Get historical context specific to current setup for this user
        tracker_context = self.trade_tracker.get_historical_context_for_ai(current_setup, user_id=user_id)
        if tracker_context:
            context_parts.append("\nSIMILAR PAST SETUPS:")
            context_parts.append(tracker_context)
        
        # Add basic stats if available
        if user_trade_history and user_trade_history.get("has_history"):
            h = user_trade_history
            context_parts.append("\nOVERALL STATISTICS:")
            context_parts.append(
                f"  - Last {h['total_trades']} trades: {h['win_rate']:.1f}% win rate, "
                f"${h['total_pnl']:+.2f} total P&L"
            )
            context_parts.append(
                f"  - Recent trend: {h['recent_trend'].upper()} "
                f"({h['recent_wins']}/{h['recent_total']} wins in last 5 trades)"
            )
        
        if not context_parts:
            return ""
        
        full_context = "\n".join(context_parts)
        
        instructions = """
        
        INSTRUCTIONS FOR USING HISTORICAL CONTEXT:
        - If calibration shows overconfidence (predicted > actual by 10%+): Reduce your confidence by 10-15 points
        - If current setup matches a failing pattern: Consider HOLD or reduce confidence by 20+ points
        - If regime performance is poor (<40% win rate): Be extra cautious in this regime
        - Use similar past setups to validate or challenge your technical analysis
        - Focus on avoiding repeated mistakes, not on over-penalizing recent losses
        """
        
        return full_context + instructions
    
    def _build_basic_history_context(self, user_trade_history: Optional[Dict]) -> str:
        """Fallback basic history context when tracker unavailable."""
        if not user_trade_history or not user_trade_history.get("has_history"):
            return ""
        
        h = user_trade_history
        return f"""
        User's Recent Trading Performance (Last {h['total_trades']} Trades):
        - Win Rate: {h['win_rate']:.1f}% ({h['winning_trades']} wins, {h['losing_trades']} losses)
        - Average P&L per Trade: ${h['avg_pnl_per_trade']:+.2f}
        - Stop Loss Hit Rate: {h['stop_loss_rate']:.1f}%
        - Recent Trend: {h['recent_trend'].upper()} ({h['recent_wins']} wins in last {h['recent_total']} trades)
        """

    def _generate_prompt(
        self,
        latest: pd.Series,
        market_summary: Dict,
        user_trade_history: Optional[Dict] = None,
        regime_analysis: Optional[Dict] = None,
        enhanced_context: Optional[str] = None,
    ) -> str:
        history_context = ""
        
        if user_trade_history and user_trade_history.get("has_history"):
            # Build current setup for pattern matching
            current_setup = {
                "rsi": float(latest.get("rsi", 50)),
                "regime": regime_analysis.get("regime") if regime_analysis else "UNKNOWN",
                "confidence": market_summary.get("confidence", 50),
                "atr_percentage": regime_analysis.get("atr_percentage", 0) if regime_analysis else 0,
                "trend_strength": regime_analysis.get("trend_strength", 0) if regime_analysis else 0,
            }
            
            # Extract user_id for per-user tracking
            user_id = user_trade_history.get("user_id", 0)
            
            # Use pattern-aware context (replaces basic stats)
            history_context = self._build_pattern_aware_history_context(
                user_id, current_setup, user_trade_history
            )
        
        # Build regime context section if available
        regime_context = ""
        if regime_analysis:
            r = regime_analysis
            regime_context = f"""
            Market Regime Analysis:
            - Current Regime: {r.get('regime', 'UNKNOWN')}
            - Trend Strength (ADX): {r.get('trend_strength', 0):.1f} (>25 = strong trend)
            - Volatility Percentile: {r.get('volatility_percentile', 50):.0f}% (percentile among last 100 periods)
            - ATR Percentage: {r.get('atr_percentage', 0):.2f}% (current volatility level)
            - SMA Alignment: {r.get('sma_alignment', 'NEUTRAL')}
            - Volume Trend: {r.get('volume_trend', 'STABLE')}
            - Trading Status: {'ALLOWED' if r.get('allow_trading') else 'BLOCKED'}

            REGIME-BASED TRADING RULES:
            - BULL_TREND/BEAR_TREND with ADX>25: Favorable for day trading - trade in direction of trend
            - RANGE_BOUND: Not suitable for day trading - recommend 'hold' unless exceptionally strong signal
            - HIGH_VOLATILITY: Too risky for day trading - recommend 'hold' to avoid whipsaw
            - LOW_VOLATILITY: Insufficient movement - recommend 'hold' to wait for better opportunity
            
            Respect the regime classification. If trading is BLOCKED, only recommend 'hold' unless there's an exceptionally clear setup (confidence >90).
            """

        # Fix #5: Add microstructure signals to regular prompt too
        microstructure_context = ""
        if regime_analysis:
            microstructure_context = f"""
            ADVANCED MICROSTRUCTURE SIGNALS (Fix #5 Enhancement):
            - Momentum Persistence: {regime_analysis.get('momentum_persistence', 50):.0f}/100 
            - Order Flow Imbalance: {regime_analysis.get('order_flow_imbalance', 0):+.0f}/100 
            - Mean Reversion Score: {regime_analysis.get('mean_reversion_score', 0):.0f}/100 
            - Trading Quality Score: {regime_analysis.get('trading_quality_score', 0):.0f}/100
            - Trading Edge: {regime_analysis.get('trading_edge', 'NEUTRAL')}
            """
        
        enhanced_market_context = ""
        if enhanced_context:
            enhanced_market_context = f"""
            ENHANCED MARKET STRUCTURE ANALYSIS:
            {enhanced_context}
            
            INSTRUCTIONS FOR USING ENHANCED CONTEXT:
            - Price Action: Look for rejection wicks, strong bodies, and directional consistency
            - Candle Quality: Higher scores (>70) indicate conviction; low scores (<40) suggest indecision
            - Volatility Trend: Expanding ATR = momentum building; Contracting = consolidation/reversal
            - Volume Profile: Strong buying/selling pressure confirms directional moves
            - Use this context to validate or challenge basic indicator signals (RSI, MACD)
            """
        
        return f"""
            You are an expert cryptocurrency day trading analyst. Analyze this BTC/USDT market data for a 1-minute day trading strategy:

            Market Data:
            {json.dumps(market_summary, indent=2)}
            
            {history_context}
            {regime_context}
            {microstructure_context}
            {enhanced_market_context}
            Day Trading Strategy Context:
            - Target: 0.3% profit per trade
            - Stop loss: 0.5% maximum loss
            - Risk tolerance: Very conservative (1% account risk)
            - Trading timeframe: 1-5 minutes
            - Market session: 24/7 trading enabled

            Based on this data, provide a JSON response with ALL 6 required fields:
            1. "signal": "buy", "sell", or "hold" 
            2. "confidence": integer from 0-100
            3. "reasoning": brief explanation (max 100 words)
            4. "entry_price": REQUIRED - suggested entry price (use current price {latest['close']:.2f} if unsure)
            5. "stop_loss": REQUIRED - suggested stop loss price (0.3% from entry for fast execution)
            6. "take_profit": REQUIRED - suggested take profit price (0.5% from entry for fast execution)

            Consider:
            - RSI overbought (>70) or oversold (<30) conditions
            - MACD momentum and divergence
            - Volume confirmation
            - Bollinger Band squeeze/expansion
            - Price action relative to moving averages

            CRITICAL: You MUST include ALL 6 fields (signal, confidence, reasoning, entry_price, stop_loss, take_profit) in your response.
            Return ONLY valid JSON without markdown code blocks or any other formatting.

            Example format with ALL required fields:
            {{"signal": "buy", "confidence": 75, "reasoning": "Strong bullish momentum with structure support", "entry_price": {latest['close']:.2f}, "stop_loss": {latest['close'] * 0.995:.2f}, "take_profit": {latest['close'] * 1.01:.2f}}}
            
            IMPORTANT: Always provide stop_loss and take_profit based on nearby support/resistance levels, NOT fixed percentages.
            """

    @staticmethod
    def _determine_bollinger_position(latest: pd.Series) -> str:
        upper = latest.get("bb_upper")
        lower = latest.get("bb_lower")
        middle = latest.get("bb_middle")
        close_price = latest.get("close")

        if any(pd.isna(val) for val in (upper, lower, middle, close_price)):
            return "middle"

        if close_price > upper:
            return "above_upper"
        if close_price < lower:
            return "below_lower"
        if close_price > middle:
            return "upper_half"
        return "lower_half"

    @staticmethod
    def _calculate_bollinger_squeeze(latest: pd.Series) -> float:
        upper = latest.get("bb_upper")
        lower = latest.get("bb_lower")
        middle = latest.get("bb_middle")
        if any(pd.isna(val) for val in (upper, lower, middle)) or middle == 0:
            return 0.0
        return float((upper - lower) / middle * 100)

    @staticmethod
    def _extract_macd(latest: pd.Series) -> Dict:
        macd_value = latest.get("macd")
        macd_signal = latest.get("macd_signal")
        histogram = 0.0
        if not pd.isna(macd_value) and not pd.isna(macd_signal):
            histogram = float(macd_value - macd_signal)
        return {
            "macd": float(macd_value) if not pd.isna(macd_value) else 0.0,
            "signal": float(macd_signal) if not pd.isna(macd_signal) else 0.0,
            "histogram": histogram,
        }

    @staticmethod
    def _relative_to_sma(latest: pd.Series) -> float:
        sma_20 = latest.get("sma_20")
        if pd.isna(sma_20) or sma_20 == 0:
            return 0.0
        return float((latest["close"] - sma_20) / sma_20 * 100)

    @staticmethod
    def _sma_trend(latest: pd.Series, prev: pd.Series) -> str:
        sma_latest = latest.get("sma_20")
        sma_prev = prev.get("sma_20")
        if pd.isna(sma_latest) or pd.isna(sma_prev):
            return "neutral"
        if sma_latest > sma_prev:
            return "up"
        if sma_latest < sma_prev:
            return "down"
        return "neutral"

    def _parse_model_response(self, response_text: str) -> Dict:
        if not response_text:
            raise ValueError("Empty AI response")

        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if cleaned.startswith("```json") or cleaned.startswith("```JSON"):
                lines = lines[1:-1]
            else:
                lines = lines[1:-1]
            cleaned = "\n".join(lines).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(exc) from exc

    def _sanitize_analysis(self, ai_analysis: Dict, latest: pd.Series) -> Dict:
        analysis = dict(ai_analysis or {})
        signal = str(analysis.get("signal", "hold")).lower()
        if signal not in {"buy", "sell", "hold"}:
            logger.warning(f"⚠️ Invalid signal '{signal}', defaulting to 'hold'")
            signal = "hold"
        analysis["signal"] = signal

        confidence_raw = analysis.get("confidence", 0)
        try:
            confidence = int(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0
        analysis["confidence"] = max(0, min(100, confidence))

        current_price = float(latest.get("close", 0.0) or 0.0)
        entry_price = self._safe_float(analysis.get("entry_price"), current_price)
        if entry_price != analysis.get("entry_price"):
            logger.info(f"🔧 Added fallback entry_price: {entry_price}")
        analysis["entry_price"] = entry_price

        if not self._is_positive_number(analysis.get("stop_loss")):
            logger.info(f"🔧 Added fallback stop_loss for {signal} signal")
            analysis["stop_loss"] = entry_price
        else:
            analysis["stop_loss"] = float(analysis["stop_loss"])

        if not self._is_positive_number(analysis.get("take_profit")):
            logger.info(f"🔧 Added fallback take_profit for {signal} signal")
            analysis["take_profit"] = entry_price
        else:
            analysis["take_profit"] = float(analysis["take_profit"])

        analysis.setdefault("reasoning", "No reasoning provided")
        return analysis

    def _enforce_price_targets(self, analysis: Dict) -> Optional[Dict]:
        """
        Validate AI provided stop_loss and take_profit.
        Returns None if missing - DO NOT use bad fallback values.
        """
        entry_price = float(analysis.get("entry_price", 0))
        stop_loss = analysis.get("stop_loss")
        take_profit = analysis.get("take_profit")
        
        if not entry_price or entry_price <= 0:
            logger.error("❌ AI did not provide valid entry_price - SKIPPING TRADE")
            return None
        
        if not stop_loss or stop_loss <= 0:
            logger.error("❌ AI did not provide valid stop_loss - SKIPPING TRADE")
            return None
        
        if not take_profit or take_profit <= 0:
            logger.error("❌ AI did not provide valid take_profit - SKIPPING TRADE")
            return None
        
        logger.debug(
            "✅ AI provided complete targets | entry=%.2f | SL=%.2f | TP=%.2f",
            entry_price,
            stop_loss,
            take_profit,
        )
        return analysis

    def _fallback_analysis(self, reason: str, price: float | None = None) -> Dict:
        """
        Return safe fallback that will be skipped by validation.
        DO NOT provide stop_loss/take_profit - let validation reject it.
        """
        fallback_price = (
            price if price and price > 0 else self._resolve_fallback_price()
        )
        logger.warning(f"🚫 Fallback analysis triggered: {reason} - trade will be skipped")
        return {
            "signal": "hold",
            "confidence": 0,
            "reasoning": f"FALLBACK: {reason}",
            "technical_score": 0,
            "entry_price": fallback_price,
            # NO stop_loss/take_profit - will fail validation and be skipped
        }

    def _resolve_fallback_price(self) -> float:
        fallback_price = 50_000.0
        try:
            if hasattr(self, "_latest_data") and self._latest_data is not None:
                fallback_price = float(self._latest_data.get("close", fallback_price))
        except Exception:
            pass
        return fallback_price

    @staticmethod
    def _safe_float(value, default: float) -> float:
        try:
            candidate = float(value)
            return candidate if candidate > 0 else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _is_positive_number(value) -> bool:
        try:
            return float(value) > 0
        except (TypeError, ValueError):
            return False

    def _calculate_technical_score(self, latest_data) -> int:
        """Calculate technical analysis confidence score"""
        score = 50  # Base score

        try:
            # RSI scoring
            rsi = latest_data["rsi"]
            if not pd.isna(rsi):
                if 30 <= rsi <= 70:  # Neutral zone
                    score += 10
                elif rsi < 30:  # Oversold - potential buy
                    score += 15
                elif rsi > 70:  # Overbought - potential sell
                    score += 15

            # MACD scoring
            if not pd.isna(latest_data["macd"]) and not pd.isna(
                latest_data["macd_signal"]
            ):
                macd_histogram = latest_data["macd"] - latest_data["macd_signal"]
                if abs(macd_histogram) > 0.0001:  # Strong momentum
                    score += 15

            # Volume confirmation
            if not pd.isna(latest_data["volume_sma"]):
                volume_ratio = latest_data["volume"] / latest_data["volume_sma"]
                if volume_ratio > 1.2:  # Above average volume
                    score += 10

            return min(100, max(0, score))

        except Exception as e:
            logger.error(f"Technical score calculation error: {e}")
            return 50

    async def analyze_market_data_mtf(
        self,
        symbol: str,
        mtf_data: Dict[str, pd.DataFrame],
        user_trade_history: Optional[Dict] = None,
        regime_analysis: Optional[Dict] = None,
    ) -> Dict:
        """
        Multi-timeframe market analysis for day trading.

        Analyzes three timeframes:
        - Primary (15m): Main trading signals and entry points
        - Context (1h): Overall trend direction and market structure
        - Precision (5m): Fine-tuned entry timing

        Returns enhanced signal with multi-timeframe confirmation.
        """
        try:
            if self.model is None:
                logger.warning("🤖 AI model not available, using fallback analysis")
                return self._fallback_analysis("AI model unavailable")

            primary_df = mtf_data.get("primary")
            context_df = mtf_data.get("context")
            precision_df = mtf_data.get("precision")

            if primary_df is None or primary_df.empty or len(primary_df) < 20:
                return self._fallback_analysis("Insufficient primary timeframe data")

            # Extract latest data from all timeframes
            primary_latest = primary_df.iloc[-1]
            context_latest = (
                context_df.iloc[-1]
                if context_df is not None and not context_df.empty
                else primary_latest
            )
            precision_latest = (
                precision_df.iloc[-1]
                if precision_df is not None and not precision_df.empty
                else primary_latest
            )

            self._latest_data = primary_latest

            # Check trading quality (Fix #2 - Quality-based trading for MTF)
            if regime_analysis:
                trading_quality = regime_analysis.get("trading_quality_score", 0)
                if trading_quality < 30:
                    logger.info(
                        f"🚫 Low trading quality: {trading_quality}/100 - Market regime: {regime_analysis.get('regime')}"
                    )
                    return self._fallback_analysis(
                        f"Trading quality too low ({trading_quality}/100) for {regime_analysis.get('regime')} regime",
                        float(primary_latest["close"]),
                    )

            # Multi-timeframe confluence check
            mtf_confluence = self._calculate_mtf_confluence(
                primary_df, context_df, precision_df, regime_analysis
            )

            # Get dynamic threshold from settings (environment-aware)
            dynamic_threshold = (
                regime_analysis.get("dynamic_confluence_threshold", settings.get_confluence_threshold())
                if regime_analysis
                else settings.get_confluence_threshold()
            )

            if mtf_confluence < dynamic_threshold:
                logger.info(
                    f"⚠️ Low MTF confluence: {mtf_confluence}/{dynamic_threshold} "
                    f"(Mode: {settings.get_environment_name()}) - Recommending HOLD"
                )
                return self._fallback_analysis(
                    f"Insufficient multi-timeframe confluence ({mtf_confluence}/{dynamic_threshold})",
                    float(primary_latest["close"]),
                )

            # Build multi-timeframe summary for AI
            mtf_summary = self._build_mtf_summary(
                symbol,
                primary_df,
                context_df,
                precision_df,
                primary_latest,
                context_latest,
                precision_latest,
            )

            # Generate enhanced prompt with MTF context
            prompt = self._generate_mtf_prompt(
                primary_latest, mtf_summary, user_trade_history, regime_analysis
            )

            try:
                response = await self.model.generate_content_async(prompt)
                response_text = (response.text or "").strip()
                logger.info(f"🤖 Gemini MTF Response: {response.text}")
                ai_analysis = self._parse_model_response(response_text)
            except ValueError as parse_error:
                logger.error(f"❌ JSON Parse Error: {parse_error}")
                return self._fallback_analysis(
                    "AI response parsing failed",
                    float(primary_latest.get("close", 0.0)),
                )
            except Exception as gemini_error:
                logger.error(f"❌ Gemini API Error: {gemini_error}")
                return self._fallback_analysis(
                    f"AI analysis failed: {str(gemini_error)[:100]}",
                    float(primary_latest.get("close", 0.0)),
                )

            sanitized = self._sanitize_analysis(ai_analysis, primary_latest)
            enriched = self._enforce_price_targets(sanitized)

            technical_score = self._calculate_technical_score(primary_latest)
            enriched["technical_score"] = technical_score
            enriched["final_confidence"] = min(enriched["confidence"], technical_score)
            enriched["mtf_confluence"] = mtf_confluence
            enriched["timeframe_analysis"] = mtf_summary
            enriched["regime_analysis"] = regime_analysis  # Include for risk validation

            logger.info(f"🎯 Final MTF AI Analysis: {enriched}")
            return enriched

        except Exception as error:
            logger.error(f"❌ MTF AI analysis error: {error}")
            return self._fallback_analysis(f"Analysis error: {str(error)[:50]}")

    def _calculate_mtf_confluence(
        self, primary_df, context_df, precision_df, _regime_analysis
    ) -> int:
        """
        Calculate multi-timeframe confluence score.
        Checks alignment between 15m, 1h, and 5m timeframes.
        
        FIXED: More generous scoring to allow day trading opportunities.
        Target: 55+ for quality setups (was 75+)
        """
        score = 0

        try:
            primary_latest = primary_df.iloc[-1]
            context_latest = (
                context_df.iloc[-1]
                if context_df is not None and not context_df.empty
                else None
            )
            precision_latest = (
                precision_df.iloc[-1]
                if precision_df is not None and not precision_df.empty
                else None
            )

            # 1. Trend alignment across timeframes (40 points)
            if context_latest is not None:
                primary_trend = self._get_trend_direction(primary_latest)
                context_trend = self._get_trend_direction(context_latest)

                if primary_trend == context_trend and primary_trend != "neutral":
                    score += 40  # Strong alignment
                elif primary_trend == context_trend:
                    score += 10  # Neutral alignment (Fix #4 - lowered from 25)
                elif primary_trend != "neutral" or context_trend != "neutral":
                    score += 15  # One timeframe has direction 
                else:
                    score += 10  # Both neutral
            else:
                score += 25  # No context data, give reasonable credit 

            # 2. Momentum confirmation (30 points)
            primary_momentum = self._get_momentum_state(primary_latest)
            if precision_latest is not None:
                precision_momentum = self._get_momentum_state(precision_latest)
                if primary_momentum == precision_momentum:
                    score += 30
                elif primary_momentum != "neutral" or precision_momentum != "neutral":
                    score += 20  # Partial momentum 
                else:
                    score += 10  # Both neutral
            else:
                score += 20  # No precision data

            # 3. Volume confirmation (30 points) - More lenient for day trading
            if not pd.isna(primary_latest.get("volume_sma", np.nan)):
                volume_ratio = primary_latest["volume"] / primary_latest["volume_sma"]
                if volume_ratio > 1.1:
                    score += 30  # Strong volume
                elif volume_ratio > 0.9: 
                    score += 25  # Above-average volume
                elif volume_ratio > 0.7:
                    score += 15  # Acceptable volume
                else:
                    score += 10  # Low volume
            else:
                score += 15  # No volume data

            final_score = min(100, score)
            logger.debug(f"📊 MTF Confluence: {final_score}/100 (Target: 55+ for trades)")
            return final_score

        except Exception as e:
            logger.error(f"❌ MTF confluence calculation error: {e}")
            return 50

    def _get_trend_direction(self, candle_data) -> str:
        """Determine trend direction from candle data."""
        try:
            price = candle_data["close"]
            sma_20 = candle_data.get("sma_20", np.nan)
            sma_50 = candle_data.get("sma_50", np.nan)

            if pd.isna(sma_20) or pd.isna(sma_50):
                return "neutral"

            if price > sma_20 and sma_20 > sma_50:
                return "bullish"
            elif price < sma_20 and sma_20 < sma_50:
                return "bearish"
            else:
                return "neutral"
        except Exception:
            return "neutral"

    def _get_momentum_state(self, candle_data) -> str:
        """Determine momentum state from candle data."""
        try:
            macd = candle_data.get("macd", 0)
            macd_signal = candle_data.get("macd_signal", 0)
            rsi = candle_data.get("rsi", 50)

            if pd.isna(macd) or pd.isna(macd_signal) or pd.isna(rsi):
                return "neutral"

            if macd > macd_signal and rsi > 50:
                return "bullish"
            elif macd < macd_signal and rsi < 50:
                return "bearish"
            else:
                return "neutral"
        except Exception:
            return "neutral"

    def _build_mtf_summary(
        self,
        symbol,
        _primary_df,
        context_df,
        precision_df,
        primary_latest,
        context_latest,
        precision_latest,
    ) -> Dict:
        """Build comprehensive multi-timeframe market summary."""
        summary = {
            "symbol": symbol,
            "primary_timeframe": "15m",
            "context_timeframe": "1h",
            "precision_timeframe": "5m",
        }

        # Primary timeframe (15m) - main trading signals
        summary["primary"] = {
            "price": float(primary_latest["close"]),
            "trend": self._get_trend_direction(primary_latest),
            "momentum": self._get_momentum_state(primary_latest),
            "rsi": float(primary_latest.get("rsi", 50)),
            "volume_ratio": float(
                primary_latest["volume"] / primary_latest.get("volume_sma", 1)
            ),
        }

        # Context timeframe (1h) - overall trend
        if context_df is not None and not context_df.empty:
            summary["context"] = {
                "trend": self._get_trend_direction(context_latest),
                "trend_strength": float(context_latest.get("adx", 0)),
                "momentum": self._get_momentum_state(context_latest),
            }

        # Precision timeframe (5m) - entry timing
        if precision_df is not None and not precision_df.empty:
            summary["precision"] = {
                "momentum": self._get_momentum_state(precision_latest),
                "recent_direction": (
                    "up"
                    if precision_latest["close"] > precision_df.iloc[-2]["close"]
                    else "down"
                ),
            }

        return summary

    def _sanitize_for_json(self, obj):
        """Convert numpy types and other non-JSON-serializable types to native Python types."""
        if isinstance(obj, dict):
            return {key: self._sanitize_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        else:
            return obj

    def _generate_mtf_prompt(
        self, primary_latest, mtf_summary, user_history, regime_analysis
    ) -> str:
        """Generate enhanced prompt with multi-timeframe context."""
        # Sanitize regime_analysis to ensure JSON serialization works
        regime_analysis_clean = self._sanitize_for_json(regime_analysis) if regime_analysis else None
        
        prompt = f"""You are an expert day trading analyst. Analyze this multi-timeframe market data and provide a trading recommendation.

MULTI-TIMEFRAME ANALYSIS:
- Primary (15m): {json.dumps(mtf_summary.get('primary', {}), indent=2)}
- Context (1h): {json.dumps(mtf_summary.get('context', {}), indent=2)}
- Precision (5m): {json.dumps(mtf_summary.get('precision', {}), indent=2)}

CURRENT MARKET REGIME:
{json.dumps(regime_analysis_clean, indent=2) if regime_analysis_clean else 'No regime data available'}

TRADING STYLE: Day Trading
- Desired R:R Ratio: At least 2:1
- Holding period: 30 minutes to 8 hours
- Risk-Reward: Minimum 1:1.5, target 1:2 or better
- Quality over quantity: Only high-probability setups

TECHNICAL INDICATORS (15m):
- RSI: {float(primary_latest.get('rsi', 50)):.1f}
- MACD: {float(primary_latest.get('macd', 0)):.4f}
- ADX: {float(primary_latest.get('adx', 0)):.1f}
- ATR: {float(primary_latest.get('atr', 0)):.4f}

"""
        # Fix #5: Add microstructure signals to prompt
        if regime_analysis:
            prompt += f"""
ADVANCED MICROSTRUCTURE SIGNALS (Fix #5 Enhancement):
- Momentum Persistence: {regime_analysis.get('momentum_persistence', 50):.0f}/100 
  (How long momentum tends to last - high = strong trends)
- Order Flow Imbalance: {regime_analysis.get('order_flow_imbalance', 0):+.0f}/100 
  (Buying vs Selling pressure - positive = accumulation, negative = distribution)
- Mean Reversion Score: {regime_analysis.get('mean_reversion_score', 0):.0f}/100 
  (Price stretched from mean - high = excellent mean reversion opportunity)
- Trading Quality Score: {regime_analysis.get('trading_quality_score', 0):.0f}/100
  (Overall setup quality - must be >50 for trade approval)
- Trading Edge: {regime_analysis.get('trading_edge', 'NEUTRAL')}
  (TREND_FOLLOWING, MEAN_REVERSION, BREAKOUT, or NEUTRAL)
- Optimal Hold Time: {regime_analysis.get('optimal_hold_time', '10-20')} minutes

USE THESE SIGNALS TO REFINE YOUR ANALYSIS:
- If Order Flow > +40 and Momentum Persistence > 60: Strong trend-following setup
- If Mean Reversion > 70: Excellent scalping opportunity (price stretched)
- If Trading Edge = "MEAN_REVERSION": Focus on counter-trend entries
- If Trading Edge = "TREND_FOLLOWING": Go with the momentum
- Always respect the Optimal Hold Time for exit planning

"""
        
        if user_history:
            prompt += f"""
USER TRADING HISTORY:
- Total trades: {user_history.get('total_trades', 0)}
- Win rate: {user_history.get('win_rate', 0):.1f}%
- Recent trend: {user_history.get('recent_trend', 'unknown')}
"""

        prompt += """
Provide your analysis in JSON format:
{
  "signal": "buy" | "sell" | "hold",
  "confidence": 0-100,
  "reasoning": "Explain multi-timeframe alignment and why this is a high-quality setup",
  "entry_price": <suggested entry>,
  "stop_loss": <suggested stop loss price (e.g., 0.5% from entry)>,
  "take_profit": <suggested take profit price (e.g., 1.0% from entry, ensuring at least 2:1 R:R)>,
  "timeframe_alignment": "describe how 15m, 1h, and 5m align"
}

IMPORTANT: Only recommend buy/sell if you see strong multi-timeframe alignment and high probability. Default to "hold" if uncertain. Under NO circumstances should you issue a 'buy' signal if the regime is BEAR_TREND with ADX > 30. The same applies to 'sell' signals in a BULL_TREND.

CRITICAL TRADING RULES:
1. NEVER trade counter-trend in strong directional markets (ADX > 30)
   - BEAR_TREND + ADX > 30 → Only SELL signals allowed (or hold)
   - BULL_TREND + ADX > 30 → Only BUY signals allowed (or hold)

2. Natural R:R should target at least 2.0:1 from market structure
   - Use support/resistance levels for natural stops
   - If market structure suggests poor R:R, recommend HOLD
   - Look for trades where stops can be 1-1.5% and targets 2-3%

3. Trust your analysis - if timeframes align and quality is good, take the trade
   - Trading Quality Score > 50 is good, > 70 is excellent
   - Strong multi-timeframe alignment (bullish on 15m, 1h, 5m) = high confidence
   - Don't be paralyzed by minor conflicting signals
   - In day trading, slight order flow negativity can be normal during pullbacks
"""
        return prompt
