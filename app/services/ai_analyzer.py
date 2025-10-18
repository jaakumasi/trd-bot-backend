import google.generativeai as genai
from typing import Dict, Optional
import pandas as pd
import numpy as np
import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..config import settings
from .service_constants import (
    AI_BUY_STOP_LOSS_RATIO,
    AI_BUY_TAKE_PROFIT_RATIO,
    AI_SELL_STOP_LOSS_RATIO,
    AI_SELL_TAKE_PROFIT_RATIO,
    AI_HOLD_STOP_LOSS_RATIO,
    AI_HOLD_TAKE_PROFIT_RATIO,
)
import logging

logger = logging.getLogger(__name__)


class AIAnalyzer:
    def __init__(self):
        try:
            genai.configure(api_key=settings.gemini_api_key)
            # Updated to use the latest available Gemini model
            self.model = genai.GenerativeModel("models/gemini-2.5-flash")
            logger.info("✅ AI Analyzer initialized with Gemini 2.5 Flash model")
        except Exception as e:
            logger.error(f"❌ Failed to initialize AI Analyzer: {e}")
            self.model = None

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for analysis"""
        if df.empty or len(df) < 20:
            return df

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
        high = df['high']
        low = df['low']
        close = df['close']
        
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
        df['plus_di'] = 100 * (plus_dm_smooth / atr)
        df['minus_di'] = 100 * (minus_dm_smooth / atr)
        
        # ADX Calculation
        dx = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = dx.rolling(window=period).mean()
        
        # Fill NaN values with 0
        df['adx'] = df['adx'].fillna(0)
        df['plus_di'] = df['plus_di'].fillna(0)
        df['minus_di'] = df['minus_di'].fillna(0)
        
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
        regime_analysis: Optional[Dict] = None
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

            # Step 1: Check if we should even analyze (regime filter)
            if regime_analysis and not regime_analysis.get("allow_scalping"):
                logger.info(f"🚫 Scalping blocked - Market regime: {regime_analysis.get('regime')}")
                return self._fallback_analysis(
                    f"Market regime {regime_analysis.get('regime')} not suitable for scalping. "
                    f"ADX: {regime_analysis.get('trend_strength', 0):.1f}, "
                    f"ATR%: {regime_analysis.get('atr_percentage', 0):.2f}%",
                    float(latest["close"])
                )
            
            # Step 2: Signal Confluence Check (now with dynamic thresholds!)
            confluence_score = self._calculate_signal_confluence(df_with_indicators, regime_analysis)
            
            # Get dynamic threshold from advanced regime analysis (if available)
            dynamic_threshold = regime_analysis.get('dynamic_confluence_threshold', 60) if regime_analysis else 60
            
            if confluence_score < dynamic_threshold:
                logger.info(
                    f"⚠️ Low signal confluence: {confluence_score}/{dynamic_threshold} "
                    f"(dynamic threshold) - Recommending HOLD"
                )
                return self._fallback_analysis(
                    f"Insufficient signal confluence ({confluence_score}/{dynamic_threshold}). "
                    f"Trend alignment, momentum, volume, or regime not aligned.",
                    float(latest["close"])
                )

            market_summary = self._build_market_summary(
                symbol, df_with_indicators, latest, prev
            )
            prompt = self._generate_prompt(latest, market_summary, user_trade_history, regime_analysis)

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

            logger.info(f"🎯 Final AI Analysis: {enriched}")
            return enriched

        except Exception as error:
            logger.error(f"AI analysis error: {error}")
            return self._fallback_analysis(f"Analysis error: {str(error)[:50]}")

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

    def _calculate_signal_confluence(self, df: pd.DataFrame, regime_analysis: Optional[Dict]) -> int:
        """
        Multi-factor signal confluence scoring.
        
        Checks for alignment across multiple indicators:
        - Trend alignment (price vs SMAs) - 25 points
        - Momentum confirmation (MACD + RSI) - 25 points
        - Volume confirmation - 20 points
        - Volatility regime appropriateness - 30 points
        
        Returns: 0-100 score (>60 required for trade)
        """
        score = 0
        latest = df.iloc[-1]
        
        # Factor 1: Trend Alignment (25 points)
        sma_20 = latest.get('sma_20', np.nan)
        sma_50 = latest.get('sma_50', np.nan)
        price = latest.get('close', 0)
        
        if not pd.isna(sma_20) and not pd.isna(sma_50):
            if sma_20 > sma_50 and price > sma_20:  # Bullish alignment
                score += 25
            elif sma_20 < sma_50 and price < sma_20:  # Bearish alignment
                score += 25
            elif sma_20 > sma_50 and price > sma_50:  # Partial bull
                score += 15
            elif sma_20 < sma_50 and price < sma_50:  # Partial bear
                score += 15
        
        # Factor 2: Momentum Confirmation (25 points)
        rsi = latest.get('rsi', 50)
        macd = latest.get('macd', 0)
        macd_signal = latest.get('macd_signal', 0)
        
        if not pd.isna(rsi) and not pd.isna(macd):
            # RSI in valid range (not extreme)
            if 30 < rsi < 70:
                score += 10
            
            # MACD aligned with trend
            if macd > macd_signal and rsi > 50:  # Bullish momentum
                score += 15
            elif macd < macd_signal and rsi < 50:  # Bearish momentum
                score += 15
        
        # Factor 3: Volume Confirmation (20 points)
        volume = latest.get('volume', 0)
        volume_sma = latest.get('volume_sma', 1)
        
        if not pd.isna(volume_sma) and volume_sma > 0:
            volume_ratio = volume / volume_sma
            
            if volume_ratio > 1.2:  # Above-average volume
                score += 20
            elif volume_ratio > 1.0:
                score += 10
        
        # Factor 4: Regime Appropriateness (30 points + bonuses from advanced metrics)
        if regime_analysis:
            regime = regime_analysis.get('regime')
            trend_strength = regime_analysis.get('trend_strength', 0)
            atr_percentage = regime_analysis.get('atr_percentage', 0)
            
            # Use scalping quality score if available (from advanced analyzer)
            scalping_quality = regime_analysis.get('scalping_quality_score')
            
            if scalping_quality is not None:
                # REVOLUTIONARY: Use quality score instead of binary regime check
                score += int(scalping_quality * 0.5)  # 0-50 points based on quality
                
                # Bonus for high-quality microstructure
                mean_reversion = regime_analysis.get('mean_reversion_score', 0)
                if mean_reversion > 70:
                    score += 15  # Excellent mean reversion setup
                
            else:
                # Legacy scoring (for non-scalping modes)
                if regime in ['BULL_TREND', 'BEAR_TREND'] and trend_strength > 25:
                    score += 30
                    
                    # Bonus for optimal volatility range
                    if 0.1 < atr_percentage < 1.2:
                        score += 5
                        
                elif regime == 'RANGE_BOUND':
                    # Don't penalize range-bound anymore! Can be great for scalping
                    score += 10  # Give some credit
                    
                elif regime == 'HIGH_VOLATILITY':
                    # Less harsh penalty
                    if atr_percentage > 2.0:
                        score -= 15
                    else:
                        score -= 5
                        
                elif regime == 'LOW_VOLATILITY':
                    # Less harsh penalty - low vol can work!
                    score -= 5
        
        final_score = max(0, min(100, score))
        
        logger.info(f"📊 Signal Confluence Score: {final_score}/100")
        
        return final_score

    def _generate_prompt(
        self,
        latest: pd.Series,
        market_summary: Dict,
        user_trade_history: Optional[Dict] = None,
        regime_analysis: Optional[Dict] = None
    ) -> str:
        # Build historical context section if available
        history_context = ""
        if user_trade_history and user_trade_history.get("has_history"):
            h = user_trade_history
            history_context = f"""
            User's Recent Trading Performance (Last {h['total_trades']} Trades):
            - Win Rate: {h['win_rate']:.1f}% ({h['winning_trades']} wins, {h['losing_trades']} losses)
            - Average P&L per Trade: ${h['avg_pnl_per_trade']:+.2f}
            - Total P&L: ${h['total_pnl']:+.2f}
            - Average Trade Duration: {h['avg_duration_minutes']:.1f} minutes
            - Stop Loss Hit Rate: {h['stop_loss_rate']:.1f}%
            - Take Profit Hit Rate: {h['take_profit_rate']:.1f}%
            - Recent Trend: {h['recent_trend'].upper()} ({h['recent_wins']} wins in last {h['recent_total']} trades)

            INSTRUCTIONS BASED ON USER HISTORY (USE WITH MODERATION):
            - If win_rate < 30% OR (win_rate < 40% AND recent_trend is "losing"): Be more conservative. Reduce confidence by 10-15 points if considering a trade.
            - If stop_loss_rate > 70%: User frequently hits stop losses. Focus on higher-quality setups with stronger confluence.
            - If recent_trend is "losing" (only 0-1 wins in last 5 trades): Reduce confidence by 5-10 points to encourage caution.
            - If win_rate > 55% AND recent_trend is "winning": You can maintain normal confidence levels.
            
            IMPORTANT: Historical performance should influence confidence scores moderately, NOT cause complete avoidance of trading.
            For scalping strategies, some losses are normal. Do not over-penalize recent losses - focus on technical setup quality.
            Your primary job is to analyze current market conditions, with history as a secondary factor.
            """
        
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
            - Scalping Status: {'ALLOWED' if r.get('allow_scalping') else 'BLOCKED'}

            REGIME-BASED TRADING RULES:
            - BULL_TREND/BEAR_TREND with ADX>25: Favorable for scalping - trade in direction of trend
            - RANGE_BOUND: Not suitable for scalping - recommend 'hold' unless exceptionally strong signal
            - HIGH_VOLATILITY: Too risky for scalping - recommend 'hold' to avoid whipsaw
            - LOW_VOLATILITY: Insufficient movement - recommend 'hold' to wait for better opportunity
            
            Respect the regime classification. If scalping is BLOCKED, only recommend 'hold' unless there's an exceptionally clear setup (confidence >90).
            """

        return f"""
            You are an expert cryptocurrency scalping analyst. Analyze this BTC/USDT market data for a 1-minute scalping strategy:

            Market Data:
            {json.dumps(market_summary, indent=2)}
            
            {regime_context}
            Scalping Strategy Context:
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
            {{"signal": "buy", "confidence": 75, "reasoning": "Strong bullish momentum", "entry_price": {latest['close']:.2f}, "stop_loss": {latest['close'] * AI_BUY_STOP_LOSS_RATIO:.2f}, "take_profit": {latest['close'] * AI_BUY_TAKE_PROFIT_RATIO:.2f}}}
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

    def _enforce_price_targets(self, analysis: Dict) -> Dict:
        entry_price = float(analysis["entry_price"])
        signal = analysis["signal"]

        if signal == "buy":
            analysis["stop_loss"] = entry_price * AI_BUY_STOP_LOSS_RATIO
            analysis["take_profit"] = entry_price * AI_BUY_TAKE_PROFIT_RATIO
        elif signal == "sell":
            analysis["stop_loss"] = entry_price * AI_SELL_STOP_LOSS_RATIO
            analysis["take_profit"] = entry_price * AI_SELL_TAKE_PROFIT_RATIO
        else:
            analysis["stop_loss"] = entry_price * AI_HOLD_STOP_LOSS_RATIO
            analysis["take_profit"] = entry_price * AI_HOLD_TAKE_PROFIT_RATIO

        logger.debug(
            "🎯 Enforced TP/SL ratios | signal=%s | entry=%.2f | SL=%.2f | TP=%.2f",
            signal,
            entry_price,
            analysis["stop_loss"],
            analysis["take_profit"],
        )
        return analysis

    def _fallback_analysis(self, reason: str, price: float | None = None) -> Dict:
        fallback_price = (
            price if price and price > 0 else self._resolve_fallback_price()
        )
        return {
            "signal": "hold",
            "confidence": 0,
            "reasoning": reason,
            "technical_score": 0,
            "entry_price": fallback_price,
            "stop_loss": fallback_price * AI_HOLD_STOP_LOSS_RATIO,
            "take_profit": fallback_price * AI_HOLD_TAKE_PROFIT_RATIO,
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
