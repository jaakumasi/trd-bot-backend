import google.generativeai as genai
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import json
from ..config import settings
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

        return df

    async def analyze_market_data(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Use Gemini AI to analyze market data and provide trading signals"""
        try:
            # Check if model is available
            if self.model is None:
                logger.warning("🤖 AI model not available, using fallback analysis")
                return {
                    "signal": "hold",
                    "confidence": 0,
                    "reasoning": "AI model unavailable - using safe default",
                }
                
            if df.empty or len(df) < 20:
                return {
                    "signal": "hold",
                    "confidence": 0,
                    "reasoning": "Insufficient data",
                }

            # Calculate technical indicators
            df_with_indicators = self.calculate_technical_indicators(df)
            latest = df_with_indicators.iloc[-1]
            prev = df_with_indicators.iloc[-2]

            # Prepare market analysis prompt
            market_summary = {
                "symbol": symbol,
                "current_price": float(latest["close"]),
                "price_change_1h": float(
                    (latest["close"] - df_with_indicators.iloc[-12]["close"])
                    / df_with_indicators.iloc[-12]["close"]
                    * 100
                ),
                "volume_ratio": (
                    float(latest["volume"] / latest["volume_sma"])
                    if not pd.isna(latest["volume_sma"])
                    else 1.0
                ),
                "rsi": float(latest["rsi"]) if not pd.isna(latest["rsi"]) else 50,
                "macd": {
                    "macd": float(latest["macd"]) if not pd.isna(latest["macd"]) else 0,
                    "signal": (
                        float(latest["macd_signal"])
                        if not pd.isna(latest["macd_signal"])
                        else 0
                    ),
                    "histogram": (
                        float(latest["macd"] - latest["macd_signal"])
                        if not pd.isna(latest["macd"])
                        and not pd.isna(latest["macd_signal"])
                        else 0
                    ),
                },
                "bollinger_bands": {
                    "position": "middle",
                    "squeeze": (
                        float(
                            (latest["bb_upper"] - latest["bb_lower"])
                            / latest["bb_middle"]
                            * 100
                        )
                        if not pd.isna(latest["bb_upper"])
                        else 0
                    ),
                },
                "moving_averages": {
                    "price_vs_sma20": (
                        float(
                            (latest["close"] - latest["sma_20"])
                            / latest["sma_20"]
                            * 100
                        )
                        if not pd.isna(latest["sma_20"])
                        else 0
                    ),
                    "sma20_trend": (
                        "up"
                        if latest["sma_20"] > prev["sma_20"]
                        else (
                            "down"
                            if not pd.isna(latest["sma_20"])
                            and not pd.isna(prev["sma_20"])
                            else "neutral"
                        )
                    ),
                },
            }

            # Determine Bollinger Band position
            if not pd.isna(latest["bb_upper"]) and not pd.isna(latest["bb_lower"]):
                if latest["close"] > latest["bb_upper"]:
                    market_summary["bollinger_bands"]["position"] = "above_upper"
                elif latest["close"] < latest["bb_lower"]:
                    market_summary["bollinger_bands"]["position"] = "below_lower"
                elif latest["close"] > latest["bb_middle"]:
                    market_summary["bollinger_bands"]["position"] = "upper_half"
                else:
                    market_summary["bollinger_bands"]["position"] = "lower_half"

            prompt = f"""
            You are an expert cryptocurrency scalping analyst. Analyze this BTC/USDT market data for a 1-minute scalping strategy:

            Market Data:
            {json.dumps(market_summary, indent=2)}

            Scalping Strategy Context:
            - Target: 0.3% profit per trade
            - Stop loss: 0.5% maximum loss
            - Risk tolerance: Very conservative (1% account risk)
            - Trading timeframe: 1-5 minutes
            - Market session: High volume hours (8AM-4PM GMT)

            Based on this data, provide a JSON response with:
            1. "signal": "buy", "sell", or "hold"
            2. "confidence": integer from 0-100
            3. "reasoning": brief explanation (max 100 words)
            4. "entry_price": suggested entry price if buy/sell signal
            5. "stop_loss": suggested stop loss price
            6. "take_profit": suggested take profit price

            Consider:
            - RSI overbought (>70) or oversold (<30) conditions
            - MACD momentum and divergence
            - Volume confirmation
            - Bollinger Band squeeze/expansion
            - Price action relative to moving averages

            IMPORTANT: Return ONLY valid JSON without markdown code blocks or any other formatting.
            Example format:
            {{"signal": "buy", "confidence": 75, "reasoning": "Strong bullish momentum", "entry_price": 50000, "stop_loss": 49750, "take_profit": 50150}}
            """

            try:
                response = await self.model.generate_content_async(prompt)
                
                # Log full response for debugging
                logger.info(f"🤖 Gemini Full Response: {response.text}")

                # Extract JSON from markdown code blocks if present
                response_text = response.text.strip()
                if response_text.startswith("```json"):
                    # Remove markdown code block markers
                    response_text = response_text[7:]  # Remove ```json
                    if response_text.endswith("```"):
                        response_text = response_text[:-3]  # Remove ```
                elif response_text.startswith("```"):
                    # Handle generic code blocks
                    lines = response_text.split('\n')
                    response_text = '\n'.join(lines[1:-1])  # Remove first and last lines
                
                response_text = response_text.strip()
                logger.info(f"🔍 Cleaned JSON for parsing: {response_text}")

                # Parse AI response
                try:
                    ai_analysis = json.loads(response_text)
                    logger.info(f"✅ Successfully parsed AI response: {ai_analysis}")
                except json.JSONDecodeError as json_error:
                    logger.error(f"❌ JSON Parse Error: {json_error}")
                    logger.error(f"❌ Raw response text: {repr(response_text)}")
                    return {
                        "signal": "hold",
                        "confidence": 0,
                        "reasoning": "AI response parsing failed - invalid JSON format",
                        "technical_score": 0,
                    }

                # Validate and sanitize response
                valid_signals = ["buy", "sell", "hold"]
                if ai_analysis.get("signal") not in valid_signals:
                    logger.warning(f"⚠️ Invalid signal '{ai_analysis.get('signal')}', defaulting to 'hold'")
                    ai_analysis["signal"] = "hold"

                ai_analysis["confidence"] = max(
                    0, min(100, int(ai_analysis.get("confidence", 0)))
                )

                # Add technical confirmation to validate AI analysis
                technical_score = self._calculate_technical_score(latest)
                ai_analysis["technical_score"] = technical_score
                ai_analysis["final_confidence"] = min(
                    ai_analysis["confidence"], technical_score
                )

                logger.info(f"🎯 Final AI Analysis: {ai_analysis}")
                return ai_analysis

            except Exception as gemini_error:
                logger.error(f"❌ Gemini API Error: {gemini_error}")
                # Return fallback analysis if Gemini fails
                return {
                    "signal": "hold",
                    "confidence": 0,
                    "reasoning": f"AI analysis failed: {str(gemini_error)[:100]}",
                    "technical_score": 0,
                }

        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            return {
                "signal": "hold",
                "confidence": 0,
                "reasoning": f"Analysis error: {str(e)[:50]}",
                "technical_score": 0,
            }

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
