import google.generativeai as genai
from typing import Dict
import pandas as pd
import numpy as np
import json
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

        return df

    async def analyze_market_data(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Use Gemini AI to analyze market data and provide trading signals."""
        try:
            if self.model is None:
                logger.warning("🤖 AI model not available, using fallback analysis")
                return self._fallback_analysis("AI model unavailable - using safe default")

            if df.empty or len(df) < 20:
                return self._fallback_analysis("Insufficient data")

            df_with_indicators = self.calculate_technical_indicators(df)
            if df_with_indicators.empty or len(df_with_indicators) < 2:
                return self._fallback_analysis("Insufficient indicator data")

            latest = df_with_indicators.iloc[-1]
            prev = df_with_indicators.iloc[-2]
            self._latest_data = latest

            market_summary = self._build_market_summary(symbol, df_with_indicators, latest, prev)
            prompt = self._generate_prompt(latest, market_summary)

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

    def _generate_prompt(self, latest: pd.Series, market_summary: Dict) -> str:
        return (
            "\n            You are an expert cryptocurrency scalping analyst. Analyze this BTC/USDT market data for a 1-minute scalping strategy:\n\n"
            f"            Market Data:\n            {json.dumps(market_summary, indent=2)}\n\n"
            "            Scalping Strategy Context:\n"
            "            - Target: 0.3% profit per trade\n"
            "            - Stop loss: 0.5% maximum loss\n"
            "            - Risk tolerance: Very conservative (1% account risk)\n"
            "            - Trading timeframe: 1-5 minutes\n"
            "            - Market session: High volume hours (8AM-4PM GMT)\n\n"
            "            Based on this data, provide a JSON response with ALL 6 required fields:\n"
            "            1. \"signal\": \"buy\", \"sell\", or \"hold\" \n"
            "            2. \"confidence\": integer from 0-100\n"
            "            3. \"reasoning\": brief explanation (max 100 words)\n"
            f"            4. \"entry_price\": REQUIRED - suggested entry price (use current price {latest['close']:.2f} if unsure)\n"
            "            5. \"stop_loss\": REQUIRED - suggested stop loss price (0.3% from entry for fast execution)\n"
            "            6. \"take_profit\": REQUIRED - suggested take profit price (0.5% from entry for fast execution)\n\n"
            "            Consider:\n"
            "            - RSI overbought (>70) or oversold (<30) conditions\n"
            "            - MACD momentum and divergence\n"
            "            - Volume confirmation\n"
            "            - Bollinger Band squeeze/expansion\n"
            "            - Price action relative to moving averages\n\n"
            "            CRITICAL: You MUST include ALL 6 fields (signal, confidence, reasoning, entry_price, stop_loss, take_profit) in your response.\n"
            "            Return ONLY valid JSON without markdown code blocks or any other formatting.\n\n"
            f"            Example format with ALL required fields:\n            {{\"signal\": \"buy\", \"confidence\": 75, \"reasoning\": \"Strong bullish momentum\", \"entry_price\": {latest['close']:.2f}, \"stop_loss\": {latest['close'] * AI_BUY_STOP_LOSS_RATIO:.2f}, \"take_profit\": {latest['close'] * AI_BUY_TAKE_PROFIT_RATIO:.2f}}}\n            "
        )

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
        fallback_price = price if price and price > 0 else self._resolve_fallback_price()
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
