import logging
from typing import Dict, Tuple, Optional, Union, List
from decimal import Decimal
from datetime import datetime, timezone

from .service_constants import (
    DEFAULT_MAX_DAILY_TRADES,
    DEFAULT_RISK_PERCENTAGE,
    EIGHT_DECIMAL_PLACES,
    MAX_BALANCE_TRADE_RATIO,
    MAX_OPEN_POSITIONS,
    MIN_ACCOUNT_BALANCE,
    MIN_SIGNAL_CONFIDENCE,
    MIN_TRADE_VALUE,
)

logger = logging.getLogger(__name__)


class RiskValidationError(Exception):
    """Raised when a trade fails risk validation checks."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class RiskManager:
    def __init__(self):
        self.max_daily_trades = {}
        self.daily_trade_count = {}
        self.open_positions = {}  # {user_id: [position_objects]}
        self.current_volatility_percentile = None  # Store latest volatility reading
        self.last_regime_change_time = datetime.now(timezone.utc)  # Track regime changes for whipsaw protection
        self.last_regime = None  # Track last known regime

    def calculate_position_size(
        self,
        account_balance: Union[float, Decimal],
        risk_percentage: Union[float, Decimal],
        entry_price: Union[float, Decimal],
        stop_loss_price: Union[float, Decimal],
    ) -> float:
        """
        Calculate position size using the 1% RULE for DAY TRADING.

        Formula: Position Size = (Account Balance × Risk %) / (Entry Price - Stop Loss Price)

        This ensures:
        - Maximum loss per trade = 1% of account balance
        - Position size dynamically adjusts based on stop loss distance
        - Wider stops = smaller positions, tighter stops = larger positions
        - Never exceeds available capital (safety cap applied)

        Args:
            account_balance: Total account balance in USDT
            risk_percentage: Max % of balance to risk (typically 1.0 = 1%)
            entry_price: Expected entry price
            stop_loss_price: Stop-loss price

        Returns:
            Position size in base currency (e.g., BTC)
        """
        try:
            balance = self._to_float(account_balance)
            risk = self._to_float(risk_percentage)
            entry = self._to_float(entry_price)
            stop = self._to_float(stop_loss_price)

            logger.debug(
                "🔢 Position calc inputs: balance=$%.2f, risk=%.1f%%, entry=$%.2f, stop=$%.2f",
                balance,
                risk,
                entry,
                stop,
            )

            price_risk = abs(entry - stop)
            if price_risk == 0:
                logger.warning("⚠️ Price risk is zero - entry equals stop loss")
                return 0.0

            # PRIMARY METHOD: 1% Rule (Risk-Based Position Sizing)
            # This is the gold standard for day trading risk management
            risk_amount = balance * (risk / 100)
            position_size = risk_amount / price_risk

            logger.debug(
                "📊 1%% Rule calculation:\n"
                "   Risk amount: $%.2f (%.1f%% of $%.2f)\n"
                "   Price risk: $%.2f per unit\n"
                "   Initial position: %.8f units",
                risk_amount,
                risk,
                balance,
                price_risk,
                position_size,
            )

            # SAFETY CAP: Position value should not exceed MAX_BALANCE_TRADE_RATIO of balance
            # This prevents over-concentration in a single trade
            position_value = position_size * entry
            max_position_value = balance * MAX_BALANCE_TRADE_RATIO

            if position_value > max_position_value:
                logger.warning(
                    f"⚠️ Position value (${position_value:.2f}) exceeds {MAX_BALANCE_TRADE_RATIO*100:.1f}%% cap (${max_position_value:.2f})"
                )
                position_size = max_position_value / entry
                position_value = position_size * entry
                logger.info(
                    "🔻 Position reduced to: %.8f (value: $%.2f, %.1f%% of balance)",
                    position_size,
                    position_value,
                    (position_value / balance * 100),
                )

            # Validate stop loss percentage is reasonable for day trading
            stop_loss_pct = (price_risk / entry) * 100
            if stop_loss_pct < 0.5:
                logger.warning(
                    f"⚠️ Very tight stop: {stop_loss_pct:.2f}% (<0.5% may be too tight for day trading)"
                )
            elif stop_loss_pct > 3.0:
                logger.warning(
                    f"⚠️ Wide stop: {stop_loss_pct:.2f}% (>3% may be too wide for intraday)"
                )
            else:
                logger.debug(
                    f"✅ Stop loss distance: {stop_loss_pct:.2f}% (acceptable range)"
                )

            # Truncate to exchange precision (8 decimals)
            position_size = self._truncate(position_size)
            position_value = position_size * entry

            logger.info(
                "✅ FINAL position size: %.8f units\n"
                "   Position value: $%.2f (%.2f%% of balance)\n"
                "   Max risk if SL hits: $%.2f (%.2f%% of balance)\n"
                "   Stop distance: %.2f%%",
                position_size,
                position_value,
                (position_value / balance * 100) if balance > 0 else 0,
                risk_amount,
                risk,
                stop_loss_pct,
            )

            # Final validation: position should never exceed balance
            if position_value > balance:
                logger.error(
                    f"❌ CRITICAL: Position value (${position_value:.2f}) exceeds balance (${balance:.2f})!"
                )
                return 0.0

            return position_size

        except Exception as e:
            logger.error(f"❌ Position size calculation error: {e}")
            logger.error(
                "Inputs - balance: %s (%s), risk: %s (%s), entry: %s (%s), stop: %s (%s)",
                account_balance,
                type(account_balance),
                risk_percentage,
                type(risk_percentage),
                entry_price,
                type(entry_price),
                stop_loss_price,
                type(stop_loss_price),
            )
            return 0.0

    def get_adaptive_confidence_threshold(
        self, user_trade_history: Optional[Dict] = None
    ) -> float:
        """
        Calculate adaptive confidence threshold to reduce overfitting.
        
        Cap maximum threshold to prevent over-conservatism.
        
        Adjusts MIN_SIGNAL_CONFIDENCE based on recent win rate:
        - High win rate (>60%) -> Lower threshold (more aggressive)
        - Normal win rate (40-60%) -> Standard threshold
        - Low win rate (<40%) -> Slightly higher threshold
        
        Also adds randomization to prevent point-estimate overfitting.
        
        Args:
            user_trade_history: Optional dictionary with 'win_rate' and 'recent_trend'
        
        Returns:
            Adaptive confidence threshold (50-70 range, capped)
        """
        base_threshold = MIN_SIGNAL_CONFIDENCE
        
        # If no history, use base threshold
        if not user_trade_history:
            capped = min(base_threshold, 70.0)
            logger.info(
                f"� No trade history - using base threshold: {capped:.1f} "
                f"(base: {base_threshold}, capped at 70%)"
            )
            return capped
        
        win_rate = user_trade_history.get('win_rate', 50.0)
        recent_trend = user_trade_history.get('recent_trend', 'neutral')
        
        # Adaptive adjustment based on performance (but capped at 70%)
        if win_rate > 60 and recent_trend == 'winning':
            adjusted = base_threshold - 10
            logger.info(
                f"📈 High performance detected (WR: {win_rate:.1f}%, trend: {recent_trend}) "
                f"- reducing threshold to {adjusted}"
            )
        elif win_rate < 40 or recent_trend == 'losing':
            adjusted = min(base_threshold + 5, 70.0)
            logger.info(
                f"📉 Low performance detected (WR: {win_rate:.1f}%, trend: {recent_trend}) "
                f"- increasing threshold to {adjusted} (capped at 70%)"
            )
        else:
            adjusted = base_threshold
            logger.info(
                f"➡️ Normal performance (WR: {win_rate:.1f}%) - using base threshold {adjusted}"
            )
        
        # Cap at 70% maximum to prevent killing good trades
        final_threshold = min(adjusted, 70.0)
        
        # Clamp to reasonable range (50-70%)
        final_threshold = max(50, min(70, final_threshold))
        
        logger.info(f"✅ Final adaptive threshold: {final_threshold:.1f}% (capped at 70%)")
        
        logger.info(
            f"✅ Adaptive confidence threshold: {final_threshold:.1f} "
            f"(adjusted: {adjusted}, randomized with ±8%)"
        )
        
        return final_threshold

    def validate_entry_timing(
        self, signal: Dict, market_df, regime_analysis: Optional[Dict]
    ) -> Tuple[bool, str]:
        """
        PRIORITY 1: Validate entry timing with momentum confirmation.
        
        Prevents taking trades at the worst possible moment (extended moves, no confirmation).
        Professional traders wait for confirmation before entering, not chase moves.
        
        Entry Requirements:
        1. Price not overextended (within 1.5 ATR of SMA20)
        2. Volume confirmation (>1.5x average on recent candles)
        3. Momentum confirmation (2-3 candles in trade direction)
        4. No entries within 30 mins of regime change (avoid whipsaws)
        
        Returns:
            (is_valid, reason) tuple
        """
        if market_df is None or market_df.empty or len(market_df) < 20:
            return True, "Insufficient data for entry timing validation"
        
        signal_action = signal.get("signal", "hold").lower()
        if signal_action == "hold":
            return True, "Hold signal"
        
        try:
            latest = market_df.iloc[-1]
            recent = market_df.tail(5)
            
            entry_price = latest['close']
            sma_20 = latest.get('sma_20', entry_price)
            atr = latest.get('atr', entry_price * 0.01)
            volume = latest.get('volume', 0)
            volume_sma = latest.get('volume_sma', volume)
            
            # CHECK 1: Price not overextended from SMA20
            distance_from_sma = abs(entry_price - sma_20)
            max_distance = atr * 1.5
            
            if distance_from_sma > max_distance:
                extension_pct = (distance_from_sma / entry_price) * 100
                return False, (
                    f"🚫 Price overextended {extension_pct:.2f}% from SMA20 "
                    f"(max allowed: {(max_distance/entry_price)*100:.2f}%). "
                    f"Chasing extended moves = high probability of reversal. Wait for pullback."
                )
            
            # CHECK 2: Volume confirmation (>1.5x average)
            if volume > 0 and volume_sma > 0:
                volume_ratio = volume / volume_sma
                if volume_ratio < 1.5:
                    return False, (
                        f"🚫 Insufficient volume confirmation (ratio: {volume_ratio:.2f}x, need 1.5x+). "
                        f"Low volume moves are unreliable. Wait for volume expansion."
                    )
            
            # CHECK 3: Momentum confirmation (2-3 recent candles in direction)
            if len(recent) >= 3:
                closes = recent['close'].values
                
                if signal_action == "buy":
                    bullish_candles = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
                    if bullish_candles < 2:
                        return False, (
                            f"🚫 No bullish momentum confirmation ({bullish_candles}/3 up candles). "
                            f"Wait for 2-3 consecutive up candles before BUY entry."
                        )
                elif signal_action == "sell":
                    bearish_candles = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i-1])
                    if bearish_candles < 2:
                        return False, (
                            f"🚫 No bearish momentum confirmation ({bearish_candles}/3 down candles). "
                            f"Wait for 2-3 consecutive down candles before SELL entry."
                        )
            
            # CHECK 4: Regime stability check (stored in instance variable)
            # This prevents entries right after regime changes (whipsaw protection)
            if regime_analysis and hasattr(self, 'last_regime_change_time'):
                from datetime import datetime, timezone, timedelta
                time_since_regime_change = datetime.now(timezone.utc) - self.last_regime_change_time
                if time_since_regime_change < timedelta(minutes=30):
                    return False, (
                        f"🚫 Regime changed {time_since_regime_change.seconds // 60} minutes ago. "
                        f"Wait 30 minutes after regime change to avoid whipsaws."
                    )
            
            return True, f"✅ Entry timing validated: confirmation requirements met"
            
        except Exception as e:
            logger.warning(f"⚠️ Entry timing validation error: {e}. Allowing trade to proceed.")
            return True, f"Entry timing check skipped due to error: {e}"
    
    def update_regime_tracking(self, current_regime: str) -> None:
        """
        Track regime changes to prevent entries during whipsaw periods.
        """
        if self.last_regime is not None and self.last_regime != current_regime:
            self.last_regime_change_time = datetime.now(timezone.utc)
            logger.info(f"📊 Regime changed: {self.last_regime} → {current_regime} (30-min cooldown started)")
        self.last_regime = current_regime
    
    def validate_confluence(
        self, signal: Dict, market_df, regime_analysis: Optional[Dict]
    ) -> Tuple[bool, str]:
        """
        PRIORITY 4: Require confluence of multiple factors before allowing trade.
        
        Professional traders wait for multiple confirmations, not just one indicator.
        This prevents low-probability trades where only 1-2 factors align.
        
        Confluence Factors (need 3 out of 4):
        1. MTF Alignment - Multiple timeframes agree on direction
        2. Volume Confirmation - Volume > 1.5x average
        3. Momentum Persistence - Momentum strength > 70
        4. Price Near S/R - Within 1% of support (BUY) or resistance (SELL)
        
        Returns:
            (is_valid, reason) tuple with confluence score
        """
        if market_df is None or market_df.empty or len(market_df) < 50:
            return True, "Insufficient data for confluence validation"
        
        signal_action = signal.get("signal", "hold").lower()
        if signal_action == "hold":
            return True, "Hold signal"
        
        try:
            latest = market_df.iloc[-1]
            entry_price = latest['close']
            
            confluence_factors = []
            factor_details = []
            
            # FACTOR 1: MTF Alignment (check if trend indicators align)
            mtf_data = signal.get("raw", {}).get("mtf_analysis", {})
            if mtf_data:
                primary_trend = mtf_data.get("primary_trend", "neutral")
                context_trend = mtf_data.get("context_trend", "neutral")
                
                if signal_action == "buy":
                    mtf_aligned = (primary_trend in ["bullish", "up"] and 
                                  context_trend in ["bullish", "up"])
                elif signal_action == "sell":
                    mtf_aligned = (primary_trend in ["bearish", "down"] and 
                                  context_trend in ["bearish", "down"])
                else:
                    mtf_aligned = False
                
                if mtf_aligned:
                    confluence_factors.append("MTF_ALIGNMENT")
                    factor_details.append(f"✅ MTF aligned ({primary_trend}/{context_trend})")
                else:
                    factor_details.append(f"❌ MTF not aligned ({primary_trend}/{context_trend})")
            
            # FACTOR 2: Volume Confirmation (>1.5x average)
            volume = latest.get('volume', 0)
            volume_sma = latest.get('volume_sma', volume)
            
            if volume > 0 and volume_sma > 0:
                volume_ratio = volume / volume_sma
                if volume_ratio >= 1.5:
                    confluence_factors.append("VOLUME_CONFIRMATION")
                    factor_details.append(f"✅ Volume confirmed ({volume_ratio:.2f}x)")
                else:
                    factor_details.append(f"❌ Volume weak ({volume_ratio:.2f}x)")
            
            # FACTOR 3: Momentum Persistence (>70 from regime analysis)
            if regime_analysis:
                momentum = regime_analysis.get("momentum_strength", 0)
                if momentum > 70:
                    confluence_factors.append("MOMENTUM_PERSISTENCE")
                    factor_details.append(f"✅ Strong momentum ({momentum:.0f})")
                else:
                    factor_details.append(f"❌ Weak momentum ({momentum:.0f})")
            
            # FACTOR 4: Price Near S/R (within 1% of key level)
            sr_levels = self.detect_support_resistance_levels(market_df, entry_price)
            support = sr_levels.get("nearest_support", 0)
            resistance = sr_levels.get("nearest_resistance", 0)
            
            if signal_action == "buy" and support > 0:
                distance_pct = abs(entry_price - support) / entry_price * 100
                if distance_pct <= 1.0:
                    confluence_factors.append("NEAR_SUPPORT")
                    factor_details.append(f"✅ Near support ({distance_pct:.2f}% away)")
                else:
                    factor_details.append(f"❌ Not near support ({distance_pct:.2f}% away)")
            elif signal_action == "sell" and resistance > 0:
                distance_pct = abs(entry_price - resistance) / entry_price * 100
                if distance_pct <= 1.0:
                    confluence_factors.append("NEAR_RESISTANCE")
                    factor_details.append(f"✅ Near resistance ({distance_pct:.2f}% away)")
                else:
                    factor_details.append(f"❌ Not near resistance ({distance_pct:.2f}% away)")
            
            # CONFLUENCE REQUIREMENT: Need 3 out of 4 factors
            confluence_score = len(confluence_factors)
            required_confluence = 3
            
            details_str = "\n   ".join(factor_details)
            
            if confluence_score >= required_confluence:
                return True, (
                    f"✅ Confluence validated ({confluence_score}/4 factors): "
                    f"{', '.join(confluence_factors)}\n   {details_str}"
                )
            else:
                return False, (
                    f"🚫 Insufficient confluence ({confluence_score}/4, need {required_confluence}). "
                    f"Only these factors aligned: {', '.join(confluence_factors) if confluence_factors else 'NONE'}. "
                    f"\n   {details_str}\n"
                    f"Wait for stronger setup with more confirmations."
                )
                
        except Exception as e:
            logger.warning(f"⚠️ Confluence validation error: {e}. Allowing trade to proceed.")
            return True, f"Confluence check skipped due to error: {e}"
    
    def validate_regime_strategy_alignment(
        self, signal: Dict, regime_analysis: Optional[Dict]
    ) -> Tuple[bool, str]:
        """
        Validate that the trading signal aligns with the current market regime.
        
        CRITICAL: Prevents counter-trend trading in strong directional markets.
        Professional traders don't fight strong trends - they wait or trade with them.
        
        Rules:
        1. BEAR_TREND + ADX > 30: Only SHORT signals allowed
        2. BULL_TREND + ADX > 30: Only LONG signals allowed
        3. MEAN_REVERSION edge in strong trend: BLOCKED
        4. Natural R:R < 2.0: Trade must be rejected at source
        
        Returns:
            (is_valid, reason) tuple
        """
        if not regime_analysis:
            return True, "No regime data - allowing trade"
        
        signal_action = signal.get("signal", "hold").lower()
        if signal_action == "hold":
            return True, "Hold signal"
        
        regime = regime_analysis.get("regime", "")
        adx = regime_analysis.get("trend_strength", 0)
        trading_edge = regime_analysis.get("trading_edge", "")
        confidence = regime_analysis.get("confidence", 0)
        
        # PRIORITY 3: Block ALL mean reversion trades (too risky with tight stops)
        # Don't trade mean reversion with sub-1% stops.
        # Either trade trends or wait for better setups
        if trading_edge == "MEAN_REVERSION":
            return False, (
                f"🚫 MEAN REVERSION trades BLOCKED (trading edge: {trading_edge}). "
                f"Mean reversion requires wide stops (2-3%+) and patience. "
                f"Day trading strategy focuses on momentum/trend trades only. "
                f"Wait for clear trend or range breakout setup."
            )
        
        # RULE 1: Block counter-trend mean reversion in strong trends (redundant but kept for clarity)
        if trading_edge == "MEAN_REVERSION":
            if regime == "BEAR_TREND" and adx > 30:
                return False, (
                    f"🚫 MEAN REVERSION blocked in strong BEAR_TREND (ADX={adx:.1f}). "
                    f"Do not buy falling knives. Wait for trend exhaustion or trade SHORT on bounces."
                )
            if regime == "BULL_TREND" and adx > 30:
                return False, (
                    f"🚫 MEAN REVERSION blocked in strong BULL_TREND (ADX={adx:.1f}). "
                    f"Do not short strong rallies. Wait for trend exhaustion or trade LONG on dips."
                )
        
        # RULE 2: Block BUY in strong bear trends
        if signal_action == "buy" and regime == "BEAR_TREND" and adx > 30:
            return False, (
                f"🚫 BUY blocked in strong BEAR_TREND (ADX={adx:.1f}, Confidence={confidence}%). "
                f"Professional traders don't catch falling knives. "
                f"Wait for regime change or trade SHORT on rallies."
            )
        
        # RULE 3: Block SELL in strong bull trends
        if signal_action == "sell" and regime == "BULL_TREND" and adx > 30:
            return False, (
                f"🚫 SELL blocked in strong BULL_TREND (ADX={adx:.1f}, Confidence={confidence}%). "
                f"Don't short strong uptrends. "
                f"Wait for regime change or trade LONG on dips."
            )
        
        # RULE 4: Warn about counter-trend trades in moderate trends
        if regime in ["BEAR_TREND", "BULL_TREND"] and 25 < adx <= 30:
            if (signal_action == "buy" and regime == "BEAR_TREND") or \
               (signal_action == "sell" and regime == "BULL_TREND"):
                logger.warning(
                    f"⚠️ Counter-trend signal in moderate {regime} (ADX={adx:.1f}). "
                    f"Allowing but with caution - ensure excellent setup quality."
                )
        
        return True, f"✅ Signal aligns with {regime} regime"

    def validate_natural_risk_reward(
        self, entry_price: float, stop_loss: float, take_profit: float, side: str
    ) -> Tuple[bool, str, float]:
        """
        Validate that the trade has a natural R:R ratio from market structure.
        
        CRITICAL: We should NOT force-fit R:R ratios. If the market doesn't
        naturally provide 2:1 R:R, the trade setup is poor and should be rejected.
        
        Professional traders walk away from trades that don't offer good R:R,
        they don't artificially adjust targets to meet arbitrary ratios.
        
        Returns:
            (is_valid, reason, actual_rrr) tuple
        """
        MIN_RRR = 2.0  # Minimum for mainnet profitability with fees
        
        # Calculate actual R:R from given levels
        if side.lower() == "buy":
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
        else:  # sell
            risk = abs(stop_loss - entry_price)
            reward = abs(entry_price - take_profit)
        
        if risk == 0:
            return False, "🚫 Invalid: Stop loss equals entry price (zero risk)", 0.0
        
        actual_rrr = reward / risk
        
        # Check if natural R:R meets minimum
        if actual_rrr < MIN_RRR:
            risk_pct = (risk / entry_price) * 100
            reward_pct = (reward / entry_price) * 100
            return False, (
                f"🚫 Natural R:R too low: {actual_rrr:.2f}:1 (need {MIN_RRR}:1 minimum). "
                f"Risk={risk_pct:.2f}%, Reward={reward_pct:.2f}%. "
                f"Market structure doesn't support this trade - SKIP IT. "
                f"Professional traders wait for better setups, they don't force bad R:R."
            ), actual_rrr
        
        # Excellent R:R
        if actual_rrr >= 3.0:
            logger.info(f"🎯 Excellent natural R:R: {actual_rrr:.2f}:1 - High-quality setup!")
        
        return True, f"✅ Natural R:R validated: {actual_rrr:.2f}:1", actual_rrr

    def validate_trade_signal(
        self,
        user_id: int,
        signal: Dict,
        account_balance: float,
        config: Dict,
        market_df=None,
        user_trade_history: Optional[Dict] = None,
    ) -> Tuple[bool, str, Dict]:
        """
        Validate if a trade signal meets risk management criteria.
        Enhanced for day trading with adaptive R:R based on market structure.

        Args:
            user_id: User identifier
            signal: AI trading signal
            account_balance: Current account balance
            config: Trading configuration
            market_df: Optional DataFrame for S/R detection and adaptive R:R
            user_trade_history: Optional user trading history for adaptive thresholds
        """
        try:
            self._assert_trading_active(config)

            existing_positions = self.get_open_positions(user_id)
            self._assert_no_open_positions(existing_positions)

            today_count = self.daily_trade_count.get(user_id, 0)
            self._assert_daily_limit(user_id, today_count, config)

            # Calculate adaptive confidence threshold (anti-overfitting)
            adaptive_threshold = self.get_adaptive_confidence_threshold(user_trade_history)
            
            confidence = self._extract_confidence(signal)
            self._assert_confidence(confidence, adaptive_threshold)

            balance = self._to_float(account_balance)
            self._assert_sufficient_balance(balance)

            # Get signal details
            entry_price = self._extract_entry_price(signal)
            side = signal.get("signal", "buy").lower()
            risk_percentage = self._resolve_risk_percentage(config)
            
            # VALIDATION #1: Check entry timing (PRIORITY 1)
            # Ensure we have momentum confirmation and aren't chasing extended moves
            is_timing_valid, timing_reason = self.validate_entry_timing(
                signal, market_df, signal.get("raw", {}).get("regime_analysis")
            )
            if not is_timing_valid:
                logger.warning(f"[User {user_id}] {timing_reason}")
                raise RiskValidationError(timing_reason)
            logger.info(f"[User {user_id}] {timing_reason}")
            
            # VALIDATION #2: Check regime-strategy alignment
            # Block counter-trend trades in strong directional markets
            regime_analysis = signal.get("raw", {}).get("regime_analysis")
            if regime_analysis:
                is_regime_valid, regime_reason = self.validate_regime_strategy_alignment(
                    signal, regime_analysis
                )
                if not is_regime_valid:
                    logger.warning(f"[User {user_id}] {regime_reason}")
                    raise RiskValidationError(regime_reason)
                logger.info(f"[User {user_id}] {regime_reason}")
            
            # VALIDATION #3: Check confluence of multiple factors (PRIORITY 4)
            # Require 3/4 confirmations: MTF alignment, volume, momentum, S/R proximity
            is_confluence_valid, confluence_reason = self.validate_confluence(
                signal, market_df, regime_analysis
            )
            if not is_confluence_valid:
                logger.warning(f"[User {user_id}] {confluence_reason}")
                raise RiskValidationError(confluence_reason)
            logger.info(f"[User {user_id}] {confluence_reason}")

            # Use adaptive R:R if market data available, otherwise use AI suggestions
            use_ai_levels = True  # Default to AI levels
            
            if market_df is not None and not market_df.empty:
                logger.info(
                    f"🎯 [User {user_id}] Testing S/R adaptive R:R based on market structure"
                )

                # Detect support/resistance levels
                sr_levels = self.detect_support_resistance_levels(
                    market_df, entry_price
                )

                # Get ATR for volatility context
                atr = market_df.iloc[-1].get(
                    "atr", entry_price * 0.02
                )  # Default 2% if missing

                # SAFETY CHECK: Only use S/R levels if they're reasonable
                support = sr_levels["nearest_support"]
                resistance = sr_levels["nearest_resistance"]
                support_distance = abs(support - entry_price) / entry_price * 100
                resistance_distance = abs(resistance - entry_price) / entry_price * 100
                
                # Reject S/R levels if they're more than 8% away (unrealistic)
                if support_distance <= 8.0 and resistance_distance <= 8.0:
                    logger.info(f"✅ S/R levels reasonable: Support={support_distance:.1f}%, Resistance={resistance_distance:.1f}%")
                    
                    # Calculate adaptive stop loss and take profit
                    adaptive_rr = self.calculate_adaptive_risk_reward(
                        side, entry_price, sr_levels, atr
                    )

                    stop_loss = adaptive_rr["stop_loss"]
                    take_profit = adaptive_rr["take_profit"]
                    risk_reward_ratio = adaptive_rr["risk_reward_ratio"]
                    use_ai_levels = False  # Use S/R levels
                    
                    logger.info(
                        f"📊 [User {user_id}] S/R Adaptive R:R applied:"
                        f"\n   Entry: ${entry_price:.2f}"
                        f"\n   Stop Loss: ${stop_loss:.2f} (S/R: ${sr_levels['nearest_support' if side == 'buy' else 'nearest_resistance']:.2f})"
                        f"\n   Take Profit: ${take_profit:.2f} (S/R: ${sr_levels['nearest_resistance' if side == 'buy' else 'nearest_support']:.2f})"
                        f"\n   R:R Ratio: 1:{risk_reward_ratio:.2f}"
                    )
                else:
                    logger.warning(
                        f"⚠️ S/R levels unrealistic! Support={support_distance:.1f}%, Resistance={resistance_distance:.1f}%. "
                        f"Falling back to AI levels for safety."
                    )
                    
            if use_ai_levels:
                # Use AI-suggested levels (safer default)
                logger.info(f"🤖 [User {user_id}] Using AI-suggested R:R levels (safer)")
                stop_loss = self._extract_stop_loss(signal)
                take_profit = self._extract_take_profit(signal)
                risk_reward_ratio = self._calculate_ai_risk_reward(
                    entry_price, stop_loss, take_profit
                )

                logger.info(
                    f"📊 [User {user_id}] AI R:R applied:"
                    f"\n   Entry: ${entry_price:.2f}"
                    f"\n   Stop Loss: ${stop_loss:.2f}"
                    f"\n   Take Profit: ${take_profit:.2f}"
                    f"\n   R:R Ratio: 1:{risk_reward_ratio:.2f}"
                )
            else:
                # Fallback to AI-suggested levels
                logger.warning(
                    f"⚠️ [User {user_id}] No market data for adaptive R:R, using AI suggestions"
                )
                entry_price, stop_loss = self._extract_prices(signal)
                take_profit = self._resolve_take_profit(signal, entry_price)
                risk_reward_ratio = abs(take_profit - entry_price) / abs(
                    entry_price - stop_loss
                )
            
            # VALIDATION #2: Validate natural R:R from market structure
            # Reject trades where market doesn't naturally provide 2:1 R:R
            is_rrr_valid, rrr_reason, actual_rrr = self.validate_natural_risk_reward(
                entry_price, stop_loss, take_profit, side
            )
            if not is_rrr_valid:
                logger.error(f"[User {user_id}] {rrr_reason}")
                raise RiskValidationError(rrr_reason)
            
            logger.info(f"[User {user_id}] {rrr_reason}")
            risk_reward_ratio = actual_rrr  # Use validated natural R:R

            # Calculate position size using 1% rule
            position_size = self.calculate_position_size(
                balance,
                risk_percentage,
                entry_price,
                stop_loss,
            )

            self._assert_position_size(position_size)

            trade_value = self._compute_trade_value(position_size, entry_price)
            self._assert_trade_value(trade_value, balance)

            trade_params = {
                "position_size": position_size,
                "trade_value": trade_value,
                "risk_amount": self._risk_amount(balance, risk_percentage),
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_reward_ratio": risk_reward_ratio,
                "side": side,
            }

            return True, "Trade approved", trade_params

        except RiskValidationError as err:
            return False, err.message, {}
        except Exception as e:
            logger.error(f"❌ Trade validation error: {e}")
            return False, f"Validation error: {str(e)}", {}

    def record_trade(self, user_id: int):
        """Record a trade for daily limit tracking"""
        self.daily_trade_count[user_id] = self.daily_trade_count.get(user_id, 0) + 1

    def add_open_position(self, user_id: int, position_data: Dict):
        """Add a new open position for tracking"""
        if user_id not in self.open_positions:
            self.open_positions[user_id] = []

        position = {
            "trade_id": position_data["trade_id"],
            "symbol": position_data["symbol"],
            "side": position_data["side"],
            "amount": position_data["amount"],
            "entry_price": position_data["entry_price"],
            "stop_loss": position_data["stop_loss"],
            "take_profit": position_data["take_profit"],
            "entry_time": position_data["entry_time"],
            "entry_value": position_data["entry_value"],
            "fees_paid": position_data["fees_paid"],
        }

        self.open_positions[user_id].append(position)
        logger.info(
            f"📋 [User {user_id}] Position added: {position['side']} {position['amount']:.6f} {position['symbol']} @ ${position['entry_price']:.4f}"
        )

    def get_open_positions(self, user_id: int) -> list:
        """Get all open positions for a user"""
        return self.open_positions.get(user_id, [])

    def check_exit_conditions(
        self, user_id: int, current_price: float, symbol: str
    ) -> list:
        """Check if any positions should be closed based on current price"""
        positions_to_close = []

        if user_id not in self.open_positions:
            return positions_to_close

        for position in self.open_positions[user_id]:
            if position["symbol"] != symbol:
                continue

            exit_reason = self._determine_exit_reason(position, current_price)
            if exit_reason:
                positions_to_close.append(
                    {
                        "position": position,
                        "exit_reason": exit_reason,
                        "current_price": current_price,
                    }
                )

        return positions_to_close

    def reset_daily_counters(self):
        """Reset daily trade counters (called at midnight)"""
        self.daily_trade_count.clear()
        logger.info("Daily trade counters reset")

    async def get_adaptive_exit_levels(
        self, db, user_id: int, df, side: str, entry_price: float, regime_analysis: Dict = None
    ) -> Dict:
        """
        Calculate optimal SL/TP based on:
        1. Current market volatility (ATR) - inversely scaled
        2. Historical user performance in similar volatility
        3. Safety caps to prevent excessive risk
        4. **NEW: Mean reversion detection for wider stops**

        Philosophy:
        - High volatility → Wider stops (larger ATR multipliers) to avoid premature exits
        - Low volatility → Tighter stops (smaller ATR multipliers) for precision
        - **Mean reversion setups → Extra wide stops (need breathing room for bounce)**

        Returns:
            {
                "stop_loss_pct": float,
                "take_profit_pct": float,
                "stop_loss_price": float,
                "take_profit_price": float,
                "source": str,
                "atr_value": float,
                "atr_pct": float,
                "risk_reward_ratio": float,
                "volatility_regime": str
            }
        """
        try:
            # 1. ATR-Based Dynamic Stops with Adaptive Multipliers
            atr = df["atr"].iloc[-1]
            current_price = df["close"].iloc[-1]
            atr_percentage = (atr / current_price) * 100

            # Calculate ATR percentile to determine volatility regime
            atr_history = df["atr"].tail(100) if len(df) >= 100 else df["atr"]
            from scipy.stats import percentileofscore

            volatility_percentile = percentileofscore(atr_history, atr)

            # Store volatility percentile for use by position sizing
            self.current_volatility_percentile = volatility_percentile
            
            # Check if this is a mean reversion setup (requires wider stops)
            is_mean_reversion = False
            if regime_analysis:
                trading_edge = regime_analysis.get("trading_edge", "")
                mean_reversion_score = regime_analysis.get("mean_reversion_score", 0)
                # NOTE: This check should never be true due to earlier blocking,
                # but included as failsafe
                if trading_edge == "MEAN_REVERSION" or mean_reversion_score >= 70:
                    is_mean_reversion = True
                    logger.warning(
                        f"⚠️ Mean reversion setup detected - will apply wider stops if allowed through"
                    )

            # PRIORITY 2: Intelligent ATR-Based Stop Widening
            # Minimum 2x ATR for stops, 5x ATR for targets (ensures 2.5:1 R:R minimum)
            # Professional stops account for market noise, not arbitrary percentages
            
            if is_mean_reversion:
                # Mean reversion BLOCKED elsewhere, but failsafe here
                sl_multiplier = 3.5
                tp_multiplier = 8.75  # 2.5:1 R:R
                volatility_regime = "MEAN_REVERSION"
                logger.warning(
                    f"⚠️ Mean reversion setup - using EXTRA WIDE stops (should be blocked earlier)"
                )
            elif volatility_percentile > 75:
                # HIGH VOLATILITY: Minimum 2.5x ATR stops to avoid noise
                sl_multiplier = 2.5
                tp_multiplier = 6.25  # 2.5:1 R:R
                volatility_regime = "HIGH"
            elif volatility_percentile > 50:
                # MEDIUM-HIGH VOLATILITY: 2.2x ATR stops
                sl_multiplier = 2.2
                tp_multiplier = 5.5  # 2.5:1 R:R
                volatility_regime = "MEDIUM_HIGH"
            elif volatility_percentile > 25:
                # MEDIUM VOLATILITY: 2.0x ATR stops
                sl_multiplier = 2.0
                tp_multiplier = 5.0  # 2.5:1 R:R
                volatility_regime = "MEDIUM"
            else:
                # LOW VOLATILITY: Still 2x ATR minimum (noise protection)
                sl_multiplier = 2.0
                tp_multiplier = 5.0  # 2.5:1 R:R
                volatility_regime = "LOW"

            # Calculate percentage distances based on ATR multipliers
            dynamic_sl_pct = atr_percentage * sl_multiplier
            dynamic_tp_pct = atr_percentage * tp_multiplier
            
            logger.info(
                f"📏 ATR-based stops: Volatility={volatility_regime} ({volatility_percentile:.0f}th percentile), "
                f"ATR={atr_percentage:.2f}%, SL={sl_multiplier}x ATR ({dynamic_sl_pct:.2f}%), "
                f"TP={tp_multiplier}x ATR ({dynamic_tp_pct:.2f}%)"
            )

            # 2. Historical Performance Adjustment (Volatility-Aware)
            try:
                from sqlalchemy import select
                from ..models.trade import Trade

                query = (
                    select(Trade)
                    .where(Trade.user_id == user_id, Trade.status == "closed")
                    .order_by(Trade.closed_at.desc())
                    .limit(50)
                )

                result = await db.execute(query)
                trades_list = result.scalars().all()

                if len(trades_list) >= 10:
                    # Calculate average actual exit percentage for wins/losses
                    winning_exits = [
                        abs(float(t.profit_loss_percentage))
                        for t in trades_list
                        if t.profit_loss
                        and float(t.profit_loss) > 0
                        and t.profit_loss_percentage
                    ]
                    losing_exits = [
                        abs(float(t.profit_loss_percentage))
                        for t in trades_list
                        if t.profit_loss
                        and float(t.profit_loss) < 0
                        and t.profit_loss_percentage
                    ]

                    if winning_exits and len(winning_exits) >= 5:
                        # TP: Take 75% of average winning exit (conservative)
                        import numpy as np

                        historical_tp = np.mean(winning_exits) * 0.75
                        # Only adjust if historical is significantly larger
                        if historical_tp > dynamic_tp_pct * 1.2:
                            dynamic_tp_pct = min(dynamic_tp_pct * 1.3, historical_tp)

                    if losing_exits and len(losing_exits) >= 5:
                        # SL: Use 110% of average losing exit (slightly more room)
                        import numpy as np

                        historical_sl = np.mean(losing_exits) * 1.1
                        # Only adjust if historical suggests wider stops needed
                        if historical_sl > dynamic_sl_pct:
                            dynamic_sl_pct = min(dynamic_sl_pct * 1.2, historical_sl)

            except Exception as hist_error:
                logger.warning(
                    f"⚠️ Could not fetch historical performance for adaptive exits: {hist_error}"
                )
                # Continue with ATR-based values

            # 3. Fee-Adjusted Minimum Thresholds
            # CRITICAL: Stops must be wide enough that fees don't destroy R:R
            # With 0.2% total fees (0.1% entry + 0.1% exit), need minimum thresholds
            
            FEE_PERCENTAGE = 0.2  # 0.1% entry + 0.1% exit
            
            # Absolute minimums accounting for fees (for 2.5:1 R:R after fees)
            # Example: 0.8% SL - 0.2% fees = 0.6% actual risk
            #          2.0% TP - 0.2% fees = 1.8% actual reward
            #          Actual R:R = 1.8 / 0.6 = 3:1 → gives 2.5:1 after slippage
            MIN_SL_FOR_FEES = 0.8  # Minimum to maintain viable R:R after fees
            MIN_TP_FOR_FEES = 2.0  # Minimum to make profit meaningful after fees
            
            # Apply fee-adjusted minimums
            dynamic_sl_pct = max(dynamic_sl_pct, MIN_SL_FOR_FEES)
            dynamic_tp_pct = max(dynamic_tp_pct, MIN_TP_FOR_FEES)
            
            # Safety maximums (prevent excessive risk)
            ABSOLUTE_MAX_STOP_LOSS = 2.5  # Increased from 0.5% - day trading can handle wider stops
            ABSOLUTE_MAX_TAKE_PROFIT = 6.0  # Increased from 2.0% - let winners run
            
            dynamic_sl_pct = min(dynamic_sl_pct, ABSOLUTE_MAX_STOP_LOSS)
            dynamic_tp_pct = min(dynamic_tp_pct, ABSOLUTE_MAX_TAKE_PROFIT)

            # Ensure minimum risk:reward ratio of 3.0:1 (PRIORITY 2)
            # With 40% win rate, need wider R:R for profitability
            # 3:1 R:R means each win covers 3 losses (sustainable edge)
            min_risk_reward = 3.0
            if dynamic_tp_pct / dynamic_sl_pct < min_risk_reward:
                logger.warning(
                    f"⚠️  Risk:reward ratio too low ({dynamic_tp_pct/dynamic_sl_pct:.2f}:1). "
                    f"Adjusting TP to maintain {min_risk_reward}:1 minimum (accounts for fees + 40% win rate)"
                )
                dynamic_tp_pct = max(dynamic_tp_pct, dynamic_sl_pct * min_risk_reward)

            # 4. Calculate actual prices based on side
            if side.lower() == "buy":
                stop_loss_price = entry_price * (1 - dynamic_sl_pct / 100)
                take_profit_price = entry_price * (1 + dynamic_tp_pct / 100)
            else:  # sell
                stop_loss_price = entry_price * (1 + dynamic_sl_pct / 100)
                take_profit_price = entry_price * (1 - dynamic_tp_pct / 100)

            result = {
                "stop_loss_pct": dynamic_sl_pct,
                "take_profit_pct": dynamic_tp_pct,
                "stop_loss_price": stop_loss_price,
                "take_profit_price": take_profit_price,
                "source": "dynamic_atr",
                "atr_value": float(atr),
                "atr_pct": atr_percentage,
                "risk_reward_ratio": dynamic_tp_pct / dynamic_sl_pct,
                "volatility_regime": volatility_regime,
                "volatility_percentile": volatility_percentile,
                "sl_multiplier": sl_multiplier,
                "tp_multiplier": tp_multiplier,
            }

            logger.info(
                f"📊 [User {user_id}] Adaptive Exits ({volatility_regime} Vol): "
                f"SL={dynamic_sl_pct:.2f}% (${stop_loss_price:.4f}) [{sl_multiplier}x ATR], "
                f"TP={dynamic_tp_pct:.2f}% (${take_profit_price:.4f}) [{tp_multiplier}x ATR], "
                f"R:R={result['risk_reward_ratio']:.2f}:1, "
                f"ATR={atr_percentage:.3f}%, Vol_Pct={volatility_percentile:.0f}"
            )

            return result

        except Exception as e:
            logger.error(f"❌ Error calculating adaptive exit levels: {e}")
            # Fallback to fixed percentages
            if side.lower() == "buy":
                return {
                    "stop_loss_pct": 0.5,
                    "take_profit_pct": 0.3,
                    "stop_loss_price": entry_price * 0.995,
                    "take_profit_price": entry_price * 1.003,
                    "source": "fallback_fixed",
                    "atr_value": 0,
                    "atr_pct": 0,
                    "risk_reward_ratio": 0.6,
                }
            else:
                return {
                    "stop_loss_pct": 0.5,
                    "take_profit_pct": 0.3,
                    "stop_loss_price": entry_price * 1.005,
                    "take_profit_price": entry_price * 0.997,
                    "source": "fallback_fixed",
                    "atr_value": 0,
                    "atr_pct": 0,
                    "risk_reward_ratio": 0.6,
                }

    @staticmethod
    def _to_float(value: Union[float, Decimal, int]) -> float:
        return float(value)

    @staticmethod
    def _risk_amount(balance: float, risk_percentage: float) -> float:
        return balance * (risk_percentage / 100)

    def _enforce_max_position(
        self, position_size: float, entry_price: float, balance: float
    ) -> float:
        max_position_value = balance * MAX_BALANCE_TRADE_RATIO
        if position_size * entry_price > max_position_value:
            adjusted = max_position_value / entry_price
            logger.debug(
                "Position size adjusted for max balance ratio: %.8f -> %.8f",
                position_size,
                adjusted,
            )
            return adjusted
        return position_size

    @staticmethod
    def _truncate(position_size: float) -> float:
        truncated = int(position_size * EIGHT_DECIMAL_PLACES) / EIGHT_DECIMAL_PLACES
        return float(truncated)

    @staticmethod
    def _compute_trade_value(position_size: float, entry_price: float) -> float:
        return position_size * entry_price

    @staticmethod
    def _extract_confidence(signal: Dict) -> float:
        return float(signal.get("final_confidence", 0))

    @staticmethod
    def _extract_entry_price(signal: Dict) -> float:
        """Extract entry price from signal."""
        entry_price = float(signal.get("entry_price", 0))
        if entry_price <= 0:
            raise RiskValidationError("Missing or invalid entry price in signal")
        return entry_price

    @staticmethod
    def _extract_prices(signal: Dict) -> Tuple[float, float]:
        entry_price = float(signal.get("entry_price", 0))
        stop_loss = float(signal.get("stop_loss", 0))
        if entry_price <= 0 or stop_loss <= 0:
            raise RiskValidationError("Missing entry price or stop loss in signal")
        return entry_price, stop_loss

    @staticmethod
    def _resolve_risk_percentage(config: Dict) -> float:
        return float(config.get("risk_percentage", DEFAULT_RISK_PERCENTAGE))

    @staticmethod
    def _assert_position_size(position_size: float) -> None:
        if position_size <= 0:
            raise RiskValidationError("Calculated position size is zero")

    @staticmethod
    def _assert_confidence(confidence: float, adaptive_threshold: Optional[float] = None) -> None:
        """
        Validate confidence meets threshold.
        
        Args:
            confidence: Signal confidence value
            adaptive_threshold: Optional adaptive threshold (if None, uses MIN_SIGNAL_CONFIDENCE)
        """
        threshold = adaptive_threshold if adaptive_threshold is not None else MIN_SIGNAL_CONFIDENCE
        if confidence < threshold:
            raise RiskValidationError(
                f"Signal confidence too low: {confidence:.1f}% < {threshold:.1f}%"
            )

    @staticmethod
    def _assert_sufficient_balance(balance: float) -> None:
        if balance < MIN_ACCOUNT_BALANCE:
            raise RiskValidationError("Insufficient account balance for trading")

    @staticmethod
    def _assert_trade_value(trade_value: float, balance: float) -> None:
        if trade_value < MIN_TRADE_VALUE:
            raise RiskValidationError(f"Trade value too small: ${trade_value:.2f}")
        if trade_value > balance * MAX_BALANCE_TRADE_RATIO:
            raise RiskValidationError(
                f"Trade value exceeds {MAX_BALANCE_TRADE_RATIO*100:.1f}% of balance"
            )

    @staticmethod
    def _assert_trading_active(config: Dict) -> None:
        if not config.get("is_active", False):
            raise RiskValidationError("Trading bot is not active")

    def _assert_no_open_positions(self, positions: List[Dict]) -> None:
        """Check if user has reached the maximum number of open positions."""
        if len(positions) >= MAX_OPEN_POSITIONS:
            position_symbols = [p['symbol'] for p in positions]
            raise RiskValidationError(
                f"User already has {len(positions)} open position(s) ({', '.join(position_symbols)}). "
                f"Maximum allowed: {MAX_OPEN_POSITIONS}."
            )

    def _assert_daily_limit(self, user_id: int, today_count: int, config: Dict) -> None:
        max_trades = self.max_daily_trades.get(
            user_id, config.get("max_daily_trades", DEFAULT_MAX_DAILY_TRADES)
        )
        if today_count >= max_trades:
            raise RiskValidationError(f"Daily trade limit reached ({max_trades})")

    @staticmethod
    def _calculate_pnl(position: Dict, exit_price: float) -> float:
        amount = position["amount"]
        entry_price = position["entry_price"]
        side = position["side"].lower()
        if side == "buy":
            return (exit_price - entry_price) * amount
        return (entry_price - exit_price) * amount

    @staticmethod
    def _ensure_timezone(entry_time: datetime) -> datetime:
        if entry_time.tzinfo is None:
            return entry_time.replace(tzinfo=timezone.utc)
        return entry_time

    @staticmethod
    def _determine_exit_reason(position: Dict, current_price: float) -> Optional[str]:
        side = position["side"].lower()
        take_profit = position["take_profit"]
        stop_loss = position["stop_loss"]

        if side == "buy":
            if current_price >= take_profit:
                return "TAKE_PROFIT"
            if current_price <= stop_loss:
                return "STOP_LOSS"
        else:
            if current_price <= take_profit:
                return "TAKE_PROFIT"
            if current_price >= stop_loss:
                return "STOP_LOSS"
        return None

    @staticmethod
    def _resolve_take_profit(signal: Dict, entry_price: float) -> float:
        return float(signal.get("take_profit") or (entry_price * 1.003))

    def detect_support_resistance_levels(self, df, current_price: float) -> Dict:
        """
        Detect key support and resistance levels using pivot points and price action.

        Returns dict with:
            - nearest_support: Closest support level below current price
            - nearest_resistance: Closest resistance level above current price
            - support_strength: Number of times support was tested
            - resistance_strength: Number of times resistance was tested
        """
        from .service_constants import (
            SR_LOOKBACK_PERIODS,
            SR_TOUCH_THRESHOLD,
            SR_MIN_TOUCHES,
        )

        try:
            if df.empty or len(df) < 20:
                return self._default_sr_levels(current_price)

            lookback = min(SR_LOOKBACK_PERIODS, len(df))
            df_analysis = df.tail(lookback).copy()

            # Identify pivot points (local highs and lows)
            df_analysis["pivot_high"] = (
                df_analysis["high"] > df_analysis["high"].shift(1)
            ) & (df_analysis["high"] > df_analysis["high"].shift(-1))
            df_analysis["pivot_low"] = (
                df_analysis["low"] < df_analysis["low"].shift(1)
            ) & (df_analysis["low"] < df_analysis["low"].shift(-1))

            # Extract pivot levels
            resistance_candidates = df_analysis[df_analysis["pivot_high"]][
                "high"
            ].tolist()
            support_candidates = df_analysis[df_analysis["pivot_low"]]["low"].tolist()

            # Cluster nearby levels (within 0.2% of each other)
            resistance_levels = self._cluster_price_levels(
                resistance_candidates, current_price, SR_TOUCH_THRESHOLD
            )
            support_levels = self._cluster_price_levels(
                support_candidates, current_price, SR_TOUCH_THRESHOLD
            )

            # Find nearest levels
            nearest_resistance = self._find_nearest_above(
                resistance_levels, current_price
            )
            nearest_support = self._find_nearest_below(support_levels, current_price)

            # Calculate strength (number of touches)
            resistance_strength = self._count_touches(
                df_analysis, nearest_resistance, SR_TOUCH_THRESHOLD
            )
            support_strength = self._count_touches(
                df_analysis, nearest_support, SR_TOUCH_THRESHOLD
            )

            logger.debug(
                f"📍 S/R Levels: Support=${nearest_support:.2f} (strength:{support_strength}), "
                f"Resistance=${nearest_resistance:.2f} (strength:{resistance_strength}), "
                f"Current=${current_price:.2f}"
            )

            return {
                "nearest_support": nearest_support,
                "nearest_resistance": nearest_resistance,
                "support_strength": support_strength,
                "resistance_strength": resistance_strength,
                "support_distance_pct": abs(current_price - nearest_support)
                / current_price
                * 100,
                "resistance_distance_pct": abs(nearest_resistance - current_price)
                / current_price
                * 100,
            }

        except Exception as e:
            logger.error(f"❌ Error detecting S/R levels: {e}")
            return self._default_sr_levels(current_price)

    def _cluster_price_levels(
        self, levels: List[float], current_price: float, threshold: float
    ) -> List[float]:
        """Cluster nearby price levels to identify significant zones."""
        if not levels:
            return []

        clustered = []
        sorted_levels = sorted(levels)

        for level in sorted_levels:
            if not clustered:
                clustered.append(level)
                continue

            # Check if this level is close to the last clustered level
            if abs(level - clustered[-1]) / current_price < threshold:
                # Update cluster with average
                clustered[-1] = (clustered[-1] + level) / 2
            else:
                clustered.append(level)

        return clustered

    def _find_nearest_above(self, levels: List[float], price: float) -> float:
        """Find nearest level above current price."""
        # CRITICAL: Filter out insane levels (more than 10% away)
        MAX_DISTANCE_PCT = 0.10  # 10% maximum distance
        max_price = price * (1 + MAX_DISTANCE_PCT)
        
        above = [l for l in levels if price < l <= max_price]
        return min(above) if above else price * 1.03  # Default 3% above (conservative)

    def _find_nearest_below(self, levels: List[float], price: float) -> float:
        """Find nearest level below current price."""
        # CRITICAL: Filter out insane levels (more than 10% away)
        MAX_DISTANCE_PCT = 0.10  # 10% maximum distance
        min_price = price * (1 - MAX_DISTANCE_PCT)
        
        below = [l for l in levels if min_price <= l < price]
        return max(below) if below else price * 0.97  # Default 3% below (conservative)

    def _count_touches(self, df, level: float, threshold: float) -> int:
        """Count how many times price touched a level."""
        if df.empty:
            return 0

        touches = 0
        for _, row in df.iterrows():
            high = row["high"]
            low = row["low"]
            if (
                abs(high - level) / level < threshold
                or abs(low - level) / level < threshold
            ):
                touches += 1

        return touches

    def _default_sr_levels(self, current_price: float) -> Dict:
        """Default S/R levels when detection fails."""
        return {
            "nearest_support": current_price * 0.98,  # 2% below
            "nearest_resistance": current_price * 1.02,  # 2% above
            "support_strength": 1,
            "resistance_strength": 1,
            "support_distance_pct": 2.0,
            "resistance_distance_pct": 2.0,
        }

    def calculate_structure_based_stop(
        self, side: str, entry_price: float, sr_levels: Dict, atr: float
    ) -> Dict:
        """
        Calculate structure-based stop loss at actual invalidation levels.
        
        Philosophy: Place stops where the trade setup is INVALIDATED, not at arbitrary percentages.
        - BUY: Stop goes below support (if support breaks, bullish thesis invalidated)
        - SELL: Stop goes above resistance (if resistance breaks, bearish thesis invalidated)
        
        Buffer: 1.0x ATR beyond the structure level to avoid premature stops on wicks.
        
        Returns dict with:
            - stop_loss: Invalidation price level
            - stop_distance_pct: Distance from entry as percentage
            - invalidation_level: The actual S/R level used
            - buffer_used: ATR buffer applied
        """
        from .service_constants import (
            MIN_STOP_DISTANCE_PCT,
            MAX_STOP_DISTANCE_PCT,
            STRUCTURE_STOP_ATR_BUFFER,
        )
        
        try:
            side = side.lower()
            
            if side == "buy":
                # For BUY: Invalidation = support break
                # Stop goes below support with ATR buffer
                invalidation_level = sr_levels["nearest_support"]
                atr_buffer = atr * STRUCTURE_STOP_ATR_BUFFER
                stop_loss = invalidation_level - atr_buffer
                
                logger.info(
                    f"🛡️ BUY Structure Stop: Support=${invalidation_level:.2f}, "
                    f"Buffer={atr_buffer:.2f} ({STRUCTURE_STOP_ATR_BUFFER}x ATR), "
                    f"Final Stop=${stop_loss:.2f}"
                )
                
            else:  # sell
                # For SELL: Invalidation = resistance break
                # Stop goes above resistance with ATR buffer
                invalidation_level = sr_levels["nearest_resistance"]
                atr_buffer = atr * STRUCTURE_STOP_ATR_BUFFER
                stop_loss = invalidation_level + atr_buffer
                
                logger.info(
                    f"🛡️ SELL Structure Stop: Resistance=${invalidation_level:.2f}, "
                    f"Buffer={atr_buffer:.2f} ({STRUCTURE_STOP_ATR_BUFFER}x ATR), "
                    f"Final Stop=${stop_loss:.2f}"
                )
            
            # Calculate distance from entry
            stop_distance = abs(entry_price - stop_loss)
            stop_distance_pct = (stop_distance / entry_price) * 100
            
            # Enforce minimum distance (prevent too-tight stops in low volatility)
            min_distance = entry_price * MIN_STOP_DISTANCE_PCT
            if stop_distance < min_distance:
                logger.warning(
                    f"⚠️ Structure stop too tight ({stop_distance_pct:.2f}%). "
                    f"Widening to minimum {MIN_STOP_DISTANCE_PCT*100:.1f}%"
                )
                if side == "buy":
                    stop_loss = entry_price - min_distance
                else:
                    stop_loss = entry_price + min_distance
                stop_distance = min_distance
                stop_distance_pct = MIN_STOP_DISTANCE_PCT * 100
            
            # Enforce maximum distance (prevent insane stops in high volatility)
            max_distance = entry_price * MAX_STOP_DISTANCE_PCT
            if stop_distance > max_distance:
                logger.error(
                    f"🚨 Structure stop too wide ({stop_distance_pct:.2f}%). "
                    f"Capping at maximum {MAX_STOP_DISTANCE_PCT*100:.1f}%"
                )
                if side == "buy":
                    stop_loss = entry_price - max_distance
                else:
                    stop_loss = entry_price + max_distance
                stop_distance = max_distance
                stop_distance_pct = MAX_STOP_DISTANCE_PCT * 100
            
            logger.info(
                "✅ Structure-Based Stop: $%.2f (%.2f%% from entry $%.2f)",
                stop_loss,
                stop_distance_pct,
                entry_price,
            )
            
            return {
                "stop_loss": stop_loss,
                "stop_distance_pct": stop_distance_pct,
                "invalidation_level": invalidation_level,
                "buffer_used": atr_buffer,
                "method": "structure_based"
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating structure-based stop: {e}")
            # Fallback to safe default (1% stop)
            default_distance = entry_price * 0.01
            if side == "buy":
                stop_loss = entry_price - default_distance
            else:
                stop_loss = entry_price + default_distance
            
            return {
                "stop_loss": stop_loss,
                "stop_distance_pct": 1.0,
                "invalidation_level": entry_price,
                "buffer_used": 0.0,
                "method": "fallback_default"
            }

    def calculate_adaptive_risk_reward(
        self, side: str, entry_price: float, sr_levels: Dict, atr: float
    ) -> Dict:
        """
        Calculate adaptive stop-loss and take-profit based on market structure.

        Implements minimum 1:1.5 R:R with dynamic adjustment based on:
        - Support/resistance levels
        - Market volatility (ATR)
        - Side of trade (buy/sell)
        
        CRITICAL: Enforces minimum distance thresholds to prevent unrealistic tight levels
        - Minimum SL distance: 0.8% from entry (prevents whipsaw)
        - Minimum TP distance: 1.2% from entry (realistic profit target)

        Returns dict with stop_loss, take_profit, and actual risk_reward_ratio
        """
        from .service_constants import (
            MIN_RISK_REWARD_RATIO,
            TARGET_RISK_REWARD_RATIO,
            MAX_RISK_REWARD_RATIO,
        )

        # ANTI-OVERFITTING: Minimum percentage thresholds for realistic trading
        MIN_STOP_LOSS_PCT = 0.008  # 0.8% minimum SL distance (prevents tight stops)
        MIN_TAKE_PROFIT_PCT = 0.012  # 1.2% minimum TP distance (realistic profit)
        MAX_STOP_LOSS_PCT = 0.05   # 5.0% MAXIMUM SL distance (prevents insane stops)
        
        # ADAPTIVE STOP WIDENING: Scale stops with volatility regime
        # Low volatility = widen stops to avoid noise; High volatility = tighter relative stops
        atr_percentage = (atr / entry_price) * 100
        
        if atr_percentage < 0.50:
            # Very low volatility - widen stops significantly to avoid whipsaw
            stop_distance_multiplier = 2.0
            logger.info(
                f"📏 LOW VOLATILITY (ATR%: {atr_percentage:.2f}%) - "
                f"Widening stops by {stop_distance_multiplier}x to avoid noise"
            )
        elif atr_percentage < 0.70:
            # Moderate-low volatility - slightly wider stops
            stop_distance_multiplier = 1.5
            logger.info(
                f"📏 MODERATE-LOW VOLATILITY (ATR%: {atr_percentage:.2f}%) - "
                f"Widening stops by {stop_distance_multiplier}x"
            )
        elif atr_percentage > 1.5:
            # High volatility - tighter relative stops (but still using ATR buffer)
            stop_distance_multiplier = 1.2
            logger.info(
                f"📏 HIGH VOLATILITY (ATR%: {atr_percentage:.2f}%) - "
                f"Using tighter relative stops ({stop_distance_multiplier}x)"
            )
        else:
            # Optimal volatility range (0.70-1.50%) - normal stops
            stop_distance_multiplier = 1.5
            logger.info(
                f"📏 OPTIMAL VOLATILITY (ATR%: {atr_percentage:.2f}%) - "
                f"Using standard stops ({stop_distance_multiplier}x ATR buffer)"
            )
        
        try:
            side = side.lower()

            if side == "buy":
                # Stop loss: Just below nearest support
                support = sr_levels["nearest_support"]
                buffer = atr * 0.5 * stop_distance_multiplier  # Apply adaptive multiplier
                stop_loss = support - buffer

                # Take profit: Just below nearest resistance
                resistance = sr_levels["nearest_resistance"]
                take_profit = resistance - (atr * 0.3)  # Slightly below resistance

            else:  # sell
                # Stop loss: Just above nearest resistance
                resistance = sr_levels["nearest_resistance"]
                buffer = atr * 0.5 * stop_distance_multiplier  # Apply adaptive multiplier
                stop_loss = resistance + buffer

                # Take profit: Just above nearest support
                support = sr_levels["nearest_support"]
                take_profit = support + (atr * 0.3)

            # Calculate actual R:R ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)

            if risk == 0:
                logger.warning("⚠️ Risk is zero, using default levels")
                return self._default_exit_levels(side, entry_price)

            # CRITICAL: Enforce MAXIMUM stop loss distance (5% cap)
            max_sl_distance = entry_price * MAX_STOP_LOSS_PCT  # 5.0% maximum
            if risk > max_sl_distance:
                logger.error(
                    f"🚨 INSANE STOP LOSS: {(risk/entry_price*100):.2f}% is way too wide! "
                    f"Capping at maximum {MAX_STOP_LOSS_PCT*100:.1f}% for safety."
                )
                if side == "buy":
                    stop_loss = entry_price - max_sl_distance
                else:
                    stop_loss = entry_price + max_sl_distance
                risk = max_sl_distance

            # CRITICAL: Enforce minimum distances to prevent unrealistic tight levels
            min_sl_distance = entry_price * MIN_STOP_LOSS_PCT  # 0.8% minimum
            min_tp_distance = entry_price * MIN_TAKE_PROFIT_PCT  # 1.2% minimum
            
            if risk < min_sl_distance:
                logger.warning(
                    f"⚠️ Stop loss too tight ({(risk/entry_price*100):.2f}%). "
                    f"Widening to minimum {MIN_STOP_LOSS_PCT*100:.1f}%"
                )
                if side == "buy":
                    stop_loss = entry_price - min_sl_distance
                else:
                    stop_loss = entry_price + min_sl_distance
                risk = min_sl_distance
            
            if reward < min_tp_distance:
                logger.warning(
                    f"⚠️ Take profit too tight ({(reward/entry_price*100):.2f}%). "
                    f"Widening to minimum {MIN_TAKE_PROFIT_PCT*100:.1f}%"
                )
                if side == "buy":
                    take_profit = entry_price + min_tp_distance
                else:
                    take_profit = entry_price - min_tp_distance
                reward = min_tp_distance

            risk_reward_ratio = reward / risk

            # NO LONGER FORCE-FITTING R:R - Let natural market structure determine it
            # The trade will be rejected later if R:R < 2.0 (validate_natural_risk_reward)
            # This ensures we only take trades with NATURALLY good risk/reward profiles
            
            # Log warning but DON'T adjust - let validation handle rejection
            if risk_reward_ratio < MIN_RISK_REWARD_RATIO:
                logger.warning(
                    f"⚠️ Natural R:R is {risk_reward_ratio:.2f}:1 (below minimum {MIN_RISK_REWARD_RATIO}:1). "
                    f"This trade will likely be rejected - market structure doesn't support good R:R."
                )

            # Cap at maximum R:R (1:4) - beyond this is often unrealistic
            if risk_reward_ratio > MAX_RISK_REWARD_RATIO:
                logger.info(
                    f"📊 R:R very high ({risk_reward_ratio:.2f}), capping at {MAX_RISK_REWARD_RATIO}"
                )
                if side == "buy":
                    take_profit = entry_price + (risk * MAX_RISK_REWARD_RATIO)
                else:
                    take_profit = entry_price - (risk * MAX_RISK_REWARD_RATIO)

                risk_reward_ratio = MAX_RISK_REWARD_RATIO
                reward = risk * risk_reward_ratio

            risk_pct = (risk / entry_price) * 100
            reward_pct = (reward / entry_price) * 100

            logger.info(
                f"🎯 Adaptive R:R - {side.upper()}\n"
                f"   Entry: ${entry_price:.2f}\n"
                f"   Stop Loss: ${stop_loss:.2f} (-{risk_pct:.2f}%)\n"
                f"   Take Profit: ${take_profit:.2f} (+{reward_pct:.2f}%)\n"
                f"   Risk:Reward = 1:{risk_reward_ratio:.2f}"
            )

            return {
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_reward_ratio": risk_reward_ratio,
                "risk_amount": risk,
                "reward_amount": reward,
                "risk_percentage": risk_pct,
                "reward_percentage": reward_pct,
            }

        except Exception as e:
            logger.error(f"❌ Error calculating adaptive R:R: {e}")
            return self._default_exit_levels(side, entry_price)

    def _default_exit_levels(self, side: str, entry_price: float) -> Dict:
        """
        Default exit levels when adaptive calculation fails.
        Uses wider, more realistic percentage-based levels.
        """
        # Use percentage-based levels for more predictable results
        # These are realistic for Bitcoin day trading
        stop_loss_pct = 0.012  # 1.2% stop loss (realistic for BTC volatility)
        take_profit_pct = 0.024  # 2.4% take profit (1:2 R:R, achievable intraday)
        
        if side == "buy":
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)
        else:
            stop_loss = entry_price * (1 + stop_loss_pct)
            take_profit = entry_price * (1 - take_profit_pct)

        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)

        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": 2.0,
            "risk_amount": risk,
            "reward_amount": reward,
            "risk_percentage": stop_loss_pct * 100,
            "reward_percentage": take_profit_pct * 100,
        }
