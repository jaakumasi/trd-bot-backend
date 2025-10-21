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
    get_randomized_threshold,
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

            # SAFETY CAP: Position value should not exceed 3% of balance
            # This prevents over-concentration in a single trade
            position_value = position_size * entry
            max_position_value = balance * MAX_BALANCE_TRADE_RATIO  # 3% from constants

            if position_value > max_position_value:
                logger.warning(
                    f"⚠️ Position value (${position_value:.2f}) exceeds 3%% cap (${max_position_value:.2f})"
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
        
        # If no history, use randomized base threshold
        if not user_trade_history:
            randomized = get_randomized_threshold(base_threshold, variance_pct=0.10)
            # Cap at 70% maximum
            capped = min(randomized, 70.0)
            logger.info(
                f"🎲 No trade history - using randomized threshold: {capped:.1f} "
                f"(base: {base_threshold}, capped at 70%)"
            )
            return capped
        
        win_rate = user_trade_history.get('win_rate', 50.0)
        recent_trend = user_trade_history.get('recent_trend', 'neutral')
        
        # Adaptive adjustment based on performance (but capped at 70%)
        if win_rate > 60 and recent_trend == 'winning':
            # System is hot - be more aggressive
            adjusted = base_threshold - 10
            logger.info(
                f"📈 High performance detected (WR: {win_rate:.1f}%, trend: {recent_trend}) "
                f"- reducing threshold to {adjusted}"
            )
        elif win_rate < 40 or recent_trend == 'losing':
            # System struggling - require higher confidence (but cap at 70%)
            adjusted = min(base_threshold + 5, 70.0)
            logger.info(
                f"📉 Low performance detected (WR: {win_rate:.1f}%, trend: {recent_trend}) "
                f"- increasing threshold to {adjusted} (capped at 70%)"
            )
        else:
            # Normal performance - use base with slight randomization
            adjusted = base_threshold
            logger.info(
                f"➡️ Normal performance (WR: {win_rate:.1f}%) - using base threshold {adjusted}"
            )
        
        # Add randomization to prevent overfitting to exact values
        final_threshold = get_randomized_threshold(adjusted, variance_pct=0.08)
        
        # Cap at 70% maximum to prevent killing good trades
        final_threshold = min(final_threshold, 70.0)
        
        # Clamp to reasonable range (50-70%)
        final_threshold = max(50, min(70, final_threshold))
        
        logger.info(f"✅ Final adaptive threshold: {final_threshold:.1f}% (capped at 70%)")
        
        logger.info(
            f"✅ Adaptive confidence threshold: {final_threshold:.1f} "
            f"(adjusted: {adjusted}, randomized with ±8%)"
        )
        
        return final_threshold

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

            # Use adaptive R:R if market data available, otherwise use AI suggestions
            # CRITICAL: For safety, ALWAYS prefer AI-suggested levels for mean reversion
            use_ai_levels = True  # Default to safer AI levels
            
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

            # Adaptive Multipliers based on volatility regime
            # High volatility (>75th percentile) → Wider stops (2.5-3x ATR)
            # Medium volatility (25th-75th) → Standard stops (1.8-2.2x ATR)
            # Low volatility (<25th percentile) → Tighter stops (1.2-1.5x ATR)
            # **MEAN REVERSION → EXTRA WIDE (3.0-4.0x ATR for breathing room)**

            if is_mean_reversion:
                # Mean reversion needs MUCH wider stops (bounces are volatile)
                sl_multiplier = 3.5
                tp_multiplier = 7.5
                volatility_regime = "MEAN_REVERSION"
                logger.info(
                    f"📏 Using WIDE stops for mean reversion: SL={sl_multiplier}x, TP={tp_multiplier}x ATR"
                )
            elif volatility_percentile > 75:
                # HIGH VOLATILITY: Use wider stops to avoid noise (2:1+ R:R)
                sl_multiplier = 2.8
                tp_multiplier = 6.0
                volatility_regime = "HIGH"
            elif volatility_percentile > 50:
                # MEDIUM-HIGH VOLATILITY (2:1+ R:R)
                sl_multiplier = 2.2
                tp_multiplier = 5.0
                volatility_regime = "MEDIUM_HIGH"
            elif volatility_percentile > 25:
                # MEDIUM VOLATILITY (2:1+ R:R)
                sl_multiplier = 1.8
                tp_multiplier = 4.0
                volatility_regime = "MEDIUM"
            else:
                # LOW VOLATILITY: Use tighter stops for precision (2:1+ R:R)
                sl_multiplier = 1.3
                tp_multiplier = 3.0
                volatility_regime = "LOW"

            # Calculate percentage distances
            dynamic_sl_pct = atr_percentage * sl_multiplier
            dynamic_tp_pct = atr_percentage * tp_multiplier

            # Calculate percentage distances
            dynamic_sl_pct = atr_percentage * sl_multiplier
            dynamic_tp_pct = atr_percentage * tp_multiplier

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

            # 3. Volatility-Adjusted Safety Caps
            # Day trading risk management - adaptive to market structure
            # High volatility periods need wider caps BUT with absolute maximum

            # ABSOLUTE MAXIMUM CAPS (adjusted for 2:1 R:R requirement)
            ABSOLUTE_MAX_STOP_LOSS = 0.5  # Never risk more than 0.5% per trade
            ABSOLUTE_MAX_TAKE_PROFIT = 2.0  # Increased from 1.5% to support 2:1 R:R

            if volatility_regime == "HIGH":
                # High volatility: wider stops but capped (2:1 R:R friendly)
                max_sl = 0.5  
                min_sl = 0.3
                max_tp = 1.5 
                min_tp = 0.8
            elif volatility_regime in ["MEDIUM_HIGH", "MEDIUM"]:
                max_sl = 0.4
                min_sl = 0.25
                max_tp = 1.0
                min_tp = 0.6
            else:  # LOW volatility
                max_sl = 0.35 
                min_sl = 0.2
                max_tp = 0.8
                min_tp = 0.5

            # Apply volatility-specific caps
            dynamic_sl_pct = min(dynamic_sl_pct, max_sl)
            dynamic_sl_pct = max(dynamic_sl_pct, min_sl)
            dynamic_tp_pct = min(dynamic_tp_pct, max_tp)
            dynamic_tp_pct = max(dynamic_tp_pct, min_tp)

            # Apply absolute maximum caps (safety override)
            dynamic_sl_pct = min(dynamic_sl_pct, ABSOLUTE_MAX_STOP_LOSS)
            dynamic_tp_pct = min(dynamic_tp_pct, ABSOLUTE_MAX_TAKE_PROFIT)

            # Ensure minimum risk:reward ratio of 2.0:1 for mainnet profitability
            # This accounts for fees, slippage, and ensures sustainable edge
            min_risk_reward = 2.0
            if dynamic_tp_pct / dynamic_sl_pct < min_risk_reward:
                logger.warning(
                    f"⚠️  Risk:reward ratio too low ({dynamic_tp_pct/dynamic_sl_pct:.2f}:1). "
                    f"Adjusting TP to maintain {min_risk_reward}:1 (mainnet requirement)"
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
            raise RiskValidationError("Trade value exceeds 10% of balance")

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
        try:
            side = side.lower()

            if side == "buy":
                # Stop loss: Just below nearest support
                support = sr_levels["nearest_support"]
                buffer = atr * 0.5  # Half ATR below support for breathing room
                stop_loss = support - buffer

                # Take profit: Just below nearest resistance
                resistance = sr_levels["nearest_resistance"]
                take_profit = resistance - (atr * 0.3)  # Slightly below resistance

            else:  # sell
                # Stop loss: Just above nearest resistance
                resistance = sr_levels["nearest_resistance"]
                buffer = atr * 0.5
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

            # Validate minimum R:R (1:1.5)
            if risk_reward_ratio < MIN_RISK_REWARD_RATIO:
                logger.warning(
                    f"⚠️ R:R too low ({risk_reward_ratio:.2f}), adjusting to minimum {MIN_RISK_REWARD_RATIO}"
                )
                # Extend take profit to meet minimum R:R
                if side == "buy":
                    take_profit = entry_price + (risk * MIN_RISK_REWARD_RATIO)
                else:
                    take_profit = entry_price - (risk * MIN_RISK_REWARD_RATIO)

                risk_reward_ratio = MIN_RISK_REWARD_RATIO

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
