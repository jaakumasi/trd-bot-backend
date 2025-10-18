import logging
from typing import Dict, Tuple, Optional, Union, List
from decimal import Decimal
from datetime import datetime, timezone

from .service_constants import (
    DEFAULT_MAX_DAILY_TRADES,
    DEFAULT_RISK_PERCENTAGE,
    EIGHT_DECIMAL_PLACES,
    MAX_BALANCE_TRADE_RATIO,
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

    def calculate_position_size(
        self,
        account_balance: Union[float, Decimal],
        risk_percentage: Union[float, Decimal],
        entry_price: Union[float, Decimal],
        stop_loss_price: Union[float, Decimal],
    ) -> float:
        """
        Calculate safe position size for SPOT TRADING (no leverage).

        FIXED FORMULA for spot-only trading:
        - Ensures position never exceeds available capital
        - Applies both risk-based and capital-based constraints
        - Adds absolute safety caps for scalping

        Args:
            account_balance: Total account balance in USDT
            risk_percentage: Max % of balance to risk (e.g., 1.0 = 1%)
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
                "Position calc inputs: balance=$%.2f, risk=%s%%, entry=$%.2f, stop=$%.2f",
                balance,
                risk,
                entry,
                stop,
            )

            price_risk = abs(entry - stop)
            if price_risk == 0:
                logger.warning("Price risk is zero - entry price equals stop loss price")
                return 0.0

            # Method 1: Risk-based sizing (how much to buy based on acceptable loss)
            risk_amount = balance * (risk / 100)
            risk_based_position = risk_amount / price_risk

            # Method 2: Capital-based cap with VOLATILITY ADJUSTMENT
            # SAFETY FEATURE: Reduce position size in high-volatility periods
            #
            # Base position sizes:
            #   - 0.03 (3%) = Very conservative, ~$1.50 loss on $10K if SL hits
            #   - 0.05 (5%) = Conservative (DEFAULT), ~$2.50 loss on $10K if SL hits
            #   - 0.08 (8%) = Moderate, ~$4.00 loss on $10K if SL hits
            #   - 0.10 (10%) = Aggressive, ~$5.00 loss on $10K if SL hits

            # Check if we have volatility data available (set by get_adaptive_exit_levels)
            if self.current_volatility_percentile is not None and self.current_volatility_percentile > 75:
                # HIGH VOLATILITY DETECTED (>75th percentile)
                # Reduce position size to 3% for safety
                max_position_pct = 0.03
                logger.info(
                    f"🔻 Position size REDUCED to 3% (HIGH volatility: "
                    f"{self.current_volatility_percentile:.0f}th percentile) - SAFETY OVERRIDE"
                )
            else:
                # NORMAL or LOW volatility: Use standard 5%
                max_position_pct = 0.05
                if self.current_volatility_percentile is not None:
                    logger.debug(
                        f"📊 Position size: 5% (volatility: {self.current_volatility_percentile:.0f}th percentile)"
                    )

            capital_based_position = (balance * max_position_pct) / entry

            # Method 3: Absolute position value cap (for safety)
            # Never risk more than 10% of balance in a single position
            absolute_max_position_value = balance * MAX_BALANCE_TRADE_RATIO  # 10%
            absolute_position_limit = absolute_max_position_value / entry

            # Take the MINIMUM of all three methods (most conservative)
            position_size = min(
                risk_based_position,
                capital_based_position,
                absolute_position_limit
            )

            logger.debug(
                "Position sizing breakdown:\n"
                "  Risk-based: %.8f (risk $%.2f / $%.2f price movement)\n"
                "  Capital-based: %.8f (5%% of balance)\n"
                "  Absolute limit: %.8f (10%% max position)\n"
                "  FINAL: %.8f",
                risk_based_position,
                risk_amount,
                price_risk,
                capital_based_position,
                absolute_position_limit,
                position_size
            )

            # Additional safety: Check stop-loss percentage is reasonable for scalping
            stop_loss_pct = (price_risk / entry) * 100
            if stop_loss_pct > 0.5:  # More than 0.5% stop for scalping is aggressive
                logger.warning(
                    f"⚠️  Wide stop-loss detected: {stop_loss_pct:.2f}% "
                    f"(>0.5% not ideal for scalping)"
                )

            # Truncate to exchange precision
            position_size = self._truncate(position_size)

            position_value = position_size * entry
            logger.info(
                "✅ Calculated position size: %.8f (value: $%.2f, %.1f%% of balance)",
                position_size,
                position_value,
                (position_value / balance * 100) if balance > 0 else 0
            )

            # Final validation: position should never exceed balance
            if position_value > balance:
                logger.error(
                    f"❌ CRITICAL: Position value (${position_value:.2f}) exceeds balance (${balance:.2f})! "
                    f"This should never happen with corrected formula."
                )
                return 0.0

            return position_size

        except Exception as e:
            logger.error(f"Position size calculation error: {e}")
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

    def validate_trade_signal(
        self, user_id: int, signal: Dict, account_balance: float, config: Dict
    ) -> Tuple[bool, str, Dict]:
        """Validate if a trade signal meets risk management criteria"""
        try:
            self._assert_trading_active(config)

            existing_positions = self.get_open_positions(user_id)
            self._assert_no_open_positions(existing_positions)

            today_count = self.daily_trade_count.get(user_id, 0)
            self._assert_daily_limit(user_id, today_count, config)

            confidence = self._extract_confidence(signal)
            self._assert_confidence(confidence)

            balance = self._to_float(account_balance)
            self._assert_sufficient_balance(balance)

            entry_price, stop_loss = self._extract_prices(signal)
            risk_percentage = self._resolve_risk_percentage(config)

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
                "take_profit": self._resolve_take_profit(signal, entry_price),
            }

            return True, "Trade approved", trade_params

        except RiskValidationError as err:
            return False, err.message, {}
        except Exception as e:
            logger.error(f"Trade validation error: {e}")
            return False, f"Validation error: {str(e)}", {}

    def record_trade(self, user_id: int):
        """Record a trade for daily limit tracking"""
        self.daily_trade_count[user_id] = self.daily_trade_count.get(user_id, 0) + 1

    def add_open_position(self, user_id: int, position_data: Dict):
        """Add a new open position for tracking"""
        if user_id not in self.open_positions:
            self.open_positions[user_id] = []
        
        position = {
            'trade_id': position_data['trade_id'],
            'symbol': position_data['symbol'],
            'side': position_data['side'],
            'amount': position_data['amount'],
            'entry_price': position_data['entry_price'],
            'stop_loss': position_data['stop_loss'],
            'take_profit': position_data['take_profit'],
            'entry_time': position_data['entry_time'],
            'entry_value': position_data['entry_value'],
            'fees_paid': position_data['fees_paid']
        }
        
        self.open_positions[user_id].append(position)
        logger.info(f"📋 [User {user_id}] Position added: {position['side']} {position['amount']:.6f} {position['symbol']} @ ${position['entry_price']:.4f}")

    def get_open_positions(self, user_id: int) -> list:
        """Get all open positions for a user"""
        return self.open_positions.get(user_id, [])

    def check_exit_conditions(self, user_id: int, current_price: float, symbol: str) -> list:
        """Check if any positions should be closed based on current price"""
        positions_to_close = []
        
        if user_id not in self.open_positions:
            return positions_to_close

        for position in self.open_positions[user_id]:
            if position['symbol'] != symbol:
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
        self, 
        db, 
        user_id: int, 
        df, 
        side: str,
        entry_price: float
    ) -> Dict:
        """
        Calculate optimal SL/TP based on:
        1. Current market volatility (ATR) - inversely scaled
        2. Historical user performance in similar volatility
        3. Safety caps to prevent excessive risk
        
        Philosophy:
        - High volatility → Wider stops (larger ATR multipliers) to avoid premature exits
        - Low volatility → Tighter stops (smaller ATR multipliers) for precision
        
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
            atr = df['atr'].iloc[-1]
            current_price = df['close'].iloc[-1]
            atr_percentage = (atr / current_price) * 100
            
            # Calculate ATR percentile to determine volatility regime
            atr_history = df['atr'].tail(100) if len(df) >= 100 else df['atr']
            from scipy.stats import percentileofscore
            volatility_percentile = percentileofscore(atr_history, atr)

            # Store volatility percentile for use by position sizing
            self.current_volatility_percentile = volatility_percentile
            
            # Adaptive Multipliers based on volatility regime
            # High volatility (>75th percentile) → Wider stops (2.5-3x ATR)
            # Medium volatility (25th-75th) → Standard stops (1.8-2.2x ATR)
            # Low volatility (<25th percentile) → Tighter stops (1.2-1.5x ATR)
            
            if volatility_percentile > 75:
                # HIGH VOLATILITY: Use wider stops to avoid noise
                sl_multiplier = 2.8
                tp_multiplier = 3.5
                volatility_regime = "HIGH"
            elif volatility_percentile > 50:
                # MEDIUM-HIGH VOLATILITY
                sl_multiplier = 2.2
                tp_multiplier = 2.8
                volatility_regime = "MEDIUM_HIGH"
            elif volatility_percentile > 25:
                # MEDIUM VOLATILITY
                sl_multiplier = 1.8
                tp_multiplier = 2.3
                volatility_regime = "MEDIUM"
            else:
                # LOW VOLATILITY: Use tighter stops for precision
                sl_multiplier = 1.3
                tp_multiplier = 1.7
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
                
                query = select(Trade).where(
                    Trade.user_id == user_id,
                    Trade.status == "closed"
                ).order_by(Trade.closed_at.desc()).limit(50)
                
                result = await db.execute(query)
                trades_list = result.scalars().all()
                
                if len(trades_list) >= 10:
                    # Calculate average actual exit percentage for wins/losses
                    winning_exits = [
                        abs(float(t.profit_loss_percentage)) 
                        for t in trades_list 
                        if t.profit_loss and float(t.profit_loss) > 0 and t.profit_loss_percentage
                    ]
                    losing_exits = [
                        abs(float(t.profit_loss_percentage))
                        for t in trades_list
                        if t.profit_loss and float(t.profit_loss) < 0 and t.profit_loss_percentage
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
                logger.warning(f"⚠️ Could not fetch historical performance for adaptive exits: {hist_error}")
                # Continue with ATR-based values
            
            # 3. Volatility-Adjusted Safety Caps
            # TIGHTENED for scalping - prioritize capital preservation
            # High volatility periods need wider caps BUT with absolute maximum

            # ABSOLUTE MAXIMUM CAPS (regardless of volatility)
            ABSOLUTE_MAX_STOP_LOSS = 0.5  # Never risk more than 0.5% per trade
            ABSOLUTE_MAX_TAKE_PROFIT = 1.5  # Never target more than 1.5% profit

            if volatility_regime == "HIGH":
                # High volatility: wider stops but capped
                max_sl = 0.5  # Reduced from 1.5% - too risky for scalping
                min_sl = 0.3
                max_tp = 1.2  # Reduced from 2.5%
                min_tp = 0.6
            elif volatility_regime in ["MEDIUM_HIGH", "MEDIUM"]:
                max_sl = 0.4  # Reduced from 1.0%
                min_sl = 0.25
                max_tp = 0.8  # Reduced from 1.8%
                min_tp = 0.4
            else:  # LOW volatility
                max_sl = 0.35  # Reduced from 0.7%
                min_sl = 0.2
                max_tp = 0.6  # Reduced from 1.0%
                min_tp = 0.3

            # Apply volatility-specific caps
            dynamic_sl_pct = min(dynamic_sl_pct, max_sl)
            dynamic_sl_pct = max(dynamic_sl_pct, min_sl)
            dynamic_tp_pct = min(dynamic_tp_pct, max_tp)
            dynamic_tp_pct = max(dynamic_tp_pct, min_tp)

            # Apply absolute maximum caps (safety override)
            dynamic_sl_pct = min(dynamic_sl_pct, ABSOLUTE_MAX_STOP_LOSS)
            dynamic_tp_pct = min(dynamic_tp_pct, ABSOLUTE_MAX_TAKE_PROFIT)

            # Ensure minimum risk:reward ratio of 1.2:1 for profitability
            min_risk_reward = 1.2
            if dynamic_tp_pct / dynamic_sl_pct < min_risk_reward:
                logger.warning(
                    f"⚠️  Risk:reward ratio too low ({dynamic_tp_pct/dynamic_sl_pct:.2f}:1). "
                    f"Adjusting TP to maintain {min_risk_reward}:1"
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
                "tp_multiplier": tp_multiplier
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
                    "risk_reward_ratio": 0.6
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
                    "risk_reward_ratio": 0.6
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
    def _assert_confidence(confidence: float) -> None:
        if confidence < MIN_SIGNAL_CONFIDENCE:
            raise RiskValidationError(f"Signal confidence too low: {confidence}%")

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
        if not positions:
            return
        position_details = positions[0]
        raise RiskValidationError(
            f"User already has {len(positions)} open position(s) for {position_details['symbol']}. Only one trade allowed at a time."
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


