import logging
from typing import Dict, Tuple, Optional
from decimal import Decimal, ROUND_DOWN
import asyncio

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self):
        self.max_daily_trades = {}
        self.daily_trade_count = {}
        self.open_positions = {}

    def calculate_position_size(
        self,
        account_balance: float,
        risk_percentage: float,
        entry_price: float,
        stop_loss_price: float,
    ) -> float:
        """Calculate safe position size based on risk management rules"""
        try:
            # Calculate risk per trade in USD
            risk_amount = account_balance * (risk_percentage / 100)

            # Calculate price difference (risk per unit)
            price_risk = abs(entry_price - stop_loss_price)

            if price_risk == 0:
                return 0.0

            # Calculate position size
            position_size = risk_amount / price_risk

            # Ensure we don't risk more than intended
            max_position_value = account_balance * 0.1  # Max 10% of balance per trade
            if position_size * entry_price > max_position_value:
                position_size = max_position_value / entry_price

            # Round down to avoid over-allocation
            return float(
                Decimal(str(position_size)).quantize(
                    Decimal("0.00000001"), rounding=ROUND_DOWN
                )
            )

        except Exception as e:
            logger.error(f"Position size calculation error: {e}")
            return 0.0

    def validate_trade_signal(
        self, user_id: int, signal: Dict, account_balance: float, config: Dict
    ) -> Tuple[bool, str, Dict]:
        """Validate if a trade signal meets risk management criteria"""
        try:
            # Check if trading is active
            if not config.get("is_active", False):
                return False, "Trading bot is not active", {}

            # Check daily trade limit
            today_count = self.daily_trade_count.get(user_id, 0)
            max_trades = config.get("max_daily_trades", 10)

            if today_count >= max_trades:
                return False, f"Daily trade limit reached ({max_trades})", {}

            # Check minimum confidence score
            confidence = signal.get("final_confidence", 0)
            if confidence < 60:  # Minimum confidence threshold
                return False, f"Signal confidence too low: {confidence}%", {}

            # Check account balance
            if account_balance < 10:  # Minimum $10 to trade
                return False, "Insufficient account balance for trading", {}

            # Calculate position details
            entry_price = signal.get("entry_price", 0)
            stop_loss = signal.get("stop_loss", 0)

            if not entry_price or not stop_loss:
                return False, "Missing entry price or stop loss in signal", {}

            position_size = self.calculate_position_size(
                account_balance,
                config.get("risk_percentage", 1.0),
                entry_price,
                stop_loss,
            )

            if position_size == 0:
                return False, "Calculated position size is zero", {}

            # Calculate trade value
            trade_value = position_size * entry_price

            if trade_value < 10:  # Minimum trade value $10
                return False, f"Trade value too small: ${trade_value:.2f}", {}

            if trade_value > account_balance * 0.1:  # Max 10% per trade
                return False, f"Trade value exceeds 10% of balance", {}

            # All checks passed
            trade_params = {
                "position_size": position_size,
                "trade_value": trade_value,
                "risk_amount": account_balance
                * (config.get("risk_percentage", 1.0) / 100),
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": signal.get(
                    "take_profit", entry_price * 1.003
                ),  # Default 0.3% profit
            }

            return True, "Trade approved", trade_params

        except Exception as e:
            logger.error(f"Trade validation error: {e}")
            return False, f"Validation error: {str(e)}", {}

    def record_trade(self, user_id: int):
        """Record a trade for daily limit tracking"""
        self.daily_trade_count[user_id] = self.daily_trade_count.get(user_id, 0) + 1

    def reset_daily_counters(self):
        """Reset daily trade counters (called at midnight)"""
        self.daily_trade_count.clear()
        logger.info("Daily trade counters reset")
