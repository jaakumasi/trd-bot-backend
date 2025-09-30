import logging
from typing import Dict, Tuple, Optional, Union
from decimal import Decimal
from datetime import datetime, timezone
import asyncio

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self):
        self.max_daily_trades = {}
        self.daily_trade_count = {}
        self.open_positions = {}  # {user_id: [position_objects]}
        self.account_balances = {}  # {user_id: balance}

    def calculate_position_size(
        self,
        account_balance: Union[float, Decimal],
        risk_percentage: Union[float, Decimal],
        entry_price: Union[float, Decimal],
        stop_loss_price: Union[float, Decimal],
    ) -> float:
        """Calculate safe position size based on risk management rules"""
        try:
            # Ensure all inputs are floats to avoid type mismatches
            account_balance = float(account_balance)
            risk_percentage = float(risk_percentage)
            entry_price = float(entry_price)
            stop_loss_price = float(stop_loss_price)
            
            logger.debug(f"Position calc inputs: balance=${account_balance:.2f}, risk={risk_percentage}%, entry=${entry_price:.2f}, stop=${stop_loss_price:.2f}")

            # Calculate risk per trade in USD
            risk_amount = account_balance * (risk_percentage / 100)

            # Calculate price difference (risk per unit)
            price_risk = abs(entry_price - stop_loss_price)

            if price_risk == 0:
                logger.warning("Price risk is zero - entry price equals stop loss price")
                return 0.0

            # Calculate position size
            position_size = risk_amount / price_risk
            logger.debug(f"Initial position size: {position_size:.8f}")

            # Ensure we don't risk more than intended
            max_position_value = account_balance * 0.1  # Max 10% of balance per trade
            if position_size * entry_price > max_position_value:
                position_size = max_position_value / entry_price
                logger.debug(f"Position size adjusted for max 10% balance: {position_size:.8f}")

            # Round down to avoid over-allocation using simple rounding instead of Decimal
            position_size = float(int(position_size * 100000000) / 100000000)  # 8 decimal places
            
            logger.info(f"✅ Calculated position size: {position_size:.8f} (${position_size * entry_price:.2f})")
            return position_size

        except Exception as e:
            logger.error(f"Position size calculation error: {e}")
            logger.error(f"Inputs - balance: {account_balance} ({type(account_balance)}), risk: {risk_percentage} ({type(risk_percentage)}), entry: {entry_price} ({type(entry_price)}), stop: {stop_loss_price} ({type(stop_loss_price)})")
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

            # Calculate position details - ensure all values are floats
            entry_price = float(signal.get("entry_price", 0))
            stop_loss = float(signal.get("stop_loss", 0))

            if not entry_price or not stop_loss:
                return False, "Missing entry price or stop loss in signal", {}

            # Convert risk_percentage from Decimal to float
            risk_percentage = float(config.get("risk_percentage", 1.0))
            
            position_size = self.calculate_position_size(
                account_balance,
                risk_percentage,
                entry_price,
                stop_loss,
            )

            if position_size == 0:
                return False, "Calculated position size is zero", {}

            # Calculate trade value - both values are now guaranteed to be floats
            trade_value = position_size * entry_price

            if trade_value < 10:  # Minimum trade value $10
                return False, f"Trade value too small: ${trade_value:.2f}", {}

            if trade_value > account_balance * 0.1:  # Max 10% per trade
                return False, "Trade value exceeds 10% of balance", {}

            # All checks passed
            take_profit = float(signal.get("take_profit", entry_price * 1.003))  # Default 0.3% profit
            
            trade_params = {
                "position_size": position_size,
                "trade_value": trade_value,
                "risk_amount": account_balance * (risk_percentage / 100),
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            }

            return True, "Trade approved", trade_params

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

    def close_position(self, user_id: int, trade_id: str, exit_price: float, exit_reason: str, fees_paid: float) -> Optional[Dict]:
        """Close a position and calculate P&L"""
        if user_id not in self.open_positions:
            return None
            
        for i, position in enumerate(self.open_positions[user_id]):
            if position['trade_id'] == trade_id:
                # Calculate P&L
                if position['side'].lower() == 'buy':
                    # For buy positions: profit when price goes up
                    pnl = (exit_price - position['entry_price']) * position['amount']
                else:
                    # For sell positions: profit when price goes down
                    pnl = (position['entry_price'] - exit_price) * position['amount']
                
                # Subtract all fees
                total_fees = position['fees_paid'] + fees_paid
                net_pnl = pnl - total_fees
                
                # Calculate percentage return
                pnl_percentage = (net_pnl / position['entry_value']) * 100
                
                # Handle timezone-aware and timezone-naive datetime comparison
                current_time = datetime.now(timezone.utc)
                entry_time = position['entry_time']
                
                # If entry_time is timezone-aware, use it as is; otherwise make it UTC
                if entry_time.tzinfo is None:
                    entry_time = entry_time.replace(tzinfo=timezone.utc)
                
                duration_seconds = (current_time - entry_time).total_seconds()
                
                closed_position = {
                    **position,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'exit_time': current_time,
                    'exit_fees': fees_paid,
                    'total_fees': total_fees,
                    'gross_pnl': pnl,
                    'net_pnl': net_pnl,
                    'pnl_percentage': pnl_percentage,
                    'duration_seconds': duration_seconds
                }
                
                # Remove from open positions
                self.open_positions[user_id].pop(i)
                
                # Update account balance
                if user_id not in self.account_balances:
                    self.account_balances[user_id] = 10000.0  # Default test balance
                
                self.account_balances[user_id] += net_pnl
                
                logger.info(f"💰 [User {user_id}] Position CLOSED: {net_pnl:+.2f} USD ({pnl_percentage:+.2f}%) - {exit_reason}")
                
                return closed_position
        
        return None

    def check_exit_conditions(self, user_id: int, current_price: float, symbol: str) -> list:
        """Check if any positions should be closed based on current price"""
        positions_to_close = []
        
        if user_id not in self.open_positions:
            return positions_to_close
            
        for position in self.open_positions[user_id]:
            if position['symbol'] != symbol:
                continue
                
            should_close = False
            exit_reason = ""
            
            if position['side'].lower() == 'buy':
                # For buy positions
                if current_price >= position['take_profit']:
                    should_close = True
                    exit_reason = "TAKE_PROFIT"
                elif current_price <= position['stop_loss']:
                    should_close = True
                    exit_reason = "STOP_LOSS"
            else:
                # For sell positions
                if current_price <= position['take_profit']:
                    should_close = True
                    exit_reason = "TAKE_PROFIT"
                elif current_price >= position['stop_loss']:
                    should_close = True
                    exit_reason = "STOP_LOSS"
            
            if should_close:
                positions_to_close.append({
                    'position': position,
                    'exit_reason': exit_reason,
                    'current_price': current_price
                })
        
        return positions_to_close

    def get_account_balance(self, user_id: int) -> float:
        """Get current account balance for user"""
        return self.account_balances.get(user_id, 10000.0)  # Default test balance
    
    def set_account_balance(self, user_id: int, balance: float):
        """Set account balance for user"""
        self.account_balances[user_id] = balance

    def reset_daily_counters(self):
        """Reset daily trade counters (called at midnight)"""
        self.daily_trade_count.clear()
        logger.info("Daily trade counters reset")
