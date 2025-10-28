"""
Paper Trading Service - Mainnet Paper Trading
Uses real mainnet data to simulate trades without actual execution.
Tracks positions in-memory and records P&L to database.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from ..logging_config import get_trading_metrics_logger

logger = logging.getLogger(__name__)
metrics_logger = get_trading_metrics_logger()


class PaperPosition:
    """Represents a paper trading position"""
    
    def __init__(
        self,
        user_id: int,
        trade_id: str,
        symbol: str,
        side: str,
        amount: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        entry_value: float,
        fees_paid: float = 0.0,
    ):
        self.user_id = user_id
        self.trade_id = trade_id
        self.symbol = symbol
        self.side = side
        self.amount = amount
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.entry_value = entry_value
        self.fees_paid = fees_paid
        self.opened_at = datetime.now(timezone.utc)
        self.status = "open"
        
    def calculate_pnl(self, current_price: float) -> Tuple[float, float]:
        """
        Calculate current P&L for the position.
        
        Returns:
            Tuple[float, float]: (absolute_pnl, percentage_pnl)
        """
        if self.side == "buy":
            pnl = (current_price - self.entry_price) * self.amount
        else:  # sell/short
            pnl = (self.entry_price - current_price) * self.amount
        
        pnl -= self.fees_paid  # Subtract entry fees
        pnl_percentage = (pnl / self.entry_value) * 100
        
        return pnl, pnl_percentage
    
    def should_close_at_sl(self, current_price: float) -> bool:
        """Check if position should close at stop loss"""
        if self.side == "buy":
            return current_price <= self.stop_loss
        else:
            return current_price >= self.stop_loss
    
    def should_close_at_tp(self, current_price: float) -> bool:
        """Check if position should close at take profit"""
        if self.side == "buy":
            return current_price >= self.take_profit
        else:
            return current_price <= self.take_profit
    
    def to_dict(self) -> Dict:
        """Convert position to dictionary for logging"""
        return {
            "user_id": self.user_id,
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side,
            "amount": self.amount,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "entry_value": self.entry_value,
            "fees_paid": self.fees_paid,
            "opened_at": self.opened_at.isoformat(),
            "status": self.status,
        }


class PaperTradingService:
    """
    Manages paper trading state with mainnet data.
    
    Features:
    - In-memory position tracking (TP/SL levels)
    - Realistic fee simulation (0.1% per trade)
    - Trade P&L tracking in database
    - No fake balance tracking - focus on trade outcomes
    """
    
    TRADING_FEE_RATE = 0.001  # 0.1% per trade (Binance standard)
    
    def __init__(self):
        self.user_positions: Dict[int, List[PaperPosition]] = {}  # user_id -> positions
        self.position_monitoring_task: Optional[asyncio.Task] = None
        
        logger.info("📄 Paper Trading Service initialized (mainnet data, simulated trades)")
    
    def get_user_positions(self, user_id: int) -> List[PaperPosition]:
        """Get all open positions for a user"""
        return self.user_positions.get(user_id, [])
    
    def has_open_position(self, user_id: int, symbol: str) -> bool:
        """Check if user has an open position for the symbol"""
        positions = self.get_user_positions(user_id)
        return any(pos.symbol == symbol and pos.status == "open" for pos in positions)
    
    def open_position(
        self,
        user_id: int,
        trade_id: str,
        symbol: str,
        side: str,
        amount: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> Tuple[bool, str, Optional[PaperPosition]]:
        """
        Open a paper trading position.
        
        Returns:
            Tuple[bool, str, Optional[PaperPosition]]: (success, message, position)
        """
        try:
            entry_value = amount * entry_price
            entry_fee = entry_value * self.TRADING_FEE_RATE
            
            position = PaperPosition(
                user_id=user_id,
                trade_id=trade_id,
                symbol=symbol,
                side=side,
                amount=amount,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_value=entry_value,
                fees_paid=entry_fee,
            )
            
            if user_id not in self.user_positions:
                self.user_positions[user_id] = []
            self.user_positions[user_id].append(position)
            
            logger.info(
                f"📄 [PAPER] Opened {side.upper()} position | User: {user_id} | "
                f"Symbol: {symbol} | Amount: {amount} | Entry: ${entry_price:.2f} | "
                f"SL: ${stop_loss:.2f} | TP: ${take_profit:.2f} | Entry Fee: ${entry_fee:.4f}"
            )
            
            metrics_logger.info(
                f"PAPER_POSITION_OPEN | USER={user_id} | SYMBOL={symbol} | "
                f"SIDE={side} | ENTRY={entry_price:.2f} | VALUE=${entry_value:.2f}"
            )
            
            return True, "Position opened successfully", position
            
        except Exception as e:
            logger.error(f"📄 [PAPER] Error opening position: {e}")
            return False, f"Error: {str(e)}", None
    
    def close_position(
        self,
        user_id: int,
        trade_id: str,
        exit_price: float,
        exit_reason: str = "MANUAL",
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        Close a paper trading position.
        
        Returns:
            Tuple[bool, str, Optional[Dict]]: (success, message, trade_summary)
        """
        try:
            positions = self.get_user_positions(user_id)
            position = next((p for p in positions if p.trade_id == trade_id and p.status == "open"), None)
            
            if not position:
                return False, f"Position {trade_id} not found or already closed", None
            
            exit_value = position.amount * exit_price
            exit_fee = exit_value * self.TRADING_FEE_RATE
            
            if position.side == "buy":
                gross_pnl = (exit_price - position.entry_price) * position.amount
            else:
                gross_pnl = (position.entry_price - exit_price) * position.amount
            
            net_pnl = gross_pnl - position.fees_paid - exit_fee
            pnl_percentage = (net_pnl / position.entry_value) * 100
            
            position.status = "closed"
            
            duration = (datetime.now(timezone.utc) - position.opened_at).total_seconds()
            
            trade_summary = {
                "trade_id": trade_id,
                "user_id": user_id,
                "symbol": position.symbol,
                "side": position.side,
                "amount": position.amount,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
                "entry_value": position.entry_value,
                "exit_value": exit_value,
                "total_fees": position.fees_paid + exit_fee,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "pnl_percentage": pnl_percentage,
                "exit_reason": exit_reason,
                "duration_seconds": int(duration),
                "opened_at": position.opened_at.isoformat(),
                "closed_at": datetime.now(timezone.utc).isoformat(),
            }
            
            logger.info(
                f"📄 [PAPER] Closed {position.side.upper()} position | User: {user_id} | "
                f"Symbol: {position.symbol} | Entry: ${position.entry_price:.2f} | "
                f"Exit: ${exit_price:.2f} | P&L: ${net_pnl:.2f} ({pnl_percentage:+.2f}%) | "
                f"Reason: {exit_reason}"
            )
            
            metrics_logger.info(
                f"PAPER_POSITION_CLOSE | USER={user_id} | SYMBOL={position.symbol} | "
                f"ENTRY={position.entry_price:.2f} | EXIT={exit_price:.2f} | "
                f"PNL={net_pnl:.2f} | PNL_PCT={pnl_percentage:.2f} | REASON={exit_reason}"
            )
            
            return True, "Position closed successfully", trade_summary
            
        except Exception as e:
            logger.error(f"📄 [PAPER] Error closing position: {e}")
            return False, f"Error: {str(e)}", None
    
    def check_position_exits(self, user_id: int, current_price: float, symbol: str) -> List[Dict]:
        """
        Check if any positions for a symbol should be closed based on current price.
        
        Args:
            user_id: User ID
            current_price: Current market price
            symbol: Trading symbol
            
        Returns:
            List of positions that hit TP/SL with exit info
        """
        exits = []
        positions = self.get_user_positions(user_id)
        
        for position in positions:
            if position.status != "open" or position.symbol != symbol:
                continue
            
            exit_info = None
            
            if position.should_close_at_sl(current_price):
                exit_info = {
                    "position": position,
                    "exit_price": position.stop_loss,
                    "exit_reason": "STOP_LOSS"
                }
            elif position.should_close_at_tp(current_price):
                exit_info = {
                    "position": position,
                    "exit_price": position.take_profit,
                    "exit_reason": "TAKE_PROFIT"
                }
            
            if exit_info:
                exits.append(exit_info)
        
        return exits
