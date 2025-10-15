from sqlalchemy import Column, Integer, String, DateTime, Boolean, DECIMAL, ForeignKey, func
from sqlalchemy.orm import relationship
from ..database import Base

class Trade(Base):
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    trade_id = Column(String(100), unique=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # 'buy' or 'sell'
    amount = Column(DECIMAL(20, 8), nullable=False)
    price = Column(DECIMAL(20, 8), nullable=False)  # Entry price
    total_value = Column(DECIMAL(20, 8), nullable=False)
    fee = Column(DECIMAL(20, 8), default=0)  # Entry fee
    status = Column(String(20), default='pending')  # 'pending', 'filled', 'closed'
    is_test_trade = Column(Boolean, default=True)
    strategy_used = Column(String(50))
    ai_signal_confidence = Column(DECIMAL(5, 2))
    executed_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # OCO (One-Cancels-Other) order tracking
    oco_order_id = Column(String(100))  # Binance OCO order list ID
    
    # Exit tracking fields
    closed_at = Column(DateTime(timezone=True))
    exit_price = Column(DECIMAL(20, 8))
    exit_fee = Column(DECIMAL(20, 8), default=0)
    exit_reason = Column(String(50))  # 'TAKE_PROFIT', 'STOP_LOSS', 'MANUAL', 'TIMEOUT', 'QUICK_PROFIT', 'BREAKEVEN_TIMEOUT', 'TIME_STOP_LOSS'
    
    # P&L tracking fields
    profit_loss = Column(DECIMAL(20, 8))  # Net P&L after all fees
    profit_loss_percentage = Column(DECIMAL(10, 4))  # P&L as percentage of entry value
    duration_seconds = Column(Integer)  # How long the trade was open
    
    # Relationships
    user = relationship("User", back_populates="trades")

class TradingConfig(Base):
    __tablename__ = "trading_configs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    strategy_name = Column(String(50), nullable=False, default='scalping')
    risk_percentage = Column(DECIMAL(5, 2), default=1.0)
    trading_pair = Column(String(20), default='BTCUSDT')
    is_active = Column(Boolean, default=False)
    is_test_mode = Column(Boolean, default=True)
    max_daily_trades = Column(Integer, default=10)
    stop_loss_percentage = Column(DECIMAL(5, 2), default=0.5)
    take_profit_percentage = Column(DECIMAL(5, 2), default=0.3)
    updated_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="trading_configs")


class OpenPosition(Base):
    __tablename__ = "open_positions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    trade_id = Column(String(100), unique=True, index=True, nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # 'buy' or 'sell'
    amount = Column(DECIMAL(20, 8), nullable=False)
    entry_price = Column(DECIMAL(20, 8), nullable=False)
    stop_loss = Column(DECIMAL(20, 8), nullable=True)
    take_profit = Column(DECIMAL(20, 8), nullable=True)
    entry_value = Column(DECIMAL(20, 8), nullable=False)
    fees_paid = Column(DECIMAL(20, 8), default=0)
    is_test_trade = Column(Boolean, default=True)
    opened_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # OCO (One-Cancels-Other) order tracking
    oco_order_id = Column(String(100))  # Binance OCO order list ID for automatic TP/SL
    
    # Relationships
    user = relationship("User")
    