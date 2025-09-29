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
    price = Column(DECIMAL(20, 8), nullable=False)
    total_value = Column(DECIMAL(20, 8), nullable=False)
    fee = Column(DECIMAL(20, 8), default=0)
    status = Column(String(20), default='pending')
    is_test_trade = Column(Boolean, default=True)
    strategy_used = Column(String(50))
    ai_signal_confidence = Column(DECIMAL(5, 2))
    executed_at = Column(DateTime(timezone=True), server_default=func.now())
    closed_at = Column(DateTime(timezone=True))
    
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
    