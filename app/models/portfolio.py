from sqlalchemy import Column, Integer, String, DECIMAL, Boolean, ForeignKey, DateTime, func
from sqlalchemy.orm import relationship
from ..database import Base

class Portfolio(Base):
    __tablename__ = "portfolio"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    asset = Column(String(20), nullable=False)
    balance = Column(DECIMAL(20, 8), nullable=False, default=0)
    locked_balance = Column(DECIMAL(20, 8), default=0)
    average_buy_price = Column(DECIMAL(20, 8), default=0)
    is_test_balance = Column(Boolean, default=True)
    
    # Performance tracking fields
    total_realized_pnl = Column(DECIMAL(20, 8), default=0)  # Cumulative P&L from closed trades
    total_trades = Column(Integer, default=0)  # Total number of trades executed
    winning_trades = Column(Integer, default=0)  # Number of profitable trades
    losing_trades = Column(Integer, default=0)  # Number of losing trades
    win_rate = Column(DECIMAL(5, 2), default=0)  # Percentage of winning trades
    
    updated_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="portfolio")
    