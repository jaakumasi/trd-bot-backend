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
    updated_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="portfolio")
    