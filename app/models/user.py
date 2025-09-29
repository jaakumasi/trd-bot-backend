from sqlalchemy import Column, Integer, String, DateTime, Boolean, func
from sqlalchemy.orm import relationship
from ..database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)  
    api_key_hash = Column(String(255), nullable=True)    # Optional: for API key storage
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    trading_configs = relationship("TradingConfig", back_populates="user")
    trades = relationship("Trade", back_populates="user")
    portfolio = relationship("Portfolio", back_populates="user")
    