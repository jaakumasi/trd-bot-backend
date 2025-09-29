import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database
    database_url: str = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/crypto_bot")
    
    # Binance API
    binance_api_key: str = os.getenv("BINANCE_API_KEY", "")
    binance_secret_key: str = os.getenv("BINANCE_SECRET_KEY", "")
    binance_testnet: bool = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
    
    # Gemini AI
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    
    # Security
    jwt_secret: str = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600
    
    # Trading Configuration
    default_risk_percentage: float = 1.0
    default_trading_pair: str = "BTCUSDT"
    trading_active_hours_start: str = os.getenv("TRADING_ACTIVE_HOURS_START", "08:00")
    trading_active_hours_end: str = os.getenv("TRADING_ACTIVE_HOURS_END", "16:00")
    
    # CORS
    cors_origins: List[str] = [
        "http://localhost:4200",
        "https://*.vercel.app"
    ]
    
    class Config:
        env_file = ".env"

settings = Settings()
