import os
from typing import List, Union
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
import json

class Settings(BaseSettings):
    # Database
    database_url: str = Field(default="postgresql://user:pass@localhost/crypto_bot")
    
    # Binance API
    binance_api_key: str = Field(default="")
    binance_secret_key: str = Field(default="")
    binance_testnet: bool = Field(default=True)
    
    # Gemini AI
    gemini_api_key: str = Field(default="")
    
    # Security
    jwt_secret: str = Field(default="")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiration: int = Field(default=3600)
    
    # Trading Configuration - DAY TRADING OPTIMIZED
    default_risk_percentage: float = Field(default=1.0)  # 1% rule
    default_trading_pair: str = Field(default="BTCUSDT")
    trading_active_hours_start: str = Field(default="00:00")  # 24/7 crypto market
    trading_active_hours_end: str = Field(default="23:59")    # Full day coverage
    use_mock_binance: bool = Field(default=False)
    regime_filter_mode: str = Field(default="balanced")  # strict, balanced, permissive, day_trading 
    
    # CORS - accepts both string and list
    cors_origins: Union[str, List[str]] = Field(
        default=["http://localhost:4200", "https://*.vercel.app"]
    )
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379")

    @field_validator('cors_origins', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list"""
        if isinstance(v, str):
            # Try to parse as JSON array first
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
            
            # If not JSON, split by comma
            return [origin.strip() for origin in v.split(',') if origin.strip()]
        
        return v

    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()
