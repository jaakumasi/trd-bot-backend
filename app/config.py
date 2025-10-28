import os
from typing import List, Union
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
import json

class Settings(BaseSettings):
    # Database
    database_url: str = Field(default="postgresql://user:pass@localhost/crypto_bot")
    
    # Binance API - Mainnet Keys
    binance_api_key: str = Field(default="")
    binance_secret_key: str = Field(default="")
    
    # Binance API - Testnet Keys
    testnet_binance_api_key: str = Field(default="")
    testnet_binance_secret_key: str = Field(default="")
    
    # Trading Mode Configuration
    binance_testnet: bool = Field(default=True)
    use_paper_trading: bool = Field(default=False)
    
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
    regime_filter_mode: str = Field(default="day_trading")  # day_trading mode for aggressive intraday 
    
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
    
    def get_active_binance_keys(self) -> tuple[str, str]:
        """
        Returns the appropriate Binance API key pair based on BINANCE_TESTNET setting.
        
        Returns:
            tuple[str, str]: (api_key, secret_key)
        """
        if self.binance_testnet:
            return (self.testnet_binance_api_key, self.testnet_binance_secret_key)
        return (self.binance_api_key, self.binance_secret_key)

settings = Settings()
