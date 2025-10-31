import os
from typing import List, Union
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from enum import Enum
import json

class TradingMode(str, Enum):
    """
    Trading mode configuration - single source of truth for environment settings.
    """
    MOCK = "mock"                    # Mock service, no real API calls, lenient thresholds
    TESTNET = "testnet"              # Binance testnet API, fake money, lenient thresholds
    PAPER_MAINNET = "paper_mainnet"  # Mainnet API, simulated trades, strict thresholds
    LIVE_MAINNET = "live_mainnet"    # Live mainnet trading with REAL MONEY, strict thresholds

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
    trading_mode: TradingMode = Field(default=TradingMode.TESTNET)
    
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
    regime_filter_mode: str = Field(default="day_trading")  # day_trading mode for aggressive intraday
    
    # Trading Profile Configuration
    trading_profile: str = Field(default="normal")  # strict/normal/permissive - affects thresholds
    ai_override_enabled: bool = Field(default=True)  # Allow AI to override quality filters
    min_confluence_threshold: int = Field(default=60)  # Minimum confluence score for trades (0-100)
    fee_estimate_pct: float = Field(default=0.001)  # Fee estimate for P&L calculations
    
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
        Returns the appropriate Binance API key pair based on trading mode.
        
        Returns:
            tuple[str, str]: (api_key, secret_key)
        """
        if self.trading_mode in (TradingMode.MOCK, TradingMode.TESTNET):
            return (self.testnet_binance_api_key, self.testnet_binance_secret_key)
        else:  # PAPER_MAINNET or LIVE_MAINNET
            return (self.binance_api_key, self.binance_secret_key)
    
    def use_mock_service(self) -> bool:
        """Check if we should use mock Binance service instead of real API."""
        return self.trading_mode == TradingMode.MOCK
    
    def use_paper_execution(self) -> bool:
        """Check if trades should be simulated (not executed on exchange)."""
        return self.trading_mode in (TradingMode.MOCK, TradingMode.TESTNET, TradingMode.PAPER_MAINNET)
    
    def use_mainnet_thresholds(self) -> bool:
        """Check if we should use strict mainnet quality thresholds."""
        return self.trading_mode in (TradingMode.PAPER_MAINNET, TradingMode.LIVE_MAINNET)
    
    def is_mainnet_live(self) -> bool:
        """
        Check if we're running on live mainnet.
        
        Returns:
            bool: True if live mainnet trading is active
        """
        return self.trading_mode == TradingMode.LIVE_MAINNET
    
    def get_environment_name(self) -> str:
        """Get human-readable environment name for logging."""
        env_map = {
            TradingMode.MOCK: "MOCK",
            TradingMode.TESTNET: "TESTNET",
            TradingMode.PAPER_MAINNET: "PAPER_MAINNET",
            TradingMode.LIVE_MAINNET: "LIVE_MAINNET"
        }
        return env_map.get(self.trading_mode, "UNKNOWN")
    
    def get_confluence_threshold(self) -> int:
        """
        Get appropriate confluence threshold based on trading profile and environment.
        Mainnet uses stricter thresholds for safety.
        
        Returns:
            int: Confluence threshold (0-100)
        """
        base_threshold = self.min_confluence_threshold
        
        # Apply profile adjustments
        if self.trading_profile == "strict":
            base_threshold += 10
        elif self.trading_profile == "permissive":
            base_threshold -= 5
        
        # Mainnet safety: increase threshold
        if self.use_mainnet_thresholds() and self.is_mainnet_live():
            base_threshold = max(base_threshold, 65)
        
        return max(50, min(100, base_threshold))

settings = Settings()
