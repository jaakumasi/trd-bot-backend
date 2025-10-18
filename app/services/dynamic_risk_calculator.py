"""
Dynamic Risk/Reward Calculator
ATR-based intelligent stop-loss and take-profit calculation that maintains 1:2 ratio.
Adapts to market volatility automatically - works identically on testnet and mainnet.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class DynamicRiskCalculator:
    """
    Calculate stop-loss and take-profit levels based on ATR (Average True Range).
    
    Philosophy:
    - Stop-loss should be wide enough to avoid noise but tight enough to limit risk
    - Take-profit should always be 2x the stop distance (1:2 risk/reward ratio)
    - ATR measures real market volatility better than fixed percentages
    
    Industry Standard:
    - Stop-Loss = 1.5 × ATR (captures 85% of normal price fluctuations)
    - Take-Profit = 3.0 × ATR (maintains strict 1:2 ratio)
    """
    
    def __init__(
        self,
        atr_multiplier_stop: float = 1.5,
        atr_multiplier_target: float = 3.0,
        min_risk_pct: float = 0.3,
        max_risk_pct: float = 1.0,
    ):
        """
        Initialize the dynamic risk calculator.
        
        Args:
            atr_multiplier_stop: Multiplier for stop-loss (default 1.5)
            atr_multiplier_target: Multiplier for take-profit (default 3.0 = 1:2 ratio)
            min_risk_pct: Minimum risk percentage (safety floor, default 0.3%)
            max_risk_pct: Maximum risk percentage (safety ceiling, default 1.0%)
        """
        self.atr_multiplier_stop = atr_multiplier_stop
        self.atr_multiplier_target = atr_multiplier_target
        self.min_risk_pct = min_risk_pct
        self.max_risk_pct = max_risk_pct
        
        logger.info("💰 Dynamic Risk Calculator initialized:")
        logger.info(f"   📊 Stop-Loss: {atr_multiplier_stop}× ATR")
        logger.info(f"   🎯 Take-Profit: {atr_multiplier_target}× ATR (1:2 ratio)")
        logger.info(f"   🛡️  Risk Range: {min_risk_pct}% - {max_risk_pct}%")
    
    def calculate_levels(
        self,
        entry_price: float,
        atr: float,
        direction: str,
        current_price: float = None,
    ) -> Dict:
        """
        Calculate dynamic stop-loss and take-profit levels based on ATR.
        
        Args:
            entry_price: Trade entry price
            atr: Current Average True Range value
            direction: "BUY" or "SELL"
            current_price: Current market price (optional, defaults to entry_price)
        
        Returns:
            Dictionary with:
            - stop_loss: Calculated stop-loss price
            - take_profit: Calculated take-profit price
            - stop_distance_pct: Stop distance as percentage
            - profit_distance_pct: Profit distance as percentage
            - risk_reward_ratio: Actual risk/reward ratio
            - atr_pct: ATR as percentage of price
        """
        if current_price is None:
            current_price = entry_price
        
        # Calculate ATR as percentage of price
        atr_pct = (atr / entry_price) * 100
        
        # Calculate raw stop and target distances
        stop_distance = atr * self.atr_multiplier_stop
        target_distance = atr * self.atr_multiplier_target
        
        # Convert to percentages
        stop_distance_pct = (stop_distance / entry_price) * 100
        profit_distance_pct = (target_distance / entry_price) * 100
        
        # Apply safety limits (prevent too tight or too wide stops)
        stop_distance_pct = max(self.min_risk_pct, min(stop_distance_pct, self.max_risk_pct))
        profit_distance_pct = stop_distance_pct * 2  # Always maintain 1:2 ratio
        
        # Recalculate actual distances after safety limits
        stop_distance = entry_price * (stop_distance_pct / 100)
        target_distance = entry_price * (profit_distance_pct / 100)
        
        # Calculate actual price levels based on direction
        if direction.upper() == "BUY":
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + target_distance
        else:  # SELL
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - target_distance
        
        # Calculate actual risk/reward ratio
        risk_reward_ratio = profit_distance_pct / stop_distance_pct
        
        result = {
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "stop_distance_pct": round(stop_distance_pct, 3),
            "profit_distance_pct": round(profit_distance_pct, 3),
            "risk_reward_ratio": round(risk_reward_ratio, 2),
            "atr": round(atr, 4),
            "atr_pct": round(atr_pct, 4),
            "entry_price": entry_price,
            "direction": direction.upper(),
        }
        
        # Log the calculation
        logger.info(f"📊 Dynamic Risk Calculation for {direction.upper()}:")
        logger.info(f"   💵 Entry: ${entry_price:.2f}")
        logger.info(f"   📉 ATR: ${atr:.4f} ({atr_pct:.4f}%)")
        logger.info(f"   🛑 Stop-Loss: ${result['stop_loss']:.2f} (-{stop_distance_pct:.3f}%)")
        logger.info(f"   🎯 Take-Profit: ${result['take_profit']:.2f} (+{profit_distance_pct:.3f}%)")
        logger.info(f"   ⚖️  Risk/Reward: 1:{risk_reward_ratio:.2f}")
        
        return result
    
    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        risk_percentage: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Calculate position size based on account balance and risk percentage.
        
        This ensures the system works identically on $10 or $10,000 accounts.
        
        Args:
            account_balance: Current account balance in USD
            entry_price: Trade entry price
            stop_loss: Stop-loss price
            risk_percentage: Percentage of account to risk (default 1.0%)
        
        Returns:
            Tuple of (position_size_in_asset, position_value_in_usd)
        """
        # Calculate risk amount in USD
        risk_amount_usd = account_balance * (risk_percentage / 100)
        
        # Calculate price distance to stop-loss
        price_distance = abs(entry_price - stop_loss)
        
        # Calculate position size in asset units
        # risk_amount = position_size × price_distance
        # position_size = risk_amount / price_distance
        position_size = risk_amount_usd / price_distance if price_distance > 0 else 0
        
        # Calculate position value in USD
        position_value_usd = position_size * entry_price
        
        logger.info("💰 Position Size Calculation:")
        logger.info(f"   🏦 Account Balance: ${account_balance:.2f}")
        logger.info(f"   🎲 Risk Amount: ${risk_amount_usd:.2f} ({risk_percentage}%)")
        logger.info(f"   📊 Position Size: {position_size:.6f} units")
        logger.info(f"   💵 Position Value: ${position_value_usd:.2f}")
        
        return position_size, position_value_usd
    
    def calculate_progressive_trailing_stops(
        self,
        entry_price: float,
        current_price: float,
        direction: str,
        take_profit: float,
    ) -> Dict:
        """
        Calculate progressive trailing stop levels.
        
        Progressive Strategy:
        - Phase 1: +0.3% profit → move stop to breakeven (entry price)
        - Phase 2: +0.5% profit → move stop to +0.3% profit
        - Phase 3: +0.7% profit → move stop to +0.5% profit
        - Phase 4: Reach take-profit target
        
        Args:
            entry_price: Original entry price
            current_price: Current market price
            direction: "BUY" or "SELL"
            take_profit: Take-profit target price
        
        Returns:
            Dictionary with trailing stop information
        """
        # Calculate current profit percentage
        if direction.upper() == "BUY":
            profit_pct = ((current_price - entry_price) / entry_price) * 100
        else:  # SELL
            profit_pct = ((entry_price - current_price) / entry_price) * 100
        
        # Determine trailing stop phase and level
        trailing_stop = None
        phase = "NONE"
        phase_description = "Position not yet in profit"
        
        if profit_pct >= 0.7:
            # Phase 3: Lock in +0.5% profit
            if direction.upper() == "BUY":
                trailing_stop = entry_price * 1.005  # +0.5%
            else:
                trailing_stop = entry_price * 0.995  # +0.5%
            phase = "PHASE_3"
            phase_description = "Trailing at +0.5% profit (approaching target)"
            
        elif profit_pct >= 0.5:
            # Phase 2: Lock in +0.3% profit
            if direction.upper() == "BUY":
                trailing_stop = entry_price * 1.003  # +0.3%
            else:
                trailing_stop = entry_price * 0.997  # +0.3%
            phase = "PHASE_2"
            phase_description = "Trailing at +0.3% profit (mid-trade)"
            
        elif profit_pct >= 0.3:
            # Phase 1: Move to breakeven
            trailing_stop = entry_price
            phase = "PHASE_1"
            phase_description = "Trailing at breakeven (capital protected)"
        
        return {
            "current_profit_pct": round(profit_pct, 3),
            "trailing_stop": round(trailing_stop, 2) if trailing_stop else None,
            "phase": phase,
            "phase_description": phase_description,
            "should_trail": trailing_stop is not None,
            "distance_to_target_pct": round(
                abs((take_profit - current_price) / current_price * 100), 3
            ) if take_profit else 0,
        }
