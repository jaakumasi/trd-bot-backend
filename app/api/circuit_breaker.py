"""
Circuit Breaker Admin API
=========================
Endpoints for monitoring and managing circuit breaker state.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from ..database import get_db
from .auth import get_current_user
from ..models.user import User
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/circuit-breaker", tags=["Circuit Breaker"])

# This will be injected by main.py when bot starts
_circuit_breaker_instance = None


def set_circuit_breaker_instance(circuit_breaker):
    """Set the circuit breaker instance from trading bot"""
    global _circuit_breaker_instance
    _circuit_breaker_instance = circuit_breaker
    logger.info("🛡️  Circuit breaker instance registered with API")


def get_circuit_breaker():
    """Dependency to get circuit breaker instance"""
    if _circuit_breaker_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Circuit breaker not initialized - Trading bot may not be running"
        )
    return _circuit_breaker_instance


@router.get("/status/{user_id}")
async def get_user_circuit_breaker_status(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get circuit breaker status for a specific user.
    
    Returns:
    - is_halted: Whether trading is currently halted
    - daily_pnl: Total P&L for today
    - daily_trade_count: Number of trades today
    - consecutive_losses: Current loss streak
    - trades_remaining_today: How many more trades allowed
    - halt_info: Details if currently halted
    """
    circuit_breaker = get_circuit_breaker()
    
    # Only allow users to see their own status (or admins to see any)
    if current_user.id != user_id and not getattr(current_user, 'is_admin', False):
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Get user's account balance (simplified - you may want to get real balance)
    from ..services.binance_service import BinanceService
    from ..services.mock_binance_service import MockBinanceService
    from ..config import settings
    
    binance = MockBinanceService() if settings.use_mock_binance else BinanceService()
    await binance.initialize()
    balances = binance.get_account_balance()
    usdt_balance = balances.get("USDT", {}).get("free", 1000.0)
    
    status = circuit_breaker.get_user_status(user_id, usdt_balance)
    
    return {
        "success": True,
        "user_id": user_id,
        "status": status,
        "config": {
            "max_daily_loss_pct": circuit_breaker.config['max_daily_loss_pct'],
            "max_consecutive_losses": circuit_breaker.config['max_consecutive_losses'],
            "max_daily_trades": circuit_breaker.config['max_daily_trades'],
        }
    }


@router.get("/status")
async def get_all_users_status(
    current_user: User = Depends(get_current_user),
):
    """
    Get circuit breaker status for all users (admin only).
    Shows overview of all users being tracked.
    """
    # Check if user is admin (you'll need to add this field to User model)
    if not getattr(current_user, 'is_admin', False):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    circuit_breaker = get_circuit_breaker()
    
    # Get all tracked users
    all_users = set()
    all_users.update(circuit_breaker.daily_pnl.keys())
    all_users.update(circuit_breaker.halted_users.keys())
    
    users_status = []
    for user_id in all_users:
        # Use a default balance for overview
        status = circuit_breaker.get_user_status(user_id, 1000.0)
        users_status.append({
            "user_id": user_id,
            "is_halted": status['is_halted'],
            "daily_pnl": status['daily_pnl'],
            "daily_trade_count": status['daily_trade_count'],
            "consecutive_losses": status['consecutive_losses'],
        })
    
    return {
        "success": True,
        "total_users": len(users_status),
        "users": users_status,
        "api_health": {
            "failure_count": circuit_breaker.api_failure_count,
            "cooldown_active": circuit_breaker.api_cooldown_until is not None,
        }
    }


@router.post("/resume/{user_id}")
async def manually_resume_trading(
    user_id: int,
    admin_override: bool = False,
    current_user: User = Depends(get_current_user),
):
    """
    Manually resume trading for a halted user.
    
    Parameters:
    - admin_override: If True, bypass time restrictions (admin only)
    """
    circuit_breaker = get_circuit_breaker()
    
    # Only allow users to resume their own trading or admins with override
    if current_user.id != user_id:
        if not getattr(current_user, 'is_admin', False):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Admin can override but must explicitly set the flag
        if admin_override and not getattr(current_user, 'is_admin', False):
            raise HTTPException(status_code=403, detail="Admin access required for override")
    
    # Attempt to resume
    resumed = circuit_breaker.manual_resume(user_id, admin_override=admin_override)
    
    if resumed:
        logger.info(f"✅ [User {user_id}] Trading manually resumed (admin_override={admin_override})")
        return {
            "success": True,
            "message": f"Trading resumed for user {user_id}",
            "admin_override": admin_override
        }
    else:
        # User not halted
        if user_id not in circuit_breaker.halted_users:
            return {
                "success": True,
                "message": f"User {user_id} is not currently halted",
            }
        else:
            # Halt period not expired and no override
            halt_info = circuit_breaker.halted_users[user_id]
            return {
                "success": False,
                "message": "Halt period not expired. Use admin_override=true to force resume.",
                "halt_info": {
                    "reason": halt_info['reason'],
                    "halted_until": halt_info['halted_until'].isoformat(),
                }
            }


@router.post("/reset/{user_id}")
async def reset_user_circuit_breaker(
    user_id: int,
    current_user: User = Depends(get_current_user),
):
    """
    Reset all circuit breaker state for a user (admin only).
    Use with caution - clears all tracking data.
    """
    if not getattr(current_user, 'is_admin', False):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    circuit_breaker = get_circuit_breaker()
    circuit_breaker.reset_for_user(user_id)
    
    logger.warning(f"⚠️  [Admin] Circuit breaker reset for user {user_id} by {current_user.email}")
    
    return {
        "success": True,
        "message": f"Circuit breaker state reset for user {user_id}",
    }


@router.get("/config")
async def get_circuit_breaker_config(
    current_user: User = Depends(get_current_user),
):
    """Get current circuit breaker configuration"""
    circuit_breaker = get_circuit_breaker()
    
    return {
        "success": True,
        "config": circuit_breaker.config
    }
