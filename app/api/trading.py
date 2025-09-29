from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from pydantic import BaseModel
from typing import List, Optional
from decimal import Decimal
from datetime import datetime
from ..database import get_db
from ..models.user import User
from ..models.trade import TradingConfig, Trade
from ..models.portfolio import Portfolio
from .auth import get_current_user

router = APIRouter()


# Pydantic models
class TradingConfigUpdate(BaseModel):
    strategy_name: Optional[str] = None
    risk_percentage: Optional[float] = None
    trading_pair: Optional[str] = None
    is_active: Optional[bool] = None
    is_test_mode: Optional[bool] = None
    max_daily_trades: Optional[int] = None
    stop_loss_percentage: Optional[float] = None
    take_profit_percentage: Optional[float] = None


class TradingConfigResponse(BaseModel):
    id: int
    strategy_name: str
    risk_percentage: float
    trading_pair: str
    is_active: bool
    is_test_mode: bool
    max_daily_trades: int
    stop_loss_percentage: float
    take_profit_percentage: float
    updated_at: datetime


class TradeResponse(BaseModel):
    id: int
    trade_id: str
    symbol: str
    side: str
    amount: float
    price: float
    total_value: float
    fee: float
    status: str
    is_test_trade: bool
    strategy_used: Optional[str]
    ai_signal_confidence: Optional[float]
    executed_at: datetime
    closed_at: Optional[datetime]


class TradingStatsResponse(BaseModel):
    total_trades: int
    profitable_trades: int
    total_pnl: float
    win_rate: float
    daily_trades_remaining: int
    last_trade: Optional[datetime]


@router.get("/config", response_model=TradingConfigResponse)
async def get_trading_config(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    """Get user's trading configuration"""
    query = select(TradingConfig).where(TradingConfig.user_id == current_user.id)
    result = await db.execute(query)
    config = result.scalar_one_or_none()

    if not config:
        raise HTTPException(status_code=404, detail="Trading configuration not found")

    return TradingConfigResponse(
        id=config.id,
        strategy_name=config.strategy_name,
        risk_percentage=float(config.risk_percentage),
        trading_pair=config.trading_pair,
        is_active=config.is_active,
        is_test_mode=config.is_test_mode,
        max_daily_trades=config.max_daily_trades,
        stop_loss_percentage=float(config.stop_loss_percentage),
        take_profit_percentage=float(config.take_profit_percentage),
        updated_at=config.updated_at,
    )


@router.put("/config", response_model=TradingConfigResponse)
async def update_trading_config(
    config_update: TradingConfigUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update user's trading configuration"""
    query = select(TradingConfig).where(TradingConfig.user_id == current_user.id)
    result = await db.execute(query)
    config = result.scalar_one_or_none()

    if not config:
        raise HTTPException(status_code=404, detail="Trading configuration not found")

    # Update fields
    update_data = config_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        if hasattr(config, field):
            setattr(config, field, value)

    config.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(config)

    return TradingConfigResponse(
        id=config.id,
        strategy_name=config.strategy_name,
        risk_percentage=float(config.risk_percentage),
        trading_pair=config.trading_pair,
        is_active=config.is_active,
        is_test_mode=config.is_test_mode,
        max_daily_trades=config.max_daily_trades,
        stop_loss_percentage=float(config.stop_loss_percentage),
        take_profit_percentage=float(config.take_profit_percentage),
        updated_at=config.updated_at,
    )


@router.post("/start")
async def start_trading(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    """Start automated trading for user"""
    query = select(TradingConfig).where(TradingConfig.user_id == current_user.id)
    result = await db.execute(query)
    config = result.scalar_one_or_none()

    if not config:
        raise HTTPException(status_code=404, detail="Trading configuration not found")

    config.is_active = True
    config.updated_at = datetime.utcnow()

    await db.commit()

    return {"message": "Trading started successfully", "is_active": True}


@router.post("/stop")
async def stop_trading(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    """Stop automated trading for user"""
    query = select(TradingConfig).where(TradingConfig.user_id == current_user.id)
    result = await db.execute(query)
    config = result.scalar_one_or_none()

    if not config:
        raise HTTPException(status_code=404, detail="Trading configuration not found")

    config.is_active = False
    config.updated_at = datetime.utcnow()

    await db.commit()

    return {"message": "Trading stopped successfully", "is_active": False}


@router.get("/trades", response_model=List[TradeResponse])
async def get_trades(
    limit: int = 50,
    skip: int = 0,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get user's trade history"""
    query = (
        select(Trade)
        .where(Trade.user_id == current_user.id)
        .order_by(desc(Trade.executed_at))
        .offset(skip)
        .limit(limit)
    )

    result = await db.execute(query)
    trades = result.scalars().all()

    return [
        TradeResponse(
            id=trade.id,
            trade_id=trade.trade_id,
            symbol=trade.symbol,
            side=trade.side,
            amount=float(trade.amount),
            price=float(trade.price),
            total_value=float(trade.total_value),
            fee=float(trade.fee),
            status=trade.status,
            is_test_trade=trade.is_test_trade,
            strategy_used=trade.strategy_used,
            ai_signal_confidence=(
                float(trade.ai_signal_confidence)
                if trade.ai_signal_confidence
                else None
            ),
            executed_at=trade.executed_at,
            closed_at=trade.closed_at,
        )
        for trade in trades
    ]


@router.get("/stats", response_model=TradingStatsResponse)
async def get_trading_stats(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    """Get user's trading statistics"""
    from sqlalchemy import func, and_
    from datetime import date

    # Get total trades
    total_trades_query = select(func.count(Trade.id)).where(
        Trade.user_id == current_user.id
    )
    total_trades_result = await db.execute(total_trades_query)
    total_trades = total_trades_result.scalar() or 0

    # Get today's trades count
    today = date.today()
    today_trades_query = select(func.count(Trade.id)).where(
        and_(Trade.user_id == current_user.id, func.date(Trade.executed_at) == today)
    )
    today_trades_result = await db.execute(today_trades_query)
    today_trades = today_trades_result.scalar() or 0

    # Get user's max daily trades setting
    config_query = select(TradingConfig).where(TradingConfig.user_id == current_user.id)
    config_result = await db.execute(config_query)
    config = config_result.scalar_one_or_none()
    max_daily_trades = config.max_daily_trades if config else 10

    # Calculate basic stats (simplified for now)
    profitable_trades = 0
    total_pnl = 0.0
    win_rate = 0.0
    last_trade = None

    # Get last trade
    last_trade_query = (
        select(Trade)
        .where(Trade.user_id == current_user.id)
        .order_by(desc(Trade.executed_at))
        .limit(1)
    )
    last_trade_result = await db.execute(last_trade_query)
    last_trade_obj = last_trade_result.scalar_one_or_none()

    if last_trade_obj:
        last_trade = last_trade_obj.executed_at

    return TradingStatsResponse(
        total_trades=total_trades,
        profitable_trades=profitable_trades,
        total_pnl=total_pnl,
        win_rate=win_rate,
        daily_trades_remaining=max(0, max_daily_trades - today_trades),
        last_trade=last_trade,
    )
