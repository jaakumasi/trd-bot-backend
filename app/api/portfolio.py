from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import List, Dict
from decimal import Decimal
from datetime import datetime
from ..database import get_db
from ..models.user import User
from ..models.portfolio import Portfolio
from .auth import get_current_user

router = APIRouter()


# Pydantic models
class PortfolioItemResponse(BaseModel):
    asset: str
    balance: float
    locked_balance: float
    total_balance: float
    average_buy_price: float
    current_price: float
    value_usd: float
    pnl_percentage: float
    is_test_balance: bool
    updated_at: datetime


class PortfolioSummaryResponse(BaseModel):
    total_value_usd: float
    total_pnl_usd: float
    total_pnl_percentage: float
    assets: List[PortfolioItemResponse]
    last_updated: datetime


class BalanceUpdateRequest(BaseModel):
    asset: str
    amount: float
    is_test_balance: bool = True


@router.get("/", response_model=PortfolioSummaryResponse)
async def get_portfolio(
    test_mode: bool = True,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get user's portfolio with current values"""
    # Import here to avoid circular imports
    from ..services.binance_service import BinanceService

    # Get portfolio from database
    query = select(Portfolio).where(
        Portfolio.user_id == current_user.id, Portfolio.is_test_balance == test_mode
    )
    result = await db.execute(query)
    portfolio_items = result.scalars().all()

    # Get current prices
    binance_service = BinanceService()
    await binance_service.initialize()

    assets = []
    total_value_usd = 0.0
    total_pnl_usd = 0.0

    for item in portfolio_items:
        current_price = 0.0
        value_usd = 0.0
        pnl_percentage = 0.0

        if item.asset == "USDT":
            current_price = 1.0
            value_usd = float(item.balance)
        else:
            # Get current price from Binance
            try:
                price_symbol = f"{item.asset}USDT"
                current_price = binance_service.get_symbol_price(price_symbol)
                value_usd = float(item.balance) * current_price

                # Calculate P&L if we have average buy price
                if float(item.average_buy_price) > 0:
                    pnl_percentage = (
                        (current_price - float(item.average_buy_price))
                        / float(item.average_buy_price)
                    ) * 100

            except Exception as e:
                print(f"Error getting price for {item.asset}: {e}")

        total_balance = float(item.balance) + float(item.locked_balance)

        asset_data = PortfolioItemResponse(
            asset=item.asset,
            balance=float(item.balance),
            locked_balance=float(item.locked_balance),
            total_balance=total_balance,
            average_buy_price=float(item.average_buy_price),
            current_price=current_price,
            value_usd=value_usd,
            pnl_percentage=pnl_percentage,
            is_test_balance=item.is_test_balance,
            updated_at=item.updated_at,
        )

        assets.append(asset_data)
        total_value_usd += value_usd

        # Calculate P&L in USD
        if float(item.average_buy_price) > 0 and current_price > 0:
            cost_basis = float(item.balance) * float(item.average_buy_price)
            current_value = float(item.balance) * current_price
            total_pnl_usd += current_value - cost_basis

    # Calculate total P&L percentage
    total_pnl_percentage = 0.0
    if total_value_usd > 0 and total_pnl_usd != 0:
        cost_basis = total_value_usd - total_pnl_usd
        if cost_basis > 0:
            total_pnl_percentage = (total_pnl_usd / cost_basis) * 100

    return PortfolioSummaryResponse(
        total_value_usd=total_value_usd,
        total_pnl_usd=total_pnl_usd,
        total_pnl_percentage=total_pnl_percentage,
        assets=assets,
        last_updated=datetime.utcnow(),
    )


@router.post("/add-balance")
async def add_test_balance(
    balance_request: BalanceUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Add test balance for paper trading"""
    if not balance_request.is_test_balance:
        raise HTTPException(
            status_code=400, detail="Only test balances can be added manually"
        )

    # Check if portfolio item exists
    query = select(Portfolio).where(
        Portfolio.user_id == current_user.id,
        Portfolio.asset == balance_request.asset,
        Portfolio.is_test_balance == balance_request.is_test_balance,
    )
    result = await db.execute(query)
    portfolio_item = result.scalar_one_or_none()

    if portfolio_item:
        # Update existing balance
        portfolio_item.balance = Decimal(str(balance_request.amount))
        portfolio_item.updated_at = datetime.utcnow()
    else:
        # Create new portfolio item
        portfolio_item = Portfolio(
            user_id=current_user.id,
            asset=balance_request.asset,
            balance=Decimal(str(balance_request.amount)),
            locked_balance=Decimal("0"),
            average_buy_price=(
                Decimal("1") if balance_request.asset == "USDT" else Decimal("0")
            ),
            is_test_balance=balance_request.is_test_balance,
        )
        db.add(portfolio_item)

    await db.commit()

    return {
        "message": f"Test balance updated successfully",
        "asset": balance_request.asset,
        "balance": balance_request.amount,
        "is_test": balance_request.is_test_balance,
    }


@router.delete("/reset-test-balance")
async def reset_test_balance(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    """Reset all test balances to zero"""
    from sqlalchemy import delete

    # Delete all test portfolio items for user
    delete_query = delete(Portfolio).where(
        Portfolio.user_id == current_user.id, Portfolio.is_test_balance == True
    )
    await db.execute(delete_query)

    # Add default USDT test balance
    default_balance = Portfolio(
        user_id=current_user.id,
        asset="USDT",
        balance=Decimal("16.00"),  # $16 USD equivalent to 200 GHS
        locked_balance=Decimal("0"),
        average_buy_price=Decimal("1"),
        is_test_balance=True,
    )
    db.add(default_balance)

    await db.commit()

    return {"message": "Test balance reset to $16 USDT"}


@router.get("/prices/{symbol}")
async def get_current_price(
    symbol: str, current_user: User = Depends(get_current_user)
):
    """Get current price for a symbol"""
    from ..services.binance_service import BinanceService

    binance_service = BinanceService()
    await binance_service.initialize()

    try:
        price = binance_service.get_symbol_price(symbol)
        return {
            "symbol": symbol,
            "price": price,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error getting price: {str(e)}")
