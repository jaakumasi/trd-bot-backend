from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import asyncio
import json
from typing import List
from .config import settings
from .database import engine, Base
from .api import trading, portfolio, auth
from .services.trading_bot import TradingBot
from .services.websocket_manager import WebSocketManager
from .logging_config import setup_logging
# Import all models to ensure they're registered with SQLAlchemy
from .models.user import User
from .models.trade import Trade, TradingConfig, OpenPosition
from .models.portfolio import Portfolio
from dotenv import load_dotenv
import logging

load_dotenv("../.env")

# Initialize logging system
trading_metrics_logger = setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Crypto Trading Bot API",
    description="AI-powered cryptocurrency trading bot with real-time monitoring",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket manager
ws_manager = WebSocketManager()

# Initialize trading bot
trading_bot = None


@app.on_event("startup")
async def startup_event():
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Initialize trading bot
    global trading_bot
    trading_bot = TradingBot(ws_manager)
    
    # Start the trading bot
    bot_started = await trading_bot.start()
    if bot_started:
        logger.info("🚀 Trading Bot started and running!")
    else:
        logger.error("❌ Failed to start Trading Bot")

    logger.info("🤖 Trd Bot API Started Successfully!")
    logger.info("📊 Database: Connected to PostgreSQL db")
    print(f"🔗 CORS Origins: {settings.cors_origins}")
    print(f"🧪 Test Mode: {settings.binance_testnet}")


@app.on_event("shutdown")
async def shutdown_event():
    if trading_bot:
        await trading_bot.stop()
    print("🛑 Trading Bot Stopped")


# Include API routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(trading.router, prefix="/trading", tags=["Trading"])
app.include_router(portfolio.router, prefix="/portfolio", tags=["Portfolio"])


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "bot_active": trading_bot.is_running if trading_bot else False,
        "test_mode": settings.binance_testnet,
    }


@app.get("/debug/bot-status")
async def debug_bot_status():
    """Debug endpoint to check trading bot status"""
    if not trading_bot:
        return {"error": "Trading bot not initialized"}
    
    from datetime import datetime
    from .database import get_db
    
    # Get active users count
    active_users_count = 0
    try:
        async for db in get_db():
            active_configs = await trading_bot._get_active_trading_configs(db)
            active_users_count = len(active_configs)
            break
    except Exception as e:
        logger.error(f"Error getting active users: {e}")
    
    connected_users = ws_manager.get_connected_users()
    
    return {
        "bot_running": trading_bot.is_running,
        "active_users_count": active_users_count,
        "connected_users": connected_users,
        "total_connections": ws_manager.get_connection_count(),
        "current_time": datetime.now().isoformat(),
        "is_trading_hours": trading_bot._is_trading_hours(datetime.now().time()),
        "last_analysis_times": dict(trading_bot.last_analysis_time),
    }


@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int):
    await ws_manager.connect(websocket, user_id)
    try:
        while True:
            # Keep connection alive and listen for client messages
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle different message types
            if message["type"] == "ping":
                await ws_manager.send_to_user(
                    user_id, {"type": "pong", "timestamp": message.get("timestamp")}
                )
            elif message["type"] == "get_status":
                await ws_manager.send_to_user(
                    user_id,
                    {
                        "type": "status_update",
                        "bot_running": trading_bot.is_running if trading_bot else False,
                        "test_mode": settings.binance_testnet,
                    },
                )

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, user_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket, user_id)


@app.get("/")
async def root():
    return {
        "message": "Trd Bot API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "active",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
