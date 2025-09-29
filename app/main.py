from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
import asyncio
import json
from typing import List
from .config import settings
from .database import engine, Base
from .api import trading, portfolio, auth
from .services.trading_bot import TradingBot
from .services.websocket_manager import WebSocketManager

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

    print("🤖 Crypto Trading Bot API Started Successfully!")
    print(f"📊 Database: Connected to Neon PostgreSQL")
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
    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
