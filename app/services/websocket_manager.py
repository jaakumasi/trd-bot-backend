from fastapi import WebSocket
from typing import Dict, List
import json
import asyncio
import logging

logger = logging.getLogger(__name__)


class WebSocketManager:
    def __init__(self):
        # Dictionary to store active connections: {user_id: [websocket1, websocket2, ...]}
        self.active_connections: Dict[int, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: int):
        """Accept a websocket connection and add to active connections"""
        await websocket.accept()

        if user_id not in self.active_connections:
            self.active_connections[user_id] = []

        self.active_connections[user_id].append(websocket)
        total_connections = self.get_connection_count()
        total_users = len(self.active_connections)
        
        logger.info(f"🌐 WebSocket connected for user {user_id} | Total users: {total_users} | Total connections: {total_connections}")

        # Send initial connection confirmation
        await self.send_to_user(
            user_id,
            {
                "type": "connection_established",
                "user_id": user_id,
                "timestamp": asyncio.get_event_loop().time(),
            },
        )

    def disconnect(self, websocket: WebSocket, user_id: int):
        """Remove websocket from active connections"""
        if user_id in self.active_connections:
            if websocket in self.active_connections[user_id]:
                self.active_connections[user_id].remove(websocket)

            # Remove user entry if no more connections
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]

        total_connections = self.get_connection_count()
        total_users = len(self.active_connections)
        logger.info(f"🔌 WebSocket disconnected for user {user_id} | Remaining users: {total_users} | Remaining connections: {total_connections}")

    async def send_to_user(self, user_id: int, message: dict):
        """Send message to all connections for a specific user"""
        if user_id not in self.active_connections:
            return

        # Remove closed connections
        active_connections = []

        for websocket in self.active_connections[user_id]:
            try:
                await websocket.send_text(json.dumps(message))
                active_connections.append(websocket)
            except Exception as e:
                logger.warning(f"Failed to send message to user {user_id}: {e}")

        # Update active connections list
        self.active_connections[user_id] = active_connections

        if not active_connections:
            del self.active_connections[user_id]

    async def broadcast(self, message: dict):
        """Send message to all connected users"""
        for user_id in self.active_connections.keys():
            await self.send_to_user(user_id, message)

    def get_connected_users(self) -> List[int]:
        """Get list of currently connected user IDs"""
        return list(self.active_connections.keys())

    def get_connection_count(self) -> int:
        """Get total number of active connections"""
        return sum(len(connections) for connections in self.active_connections.values())
