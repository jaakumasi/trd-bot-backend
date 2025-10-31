from typing import Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update
from datetime import datetime, timezone
import logging

from ..models.trade import OpenPosition, Trade
from .binance_service import BinanceService

logger = logging.getLogger(__name__)


class PositionSyncManager:
    """
    Manages synchronization between database positions and Binance actual state.
    Binance is always the source of truth for conflict resolution.
    """

    def __init__(self, binance_service: BinanceService):
        self.binance_service = binance_service

    async def sync_user_positions(self, db: AsyncSession, user_id: int) -> Dict:
        """
        Synchronize database positions with Binance state for a specific user.
        Returns summary of sync operations performed.
        """
        sync_summary = {
            "user_id": user_id,
            "positions_checked": 0,
            "positions_updated": 0,
            "positions_closed": 0,
            "oco_orders_updated": 0,
            "conflicts_resolved": 0,
            "errors": []
        }

        try:
            logger.info(f"🔄 Starting position sync for user {user_id}")
            db_positions = await self._get_db_positions(db, user_id)
            sync_summary["positions_checked"] = len(db_positions)

            if not db_positions:
                logger.info(f"📭 No open positions found in database for user {user_id}")
                return sync_summary

            # Process each position
            for db_position in db_positions:
                try:
                    result = await self._process_position_sync(db, db_position)
                    self._update_sync_summary(sync_summary, result)
                except Exception as pos_error:
                    error_msg = f"Position {db_position.trade_id}: {pos_error}"
                    sync_summary["errors"].append(error_msg)
                    logger.error(f"❌ {error_msg}")

            await db.commit()
            logger.info(f"✅ Position sync completed for user {user_id}: {sync_summary}")

        except Exception as e:
            error_msg = f"Sync failed for user {user_id}: {e}"
            sync_summary["errors"].append(error_msg)
            logger.error(f"❌ {error_msg}")
            await db.rollback()

        return sync_summary

    def _update_sync_summary(self, summary: Dict, result: Dict) -> None:
        """Update sync summary based on individual position result"""
        action = result["action"]
        if action == "updated":
            summary["positions_updated"] += 1
        elif action == "closed":
            summary["positions_closed"] += 1
        elif action == "oco_updated":
            summary["oco_orders_updated"] += 1
        elif action == "conflict_resolved":
            summary["conflicts_resolved"] += 1

    async def _get_db_positions(self, db: AsyncSession, user_id: int) -> List[OpenPosition]:
        """Get all open positions for a user from database (excluding paper trades)"""
        result = await db.execute(
            select(OpenPosition).where(
                OpenPosition.user_id == user_id,
                OpenPosition.is_paper_trade == False  # Exclude paper trades from sync
            )
        )
        return result.scalars().all()

    async def _process_position_sync(self, db: AsyncSession, db_position: OpenPosition) -> Dict:
        """
        Process synchronization for a single position.
        Returns action taken: 'no_change', 'updated', 'closed', 'oco_updated', 'conflict_resolved'
        """
        position_info = f"{db_position.symbol} ({db_position.trade_id})"
        
        if db_position.oco_order_id:
            return await self._sync_oco_position(db, db_position, position_info)
        else:
            return await self._sync_non_oco_position(db, db_position, position_info)

    async def _sync_oco_position(
        self,
        db: AsyncSession,
        db_position: OpenPosition,
        position_info: str
    ) -> Dict:
        """Synchronize position with OCO order tracking"""
        oco_id = db_position.oco_order_id
        oco_status = self.binance_service.get_oco_order_status(oco_id)
        
        if not oco_status:
            logger.warning(f"⚠️  Could not get OCO status for {position_info} (OCO: {oco_id})")
            return {"action": "no_change", "reason": "oco_status_unavailable"}

        return await self._handle_oco_status(db, db_position, oco_status, position_info)

    async def _handle_oco_status(
        self,
        db: AsyncSession,
        db_position: OpenPosition,
        oco_status: Dict,
        position_info: str
    ) -> Dict:
        """Handle different OCO status scenarios"""
        oco_list_status = oco_status.get("listOrderStatus", "UNKNOWN")
        
        if oco_list_status == "ALL_DONE":
            await self._close_position_from_oco(db, db_position, oco_status, position_info)
            return {"action": "closed", "reason": "oco_completed"}
        elif oco_list_status == "EXECUTING":
            logger.debug(f"📍 Position {position_info} OCO still executing")
            return {"action": "no_change", "reason": "oco_executing"}
        elif oco_list_status == "REJECT":
            logger.warning(f"⚠️  OCO rejected for {position_info} - resolving conflict")
            await self._resolve_oco_conflict(db, db_position, position_info)
            return {"action": "conflict_resolved", "reason": "oco_rejected"}
        else:
            logger.warning(f"⚠️  Unknown OCO status '{oco_list_status}' for {position_info}")
            return {"action": "no_change", "reason": f"unknown_oco_status_{oco_list_status}"}

    async def _sync_non_oco_position(
        self,
        db: AsyncSession,
        db_position: OpenPosition,
        position_info: str
    ) -> Dict:
        """Synchronize position without OCO order - use balance-based detection"""
        symbol = db_position.symbol
        base_asset = symbol.replace("USDT", "").replace("BUSD", "")
        
        account_positions = self.binance_service.get_account_positions()
        current_holding = self._get_asset_holding(account_positions, base_asset)
        
        # For day trading, if we don't hold the asset, position is likely closed
        if current_holding < 0.000001:  # Use small threshold instead of exact zero
            logger.info(f"🔄 Position {position_info} appears closed (no {base_asset} holdings)")
            await self._close_position_manual(db, db_position, "BALANCE_CHECK")
            return {"action": "closed", "reason": "no_balance"}
        else:
            logger.debug(f"📍 Position {position_info} still open ({current_holding} {base_asset})")
            return {"action": "no_change", "reason": "balance_indicates_open"}

    def _get_asset_holding(self, account_positions: List[Dict], base_asset: str) -> float:
        """Get current holding amount for a specific asset"""
        for pos in account_positions:
            if pos["asset"] == base_asset:
                return pos["total"]
        return 0.0

    async def _close_position_from_oco(
        self,
        db: AsyncSession,
        db_position: OpenPosition,
        oco_status: Dict,
        position_info: str
    ) -> None:
        """Close position and create Trade record from OCO completion"""
        try:
            executed_order, exit_reason = self._analyze_oco_execution(oco_status, db_position)
            exit_price = float(executed_order.get("price", 0))
            
            await self._create_trade_from_position(
                db, db_position, exit_price, exit_reason
            )
            
            await db.delete(db_position)
            logger.info(f"✅ Position {position_info} closed via {exit_reason} at ${exit_price}")
            
        except Exception as e:
            logger.error(f"❌ Error closing position {position_info}: {e}")
            raise

    def _analyze_oco_execution(self, oco_status: Dict, db_position: OpenPosition) -> tuple:
        """Analyze OCO execution to determine exit details"""
        executed_order = None
        exit_reason = "UNKNOWN"
        
        for order in oco_status.get("orders", []):
            if order.get("status") == "FILLED":
                executed_order = order
                exit_reason = self._determine_exit_reason(order, db_position)
                break

        if not executed_order:
            logger.warning(f"⚠️  No filled order found in OCO for {db_position.trade_id}")
            executed_order = {"price": "0", "executedQty": str(db_position.amount)}
            exit_reason = "OCO_UNKNOWN"

        return executed_order, exit_reason

    def _determine_exit_reason(self, order: Dict, db_position: OpenPosition) -> str:
        """Determine if order was take profit or stop loss"""
        exec_price = float(order["price"])
        entry_price = float(db_position.entry_price)
        
        if db_position.side.upper() == "BUY":
            # For long positions: TP > entry, SL < entry
            return "TAKE_PROFIT" if exec_price > entry_price else "STOP_LOSS"
        else:
            # For short positions: TP < entry, SL > entry
            return "TAKE_PROFIT" if exec_price < entry_price else "STOP_LOSS"

    async def _close_position_manual(
        self,
        db: AsyncSession,
        db_position: OpenPosition,
        reason: str
    ) -> None:
        """Close position without OCO details - use current market price estimate"""
        try:
            current_price = self.binance_service.get_symbol_price(db_position.symbol)
            await self._create_trade_from_position(db, db_position, current_price, reason)
            await db.delete(db_position)
            
            logger.info(f"✅ Position {db_position.symbol} ({db_position.trade_id}) closed - {reason}")
            
        except Exception as e:
            logger.error(f"❌ Error manually closing position {db_position.trade_id}: {e}")
            raise

    def _create_trade_from_position(
        self,
        db: AsyncSession,
        position: OpenPosition,
        exit_price: float,
        exit_reason: str
    ) -> None:
        """Create a completed Trade record from an OpenPosition"""
        now = datetime.now(timezone.utc)
        duration = (now - position.opened_at).total_seconds()
        
        # Calculate P&L
        entry_value = float(position.entry_value)
        exit_value = float(position.amount) * exit_price
        
        if position.side.upper() == "BUY":
            profit_loss = exit_value - entry_value - float(position.fees_paid)
        else:
            profit_loss = entry_value - exit_value - float(position.fees_paid)
        
        profit_loss_pct = (profit_loss / entry_value * 100) if entry_value > 0 else 0

        trade = Trade(
            user_id=position.user_id,
            trade_id=position.trade_id,
            symbol=position.symbol,
            side=position.side,
            amount=position.amount,
            price=position.entry_price,
            total_value=position.entry_value,
            fee=position.fees_paid,
            status='closed',
            is_test_trade=position.is_test_trade,
            executed_at=position.opened_at,
            oco_order_id=position.oco_order_id,
            closed_at=now,
            exit_price=exit_price,
            exit_fee=0,
            exit_reason=exit_reason,
            profit_loss=profit_loss,
            profit_loss_percentage=profit_loss_pct,
            duration_seconds=int(duration)
        )
        
        db.add(trade)
        logger.debug(f"📝 Created Trade record for {position.trade_id}: P&L ${profit_loss:.2f}")

    async def _resolve_oco_conflict(
        self,
        db: AsyncSession,
        db_position: OpenPosition,
        position_info: str
    ) -> None:
        """Handle OCO rejection conflicts - clear OCO ID and continue tracking"""
        try:
            await db.execute(
                update(OpenPosition)
                .where(OpenPosition.id == db_position.id)
                .values(oco_order_id=None)
            )
            
            logger.warning(f"🔧 Cleared rejected OCO for {position_info} - position continues without OCO")
            
        except Exception as e:
            logger.error(f"❌ Error resolving OCO conflict for {position_info}: {e}")
            raise

    async def sync_all_users_positions(self, db: AsyncSession, user_ids: List[int]) -> Dict:
        """
        Batch synchronization for multiple users.
        Returns aggregated sync summary.
        """
        batch_summary = {
            "users_processed": 0,
            "total_positions_checked": 0,
            "total_positions_updated": 0,
            "total_positions_closed": 0,
            "total_oco_orders_updated": 0,
            "total_conflicts_resolved": 0,
            "errors": []
        }

        for user_id in user_ids:
            try:
                sync_result = await self.sync_user_positions(db, user_id)
                self._aggregate_batch_results(batch_summary, sync_result)
                
            except Exception as user_error:
                error_msg = f"User {user_id} sync failed: {user_error}"
                batch_summary["errors"].append(error_msg)
                logger.error(f"❌ {error_msg}")

        logger.info(f"🔄 Batch position sync completed: {batch_summary}")
        return batch_summary

    def _aggregate_batch_results(self, batch_summary: Dict, sync_result: Dict) -> None:
        """Aggregate individual user sync results into batch summary"""
        batch_summary["users_processed"] += 1
        batch_summary["total_positions_checked"] += sync_result["positions_checked"]
        batch_summary["total_positions_updated"] += sync_result["positions_updated"]
        batch_summary["total_positions_closed"] += sync_result["positions_closed"]
        batch_summary["total_oco_orders_updated"] += sync_result["oco_orders_updated"]
        batch_summary["total_conflicts_resolved"] += sync_result["conflicts_resolved"]
        batch_summary["errors"].extend(sync_result["errors"])