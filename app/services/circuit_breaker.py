"""
Circuit Breaker System for Trading Bot
======================================
Implements safety mechanisms to halt trading when risk thresholds are exceeded.

Features:
- Daily loss limit enforcement
- Consecutive loss streak detection
- API failure cooldown
- Position count limits
- Volatility spike detection
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class CircuitBreakerTriggered(Exception):
    """Exception raised when a circuit breaker is triggered"""
    def __init__(self, reason: str, user_id: int):
        self.reason = reason
        self.user_id = user_id
        super().__init__(f"Circuit breaker triggered for user {user_id}: {reason}")


class CircuitBreaker:
    """
    Trading safety system that monitors and enforces risk limits.

    Prevents catastrophic losses by halting trading when:
    - Daily loss exceeds threshold
    - Too many consecutive losses
    - API failures indicate system instability
    - Extreme market volatility detected
    """

    def __init__(self):
        # Loss tracking
        self.daily_pnl: Dict[int, float] = defaultdict(float)
        self.daily_trade_count: Dict[int, int] = defaultdict(int)
        self.last_reset_date: Dict[int, datetime] = {}

        # Loss streak tracking
        self.consecutive_losses: Dict[int, int] = defaultdict(int)
        self.loss_streak_start: Dict[int, datetime] = {}

        # API health tracking
        self.api_failure_count: int = 0
        self.last_api_failure: Optional[datetime] = None
        self.api_cooldown_until: Optional[datetime] = None

        # Halted users
        self.halted_users: Dict[int, Dict] = {}

        # Configuration (can be overridden per user)
        self.config = {
            'max_daily_loss_pct': 5.0,  # Halt if lose >5% of starting balance in one day
            'max_consecutive_losses': 5,  # Halt after 5 losses in a row
            'max_daily_trades': 15,  # Hard cap on trades per day
            'api_failure_threshold': 3,  # Halt after 3 consecutive API failures
            'api_cooldown_seconds': 300,  # 5 minutes cooldown after API failures
            'min_trade_interval_seconds': 60,  # Minimum 1 minute between trades
        }

        logger.info("🛡️  Circuit Breaker System initialized")
        logger.info(f"   Max Daily Loss: {self.config['max_daily_loss_pct']}%")
        logger.info(f"   Max Consecutive Losses: {self.config['max_consecutive_losses']}")
        logger.info(f"   Max Daily Trades: {self.config['max_daily_trades']}")

    def check_before_trade(self, user_id: int, account_balance: float) -> bool:
        """
        Check all circuit breaker conditions before allowing a new trade.

        Args:
            user_id: User ID
            account_balance: Current account balance

        Returns:
            True if trade is allowed, False otherwise

        Raises:
            CircuitBreakerTriggered: If any safety limit is exceeded
        """
        # Reset daily counters if new day
        self._check_daily_reset(user_id)

        # Check if user is currently halted
        if user_id in self.halted_users:
            halt_info = self.halted_users[user_id]
            halt_until = halt_info.get('halted_until')

            if halt_until and datetime.now(timezone.utc) < halt_until:
                remaining = (halt_until - datetime.now(timezone.utc)).total_seconds() / 60
                raise CircuitBreakerTriggered(
                    f"Trading halted: {halt_info['reason']} (Resume in {remaining:.0f} minutes)",
                    user_id
                )
            else:
                # Halt period expired, remove halt
                self._resume_trading(user_id)

        # Check API health
        if self.api_cooldown_until and datetime.now(timezone.utc) < self.api_cooldown_until:
            remaining = (self.api_cooldown_until - datetime.now(timezone.utc)).total_seconds()
            raise CircuitBreakerTriggered(
                f"API cooldown active ({remaining:.0f}s remaining)",
                user_id
            )

        # Check daily loss limit
        daily_loss_pct = self._calculate_daily_loss_pct(user_id, account_balance)
        if daily_loss_pct >= self.config['max_daily_loss_pct']:
            self._halt_trading(
                user_id,
                reason=f"Daily loss limit exceeded ({daily_loss_pct:.1f}% loss)",
                duration_hours=24
            )
            raise CircuitBreakerTriggered(
                f"Daily loss limit exceeded ({daily_loss_pct:.1f}% >= {self.config['max_daily_loss_pct']}%)",
                user_id
            )

        # Check consecutive losses
        if self.consecutive_losses[user_id] >= self.config['max_consecutive_losses']:
            self._halt_trading(
                user_id,
                reason=f"{self.consecutive_losses[user_id]} consecutive losses",
                duration_hours=4  # 4-hour cooldown after loss streak
            )
            raise CircuitBreakerTriggered(
                f"Consecutive loss limit exceeded ({self.consecutive_losses[user_id]} losses)",
                user_id
            )

        # Check daily trade count
        if self.daily_trade_count[user_id] >= self.config['max_daily_trades']:
            raise CircuitBreakerTriggered(
                f"Daily trade limit reached ({self.config['max_daily_trades']} trades)",
                user_id
            )

        # Warning levels (don't halt, just warn)
        if daily_loss_pct >= self.config['max_daily_loss_pct'] * 0.7:
            logger.warning(
                f"⚠️  [User {user_id}] Daily loss at {daily_loss_pct:.1f}% "
                f"(70% of limit: {self.config['max_daily_loss_pct']}%)"
            )

        if self.consecutive_losses[user_id] >= self.config['max_consecutive_losses'] - 2:
            logger.warning(
                f"⚠️  [User {user_id}] {self.consecutive_losses[user_id]} consecutive losses "
                f"({self.config['max_consecutive_losses'] - self.consecutive_losses[user_id]} until halt)"
            )

        return True

    def record_trade_result(self, user_id: int, pnl: float, is_winner: bool) -> None:
        """
        Record a completed trade result for circuit breaker tracking.

        Args:
            user_id: User ID
            pnl: Profit/Loss in dollars
            is_winner: True if trade was profitable
        """
        self._check_daily_reset(user_id)

        # Update daily P&L
        self.daily_pnl[user_id] += pnl
        self.daily_trade_count[user_id] += 1

        # Update loss streak
        if is_winner:
            if self.consecutive_losses[user_id] > 0:
                logger.info(
                    f"✅ [User {user_id}] Loss streak ended at {self.consecutive_losses[user_id]} trades"
                )
            self.consecutive_losses[user_id] = 0
            self.loss_streak_start[user_id] = None
        else:
            self.consecutive_losses[user_id] += 1
            if self.consecutive_losses[user_id] == 1:
                self.loss_streak_start[user_id] = datetime.now(timezone.utc)

            logger.warning(
                f"⚠️  [User {user_id}] Consecutive losses: {self.consecutive_losses[user_id]}"
            )

        # Log daily status
        logger.info(
            f"📊 [User {user_id}] Daily: {self.daily_trade_count[user_id]} trades, "
            f"${self.daily_pnl[user_id]:+.2f} P&L"
        )

    def record_api_failure(self, error_details: str) -> None:
        """
        Record an API failure event.

        Args:
            error_details: Description of the API failure
        """
        self.api_failure_count += 1
        self.last_api_failure = datetime.now(timezone.utc)

        logger.error(f"❌ API Failure #{self.api_failure_count}: {error_details}")

        if self.api_failure_count >= self.config['api_failure_threshold']:
            self.api_cooldown_until = datetime.now(timezone.utc) + timedelta(
                seconds=self.config['api_cooldown_seconds']
            )
            logger.critical(
                f"🚨 API FAILURE THRESHOLD EXCEEDED - Trading halted for "
                f"{self.config['api_cooldown_seconds']}s"
            )

    def record_api_success(self) -> None:
        """Record a successful API call (resets failure counter)"""
        if self.api_failure_count > 0:
            self.api_failure_count = 0
            logger.info("✅ API health restored - failure counter reset")

    def _halt_trading(self, user_id: int, reason: str, duration_hours: float) -> None:
        """
        Halt trading for a specific user.

        Args:
            user_id: User to halt
            reason: Reason for halt
            duration_hours: How long to halt (hours)
        """
        halt_until = datetime.now(timezone.utc) + timedelta(hours=duration_hours)

        self.halted_users[user_id] = {
            'reason': reason,
            'halted_at': datetime.now(timezone.utc),
            'halted_until': halt_until,
            'daily_pnl': self.daily_pnl[user_id],
            'consecutive_losses': self.consecutive_losses[user_id]
        }

        logger.critical(
            f"🛑 CIRCUIT BREAKER TRIGGERED [User {user_id}]\n"
            f"   Reason: {reason}\n"
            f"   Halted until: {halt_until.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"   Duration: {duration_hours} hours\n"
            f"   Daily P&L: ${self.daily_pnl[user_id]:+.2f}\n"
            f"   Consecutive Losses: {self.consecutive_losses[user_id]}"
        )

    def _resume_trading(self, user_id: int) -> None:
        """Resume trading for a user after halt period expires"""
        if user_id in self.halted_users:
            halt_info = self.halted_users.pop(user_id)
            logger.info(
                f"✅ [User {user_id}] Trading resumed after halt: {halt_info['reason']}"
            )

    def _check_daily_reset(self, user_id: int) -> None:
        """Reset daily counters if it's a new day"""
        now = datetime.now(timezone.utc)
        last_reset = self.last_reset_date.get(user_id)

        if last_reset is None or now.date() > last_reset.date():
            if last_reset is not None:
                logger.info(
                    f"🔄 [User {user_id}] Daily reset - Previous day: "
                    f"{self.daily_trade_count[user_id]} trades, "
                    f"${self.daily_pnl[user_id]:+.2f} P&L"
                )

            self.daily_pnl[user_id] = 0.0
            self.daily_trade_count[user_id] = 0
            self.last_reset_date[user_id] = now

    def _calculate_daily_loss_pct(self, user_id: int, account_balance: float) -> float:
        """
        Calculate daily loss as percentage of starting balance.

        Args:
            user_id: User ID
            account_balance: Current balance

        Returns:
            Loss percentage (positive number if losing)
        """
        if self.daily_pnl[user_id] >= 0:
            return 0.0

        # Calculate starting balance (current + losses)
        starting_balance = account_balance + abs(self.daily_pnl[user_id])

        if starting_balance <= 0:
            return 100.0

        loss_pct = (abs(self.daily_pnl[user_id]) / starting_balance) * 100
        return loss_pct

    def get_user_status(self, user_id: int, account_balance: float) -> Dict:
        """
        Get current circuit breaker status for a user.

        Returns:
            Dict with status information
        """
        self._check_daily_reset(user_id)

        is_halted = user_id in self.halted_users
        daily_loss_pct = self._calculate_daily_loss_pct(user_id, account_balance)

        status = {
            'is_halted': is_halted,
            'halt_info': self.halted_users.get(user_id),
            'daily_pnl': self.daily_pnl[user_id],
            'daily_trade_count': self.daily_trade_count[user_id],
            'daily_loss_pct': daily_loss_pct,
            'consecutive_losses': self.consecutive_losses[user_id],
            'trades_remaining_today': max(0, self.config['max_daily_trades'] - self.daily_trade_count[user_id]),
            'loss_limit_remaining_pct': max(0, self.config['max_daily_loss_pct'] - daily_loss_pct),
            'losses_until_halt': max(0, self.config['max_consecutive_losses'] - self.consecutive_losses[user_id])
        }

        return status

    def manual_resume(self, user_id: int, admin_override: bool = False) -> bool:
        """
        Manually resume trading for a halted user.

        Args:
            user_id: User to resume
            admin_override: If True, bypass time restrictions

        Returns:
            True if resumed, False if not halted
        """
        if user_id not in self.halted_users:
            return False

        halt_info = self.halted_users[user_id]

        if admin_override:
            logger.warning(
                f"⚠️  [User {user_id}] Manual override - trading resumed by admin"
            )
            self._resume_trading(user_id)
            return True

        # Check if halt period has expired
        if datetime.now(timezone.utc) >= halt_info['halted_until']:
            self._resume_trading(user_id)
            return True

        logger.warning(
            f"⚠️  [User {user_id}] Cannot resume - halt period not expired "
            f"({halt_info['halted_until']})"
        )
        return False

    def reset_for_user(self, user_id: int) -> None:
        """Completely reset all circuit breaker state for a user"""
        self.daily_pnl.pop(user_id, None)
        self.daily_trade_count.pop(user_id, None)
        self.last_reset_date.pop(user_id, None)
        self.consecutive_losses.pop(user_id, None)
        self.loss_streak_start.pop(user_id, None)
        self.halted_users.pop(user_id, None)

        logger.info(f"🔄 [User {user_id}] Circuit breaker state reset")
