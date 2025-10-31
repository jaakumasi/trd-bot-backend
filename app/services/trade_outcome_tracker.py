"""
Trade Outcome Tracker - Learning System
Tracks AI predictions vs actual outcomes to enable continuous learning and calibration.

This module provides:
1. Prediction-outcome correlation tracking
2. AI confidence calibration curves
3. Pattern-based failure detection
4. Regime-specific performance analysis
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class TradeOutcomeTracker:
    """
    Tracks trade predictions and outcomes to enable learning and prevent repeated mistakes.
    
    Key Features:
    - AI Confidence Calibration: Maps predicted confidence → actual win rates
    - Pattern Recognition: Identifies failing setup patterns (e.g., "oversold RSI in downtrends")
    - Regime Analysis: Win rates per market regime (BULL_TREND, BEAR_TREND, etc.)
    - Historical Context: Provides AI with recent failure modes to avoid
    """
    
    def __init__(self, persistence_path: Optional[str] = None):
        """
        Initialize tracker with optional persistence.
        
        Args:
            persistence_path: Path to JSON file for saving/loading trade history
        """
        self.trade_log: List[Dict] = []
        self.persistence_path = persistence_path
        
        # Performance caches (updated periodically)
        self._calibration_curve: Dict[int, float] = {}
        self._regime_performance: Dict[str, Dict] = {}
        self._failing_patterns: List[Dict] = []
        self._last_analysis_time: Optional[datetime] = None
        
        # Load existing history if available
        if persistence_path:
            self._load_from_disk()
        
        logger.info("✅ TradeOutcomeTracker initialized")
    
    def log_trade_entry(
        self,
        trade_id: str,
        user_id: int,
        signal: Dict,
        market_state: Dict,
        entry_price: float,
        stop_loss: float,
        take_profit: float
    ) -> None:
        """
        Record trade setup at entry for later outcome correlation.
        
        Args:
            trade_id: Unique trade identifier
            user_id: User ID
            signal: AI signal dict (confidence, reasoning, action)
            market_state: Market regime and confluence data
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
        """
        entry = {
            # Identifiers
            'trade_id': trade_id,
            'user_id': user_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            
            # AI Prediction
            'ai_confidence': signal.get('final_confidence', signal.get('confidence', 0)),
            'ai_signal': signal.get('signal', 'unknown'),
            'ai_reasoning': signal.get('reasoning', '')[:200],  # Truncate
            
            # Setup Characteristics
            'confluence_score': signal.get('signal_confluence', signal.get('mtf_confluence', 0)),
            'technical_score': signal.get('technical_score', 0),
            'regime': market_state.get('regime', 'UNKNOWN'),
            'trend_strength': market_state.get('trend_strength', 0),
            'trading_quality': market_state.get('trading_quality_score', 0),
            'volatility_percentile': market_state.get('volatility_percentile', 50),
            'atr_percentage': market_state.get('atr_percentage', 0),
            
            # Advanced metrics (if available)
            'momentum_persistence': market_state.get('momentum_persistence', 50),
            'order_flow_imbalance': market_state.get('order_flow_imbalance', 0),
            'mean_reversion_score': market_state.get('mean_reversion_score', 0),
            'trading_edge': market_state.get('trading_edge', 'NEUTRAL'),
            
            # Price levels
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': abs(take_profit - entry_price) / abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 0 else 0,
            
            # Outcome (to be filled later)
            'outcome': None,  # 'STOP_LOSS', 'TAKE_PROFIT', 'TIMEOUT_AUTO_CLOSE', etc.
            'exit_price': None,
            'pnl': None,
            'pnl_percentage': None,
            'duration_seconds': None,
            'exit_timestamp': None,
        }
        
        self.trade_log.append(entry)
        
        logger.debug(
            f"📝 [Tracker] Entry logged: {trade_id} | "
            f"Confidence={entry['ai_confidence']:.0f}% | "
            f"Regime={entry['regime']} | Quality={entry['trading_quality']:.0f}"
        )
        
        # Persist if configured
        if self.persistence_path and len(self.trade_log) % 5 == 0:
            self._save_to_disk()
    
    def log_trade_exit(
        self,
        trade_id: str,
        outcome: str,
        exit_price: float,
        pnl: float,
        pnl_percentage: float,
        duration_seconds: float
    ) -> None:
        """
        Update trade record with actual outcome.
        
        Args:
            trade_id: Trade identifier (matches entry)
            outcome: Exit reason ('STOP_LOSS', 'TAKE_PROFIT', etc.)
            exit_price: Actual exit price
            pnl: Profit/loss in USDT
            pnl_percentage: P&L percentage
            duration_seconds: Trade duration
        """
        # Find matching entry
        for entry in reversed(self.trade_log):  # Search backwards (recent trades)
            if entry['trade_id'] == trade_id:
                entry['outcome'] = outcome
                entry['exit_price'] = exit_price
                entry['pnl'] = pnl
                entry['pnl_percentage'] = pnl_percentage
                entry['duration_seconds'] = duration_seconds
                entry['exit_timestamp'] = datetime.now(timezone.utc).isoformat()
                
                is_win = outcome == "TAKE_PROFIT"
                logger.info(
                    f"📊 [Tracker] Exit logged: {trade_id} | "
                    f"Result={'WIN' if is_win else 'LOSS'} | "
                    f"P&L={pnl:+.2f} ({pnl_percentage:+.2f}%) | "
                    f"Confidence={entry['ai_confidence']:.0f}% | "
                    f"Duration={duration_seconds/60:.1f}min"
                )
                
                # Trigger analysis update if enough new data
                self._maybe_update_analysis()
                
                # Persist
                if self.persistence_path:
                    self._save_to_disk()
                
                return
        
        logger.warning(f"⚠️ [Tracker] Trade {trade_id} not found in entry log")
    
    def get_calibration_curve(self, user_id: Optional[int] = None, force_update: bool = False) -> Dict[int, float]:
        """
        Calculate AI confidence calibration curve.
        
        Maps: AI predicted confidence → Actual win rate
        Example: {90: 65.0, 85: 58.3, 80: 52.1, ...}
        
        This reveals if AI is overconfident (90% prediction → 60% actual win rate)
        
        Args:
            user_id: Filter to specific user's trades (None = all users)
            force_update: Recalculate even if cache is fresh
            
        Returns:
            Dict mapping confidence bins to actual win rates
        """
        if not force_update and self._calibration_curve and user_id is None:
            return self._calibration_curve
        
        # Filter trades by user_id if specified
        trades = [e for e in self.trade_log if user_id is None or e.get('user_id') == user_id]
        
        # Group trades by confidence bins (rounded to nearest 5)
        bins = defaultdict(list)
        
        for entry in trades:
            if entry['outcome'] is None:
                continue  # Skip open trades
            
            confidence = entry['ai_confidence']
            confidence_bin = int(round(confidence / 5) * 5)  # Round to 5, 10, 15, etc.
            
            is_win = entry['outcome'] == "TAKE_PROFIT"
            bins[confidence_bin].append(is_win)
        
        # Calculate actual win rates per bin
        calibration = {}
        for conf_bin, outcomes in bins.items():
            if len(outcomes) >= 3:  # Minimum 3 trades for statistical validity
                actual_win_rate = sum(outcomes) / len(outcomes) * 100
                calibration[conf_bin] = actual_win_rate
        
        if user_id is None:
            self._calibration_curve = calibration
        
        user_context = f" for user {user_id}" if user_id else ""
        logger.info(f"📈 [Tracker] Calibration curve updated{user_context}: {len(calibration)} bins")
        
        return calibration
    
    def get_regime_performance(self, user_id: Optional[int] = None, force_update: bool = False) -> Dict[str, Dict]:
        """
        Analyze performance per market regime.
        
        Args:
            user_id: Filter to specific user's trades (None = all users)
            force_update: Recalculate even if cache is fresh
        
        Returns:
            {
                'BULL_TREND': {
                    'total_trades': 15,
                    'wins': 9,
                    'losses': 6,
                    'win_rate': 60.0,
                    'avg_pnl': 1.2,
                    'avg_duration_min': 18.5
                },
                'BEAR_TREND': {...},
                ...
            }
        """
        if not force_update and self._regime_performance and user_id is None:
            return self._regime_performance
        
        # Filter trades by user_id if specified
        trades = [e for e in self.trade_log if user_id is None or e.get('user_id') == user_id]
        
        regime_stats = defaultdict(lambda: {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0,
            'total_duration': 0,
            'outcomes': []
        })
        
        for entry in trades:
            if entry['outcome'] is None:
                continue
            
            regime = entry['regime']
            is_win = entry['outcome'] == "TAKE_PROFIT"
            
            regime_stats[regime]['total_trades'] += 1
            if is_win:
                regime_stats[regime]['wins'] += 1
            else:
                regime_stats[regime]['losses'] += 1
            
            regime_stats[regime]['total_pnl'] += entry['pnl'] or 0
            regime_stats[regime]['total_duration'] += entry['duration_seconds'] or 0
            regime_stats[regime]['outcomes'].append(is_win)
        
        # Calculate derived metrics
        performance = {}
        for regime, stats in regime_stats.items():
            if stats['total_trades'] > 0:
                performance[regime] = {
                    'total_trades': stats['total_trades'],
                    'wins': stats['wins'],
                    'losses': stats['losses'],
                    'win_rate': stats['wins'] / stats['total_trades'] * 100,
                    'avg_pnl': stats['total_pnl'] / stats['total_trades'],
                    'avg_duration_min': stats['total_duration'] / stats['total_trades'] / 60,
                }
        
        if user_id is None:
            self._regime_performance = performance
        
        user_context = f" for user {user_id}" if user_id else ""
        logger.info(f"📊 [Tracker] Regime performance updated{user_context}: {len(performance)} regimes")
        
        return performance
    
    def identify_failing_patterns(self, user_id: Optional[int] = None, min_occurrences: int = 3) -> List[Dict]:
        """
        Identify recurring patterns that lead to losses.
        
        This is CRITICAL for preventing repeated mistakes.
        
        Examples of patterns detected:
        - "High confidence (>85) + RANGE_BOUND regime → 75% stop loss rate"
        - "SELL signals in BULL_TREND + ADX>40 → 80% stop loss rate"
        - "Mean reversion trades with low momentum → 70% stop loss rate"
        
        Args:
            user_id: Filter to specific user's trades (None = all users)
            min_occurrences: Minimum times pattern must occur
            
        Returns:
            List of failing patterns with descriptions and statistics
        """
        # Filter trades by user_id if specified
        trades = [e for e in self.trade_log if user_id is None or e.get('user_id') == user_id]
        
        patterns = []
        
        # Pattern 1: High confidence in wrong regimes
        high_conf_range = [e for e in trades 
                          if e['outcome'] and e['ai_confidence'] >= 85 
                          and e['regime'] == 'RANGE_BOUND']
        if len(high_conf_range) >= min_occurrences:
            losses = sum(1 for e in high_conf_range if e['outcome'] == 'STOP_LOSS')
            loss_rate = losses / len(high_conf_range) * 100
            if loss_rate > 60:
                patterns.append({
                    'pattern': 'HIGH_CONFIDENCE_IN_RANGE',
                    'description': "High AI confidence (≥85%) in RANGE_BOUND markets",
                    'occurrences': len(high_conf_range),
                    'loss_rate': loss_rate,
                    'recommendation': "Reduce confidence threshold for RANGE_BOUND or avoid entirely"
                })
        
        # Pattern 2: Counter-trend trades in strong trends
        counter_trend = [e for e in trades
                        if e['outcome'] and e['trend_strength'] > 40
                        and ((e['ai_signal'] == 'buy' and e['regime'] == 'BEAR_TREND')
                             or (e['ai_signal'] == 'sell' and e['regime'] == 'BULL_TREND'))]
        if len(counter_trend) >= min_occurrences:
            losses = sum(1 for e in counter_trend if e['outcome'] == 'STOP_LOSS')
            loss_rate = losses / len(counter_trend) * 100
            if loss_rate > 60:
                patterns.append({
                    'pattern': 'COUNTER_TREND_IN_STRONG_MARKET',
                    'description': "Trading against strong trend (ADX>40)",
                    'occurrences': len(counter_trend),
                    'loss_rate': loss_rate,
                    'recommendation': "Block counter-trend trades when ADX > 40"
                })
        
        # Pattern 3: Mean reversion with weak momentum
        mean_rev_weak = [e for e in trades
                        if e['outcome'] and e.get('mean_reversion_score', 0) > 60
                        and e.get('momentum_persistence', 50) < 40]
        if len(mean_rev_weak) >= min_occurrences:
            losses = sum(1 for e in mean_rev_weak if e['outcome'] == 'STOP_LOSS')
            loss_rate = losses / len(mean_rev_weak) * 100
            if loss_rate > 60:
                patterns.append({
                    'pattern': 'MEAN_REVERSION_WEAK_MOMENTUM',
                    'description': "Mean reversion trades with weak momentum persistence",
                    'occurrences': len(mean_rev_weak),
                    'loss_rate': loss_rate,
                    'recommendation': "Require momentum_persistence > 40 for mean reversion trades"
                })
        
        # Pattern 4: Low quality setups with high AI confidence (overconfidence)
        overconf = [e for e in trades
                   if e['outcome'] and e['ai_confidence'] >= 85
                   and e['trading_quality'] < 60]
        if len(overconf) >= min_occurrences:
            losses = sum(1 for e in overconf if e['outcome'] == 'STOP_LOSS')
            loss_rate = losses / len(overconf) * 100
            if loss_rate > 60:
                patterns.append({
                    'pattern': 'OVERCONFIDENT_LOW_QUALITY',
                    'description': "High AI confidence (≥85%) despite low trading quality (<60)",
                    'occurrences': len(overconf),
                    'loss_rate': loss_rate,
                    'recommendation': "Cap AI confidence at 75% when trading_quality < 60"
                })
        
        self._failing_patterns = patterns
        
        if patterns:
            logger.warning(f"🚨 [Tracker] Identified {len(patterns)} failing patterns:")
            for p in patterns:
                logger.warning(
                    "   ⚠️ %s: %.1f%% loss rate (%d occurrences)",
                    p['pattern'], p['loss_rate'], p['occurrences']
                )
        
        return patterns
    
    def get_historical_context_for_ai(self, current_setup: Dict, user_id: Optional[int] = None, lookback: int = 50) -> str:
        """
        Generate natural language context about recent performance to include in AI prompt.
        
        This helps AI learn from mistakes: "We've lost 5 of last 7 trades in RANGE_BOUND 
        markets, so be extra cautious with this setup."
        
        Args:
            current_setup: Current market state dict
            user_id: Filter to specific user's trades (None = all users)
            lookback: How many recent trades to analyze
            
        Returns:
            Natural language summary of relevant historical performance
        """
        # Filter trades by user_id if specified
        filtered_trades = [e for e in self.trade_log if user_id is None or e.get('user_id') == user_id]
        recent_trades = [e for e in filtered_trades[-lookback:] if e['outcome'] is not None]
        
        if len(recent_trades) < 5:
            return "Limited historical data available (< 5 completed trades)."
        
        # Overall recent performance
        total = len(recent_trades)
        wins = sum(1 for e in recent_trades if e['outcome'] == 'TAKE_PROFIT')
        win_rate = wins / total * 100
        
        context_parts = [
            f"Recent Performance (Last {total} Trades):",
            f"- Overall Win Rate: {win_rate:.1f}% ({wins} wins, {total-wins} losses)",
        ]
        
        # Regime-specific context
        current_regime = current_setup.get('regime', 'UNKNOWN')
        regime_trades = [e for e in recent_trades if e['regime'] == current_regime]
        if len(regime_trades) >= 3:
            regime_wins = sum(1 for e in regime_trades if e['outcome'] == 'TAKE_PROFIT')
            regime_win_rate = regime_wins / len(regime_trades) * 100
            context_parts.append(
                f"- {current_regime} Performance: {regime_win_rate:.1f}% win rate "
                f"({len(regime_trades)} trades)"
            )
            
            if regime_win_rate < 40:
                context_parts.append(
                    f"  ⚠️ WARNING: Poor performance in {current_regime} recently. "
                    f"Reduce confidence by 10-15 points."
                )
        
        # Failing patterns warning
        patterns = self.identify_failing_patterns(user_id=user_id, min_occurrences=3)
        if patterns:
            context_parts.append("\nIdentified Failing Patterns:")
            for pattern in patterns[:2]:  # Top 2 patterns
                context_parts.append(
                    f"- {pattern['description']}: {pattern['loss_rate']:.1f}% loss rate"
                )
                context_parts.append(f"  → {pattern['recommendation']}")
        
        # Calibration insight
        calibration = self.get_calibration_curve(user_id=user_id)
        if calibration:
            ai_conf = current_setup.get('ai_confidence', 0)
            conf_bin = int(round(ai_conf / 5) * 5)
            if conf_bin in calibration:
                actual_rate = calibration[conf_bin]
                context_parts.append(
                    f"\nCalibration Note: Your {conf_bin}% confidence predictions "
                    f"historically achieve {actual_rate:.1f}% win rate."
                )
                if abs(conf_bin - actual_rate) > 15:
                    context_parts.append(
                        f"  ⚠️ CALIBRATION ISSUE: {conf_bin-actual_rate:+.1f}% overconfidence detected."
                    )
        
        return "\n".join(context_parts)
    
    def _maybe_update_analysis(self) -> None:
        """Trigger analysis update if enough new data accumulated."""
        now = datetime.now(timezone.utc)
        
        # Update every 10 trades or every hour
        if self._last_analysis_time is None or \
           (now - self._last_analysis_time).seconds > 3600 or \
           len([e for e in self.trade_log if e['outcome']]) % 10 == 0:
            
            self.get_calibration_curve(force_update=True)
            self.get_regime_performance(force_update=True)
            self.identify_failing_patterns()
            self._last_analysis_time = now
            
            logger.info("📊 [Tracker] Analysis updated")
    
    def _save_to_disk(self) -> None:
        """Persist trade log to disk."""
        if not self.persistence_path:
            return
        
        try:
            path = Path(self.persistence_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(self.trade_log, f, indent=2)
            
            logger.debug(f"💾 [Tracker] Saved {len(self.trade_log)} trades to {path}")
        except Exception as e:
            logger.error(f"❌ [Tracker] Failed to save to disk: {e}")
    
    def _load_from_disk(self) -> None:
        """Load trade log from disk."""
        if not self.persistence_path:
            return
        
        try:
            path = Path(self.persistence_path)
            if path.exists():
                with open(path, 'r') as f:
                    self.trade_log = json.load(f)
                
                logger.info(f"📂 [Tracker] Loaded {len(self.trade_log)} trades from {path}")
        except Exception as e:
            logger.error(f"❌ [Tracker] Failed to load from disk: {e}")
    
    def get_statistics_summary(self) -> Dict:
        """
        Get comprehensive statistics summary.
        
        Returns:
            Dict with calibration, regime performance, patterns, and overall stats
        """
        completed = [e for e in self.trade_log if e['outcome'] is not None]
        
        if not completed:
            return {'error': 'No completed trades yet'}
        
        wins = sum(1 for e in completed if e['outcome'] == 'TAKE_PROFIT')
        
        return {
            'total_tracked': len(self.trade_log),
            'completed_trades': len(completed),
            'open_trades': len(self.trade_log) - len(completed),
            'overall_win_rate': wins / len(completed) * 100,
            'total_pnl': sum(e['pnl'] or 0 for e in completed),
            'avg_pnl_per_trade': sum(e['pnl'] or 0 for e in completed) / len(completed),
            'calibration_curve': self.get_calibration_curve(),
            'regime_performance': self.get_regime_performance(),
            'failing_patterns': self.identify_failing_patterns(),
        }
