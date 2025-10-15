# AI Improvement Strategy: Leveraging Historical Data
**Date:** October 15, 2025

---

## 1. System Analysis Summary

- **Trade Flow**: The system correctly processes one trade per user at a time. It opens a position, monitors it, and closes it upon hitting a Stop-Loss (SL) or Take-Profit (TP), persisting all results to the database.
- **Data Storage**: The `trades` table contains a complete and accurate history of every trade's lifecycle, including entry, exit, P&L, duration, and exit reason (`TAKE_PROFIT` or `STOP_LOSS`). This data is the foundation for our strategy.
- **The Gap**: The AI Analyzer currently operates in isolation, using only real-time market data. It has no knowledge of the user's past performance, which is a major blind spot.

---

## 2. Re-evaluated Strategy Recommendation

The previous strategies are sound but can be simplified and prioritized for maximum impact with minimum effort. We will adopt a two-phased approach focusing on **Personalized Context** and **Adaptive Risk**.

### **Phase 1: Personalized AI Context (Highest Priority)**

**Concept:** Empower the AI with knowledge of the user's recent performance. The AI should know if the user is on a winning or losing streak to adjust its confidence accordingly.

**Implementation:**

1.  **Create a Data-Enrichment Method:**
    In `ai_analyzer.py`, create a new method `get_user_trade_history_context(db, user_id)`. This function will:
    - Query the `trades` table for the user's last **10-20 closed trades**.
    - Calculate key metrics:
        - Win Rate (%)
        - Average P&L per trade
        - Count of wins vs. losses
        - Recent trend (e.g., "Losing: 3 of the last 5 trades were losses").
    - Return a structured dictionary containing this summary.

2.  **Enhance the AI Prompt:**
    In `trading_bot.py`, before calling the AI, fetch this historical context. Then, inject it directly into the Gemini prompt in `ai_analyzer.py`.

    **New Prompt Section:**
    ```
    User's Recent Performance (Last 15 Trades):
    - Win Rate: 60%
    - Average P&L: +$5.50
    - Recent Trend: Winning (4 of last 5 trades were profitable)

    Instruction:
    - If the user is on a losing streak, be more conservative and only recommend a 'buy' or 'sell' on very high-confidence signals (>85%).
    - If the user is on a winning streak, you can have slightly more confidence in your analysis.
    - Always consider this historical context when determining your final 'confidence' score.
    ```

**Benefits:**
- **Quick to Implement:** Leverages existing data with minimal code changes.
- **High Impact:** Immediately makes the AI's confidence score more relevant and personalized.
- **Reduces Risk:** Automatically tightens criteria for users on a losing streak, preventing further losses.

---

### **Phase 2: Adaptive Risk Management (Medium Priority)**

**Concept:** Use historical data to set smarter, data-driven Stop-Loss (SL) and Take-Profit (TP) targets, rather than relying on fixed percentages.

**Implementation:**

1.  **Create a Risk-Adjustment Method:**
    In `risk_manager.py`, create a new method `get_adaptive_exit_levels(db, user_id)`. This function will:
    - Query the user's historical `trades`.
    - Analyze **winning trades**: Calculate the average `profit_loss_percentage`. The new TP could be set to 80% of this average to secure profits reliably.
    - Analyze **losing trades**: Calculate the average `profit_loss_percentage` for trades that hit the `STOP_LOSS`. The new SL could be set slightly wider (e.g., 120%) of this average to avoid premature exits on normal market volatility.
    - Return a dictionary with `adaptive_sl_pct` and `adaptive_tp_pct`.

2.  **Integrate into Trading Bot:**
    In `trading_bot.py`, when a new trade is validated, override the default or AI-suggested SL/TP percentages with the new adaptive values from the `RiskManager`.

**Benefits:**
- **Self-Optimizing:** The system learns from the user's actual trading patterns.
- **Reduces Premature Exits:** Stop-loss levels are based on observed volatility, not arbitrary numbers.
- **Maximizes Profit:** Take-profit levels are based on previously successful exit points.

---

## 3. Implementation Roadmap

1.  **Implement Phase 1:** Focus entirely on adding the user's historical context to the AI prompt. This is the quickest win and provides the most value upfront.
2.  **Monitor and Validate:** After deploying Phase 1, monitor the AI's performance. Does the win rate improve? Does it trade less frequently for losing users?
3.  **Implement Phase 2:** Once Phase 1 is validated, proceed with implementing adaptive SL/TP levels.
4.  **Future Work (Post-Phase 2):** Consider adding `market_condition` (e.g., 'trending', 'ranging') as a tag to each trade to analyze performance under different market structures. This was a good idea from the original document but should only be tackled after the first two phases are complete.
