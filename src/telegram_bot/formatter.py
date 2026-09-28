"""
Signal message formatting.

House style shared across every Cronus Signals strategy bot (Titan Trend,
ORB, etc.) so subscribers to more than one product see a consistent look:
  - Header: "{emoji} {Strategy Name} {timeframe} Signal — {SYMBOL}"
  - Direction: BUY / SELL
  - Numeric price levels in monospace (Telegram `<code>` tags - tap to copy,
    same visual effect as ORB's Markdown backticks without switching this
    bot's parse_mode away from HTML)
  - "Stop" (not "Stop Loss"), "Session: YYYY-MM-DD" date line

Fields below the shared block are strategy-specific and intentionally NOT
forced to match other Cronus Signals bots: Titan Trend is a multi-factor
confluence strategy with a scaled exit (Take Profit 1 + 2) and an ML
confidence score, which a single fixed-target strategy like ORB doesn't have.

Example:

    🟢 Titan Trend 15m Signal — ES
    Direction: BUY
    Entry: `7834.75`
    Stop: `7829.0`
    Take Profit 1: `7843.25`
    Take Profit 2: `7849.75`
    Risk/Reward: 1:2.7
    Confidence: 87%
    Session: 2026-09-22
    Suggested size: 1 contract(s) (1% account risk)
    Reason:
    - EMA trend bullish
    - RSI recovery from oversold
    - Model prediction: P(favorable)=82%
"""
from __future__ import annotations

from src.strategy.strategy import BUY, SELL, confidence_from_score


def _round_price(value: float, tick_size: float) -> float:
    """Round a price to the instrument's tick size for a clean display value."""
    if tick_size <= 0:
        return round(value, 2)
    return round(round(value / tick_size) * tick_size, 4)


def format_signal_message(
    symbol: str,
    timeframe: str,
    action: str,
    entry: float,
    stop_loss: float,
    take_profit_1: float,
    take_profit_2: float,
    reward_to_risk_tp2: float,
    rule_score: float,
    reasons: list,
    tick_size: float = 0.25,
    contracts: int | None = None,
    strategy_name: str | None = None,
    session_date: str | None = None,
) -> str:
    """Render one signal as the HTML-formatted Telegram message text.

    `strategy_name` (e.g. "Titan Trend", from config.yaml's
    `project.strategy_name`) drives the header line. `session_date`
    (YYYY-MM-DD, typically the signal bar's date) mirrors the "Session:"
    line used by this company's other strategy bots (e.g. ORB) so message
    history reads consistently across products.
    """
    emoji = "🟢" if action == BUY else "🔴"
    confidence = confidence_from_score(rule_score)
    name = strategy_name or "Signal"

    lines = [
        f"{emoji} <b>{name} {timeframe} Signal — {symbol}</b>",
        f"Direction: {action}",
        f"Entry: <code>{_round_price(entry, tick_size)}</code>",
        f"Stop: <code>{_round_price(stop_loss, tick_size)}</code>",
        f"Take Profit 1: <code>{_round_price(take_profit_1, tick_size)}</code>",
        f"Take Profit 2: <code>{_round_price(take_profit_2, tick_size)}</code>",
        f"Risk/Reward: 1:{reward_to_risk_tp2:.1f}",
        f"Confidence: {confidence}%",
    ]
    if session_date:
        lines.append(f"Session: {session_date}")
    if contracts is not None:
        lines.append(f"Suggested size: {contracts} contract(s) (1% account risk)")
    lines.append("Reason:")
    for r in reasons:
        # Only surface the confirmatory reasons in the outward-facing message,
        # not internal veto/skip notes (those stay in logs for debugging).
        if not any(skip in r for skip in ("skipped", "Vetoed", "suppressed")):
            lines.append(f"- {r}")

    return "\n".join(lines)


def format_hold_message(symbol: str, timeframe: str, reasons: list, strategy_name: str | None = None) -> str:
    """Optional verbose/debug message for when the bot evaluates a bar and
    decides not to trade - useful for a private ops/debug channel, not the
    paying-subscriber channel."""
    name = strategy_name or "Signal"
    lines = [f"⚪ <b>{name} {timeframe} — {symbol} — HOLD</b>", "Reason:"]
    for r in reasons:
        lines.append(f"- {r}")
    return "\n".join(lines)
