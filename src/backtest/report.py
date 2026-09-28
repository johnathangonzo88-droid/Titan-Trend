"""
Backtest performance charts and a Markdown report template.

Kept dependency-light (matplotlib only, no seaborn/plotly) since those are
already installed in most Python environments and the brief just asks for
"performance charts", not a specific library.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering - no display needed
import matplotlib.pyplot as plt
import pandas as pd

from src.backtest.metrics import summarize


def plot_equity_curve(equity: pd.Series, symbol: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(equity.index, equity.values, linewidth=1.2, color="#1f6feb")
    ax.set_title(f"{symbol} — Equity Curve")
    ax.set_ylabel("Account Equity ($)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_drawdown(equity: pd.Series, symbol: str, out_path: Path) -> None:
    running_max = equity.cummax()
    drawdown_pct = (equity - running_max) / running_max * 100.0
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.fill_between(drawdown_pct.index, drawdown_pct.values, 0, color="#da3633", alpha=0.5)
    ax.set_title(f"{symbol} — Drawdown (%)")
    ax.set_ylabel("Drawdown %")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_trade_pnl_distribution(trades: pd.DataFrame, symbol: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(trades) > 0:
        colors = ["#2ea043" if p > 0 else "#da3633" for p in trades["pnl"]]
        ax.bar(range(len(trades)), trades["pnl"].values, color=colors, width=1.0)
    ax.set_title(f"{symbol} — Per-Trade P&L")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("P&L ($)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def generate_backtest_report(
    symbol: str, trades: pd.DataFrame, equity: pd.Series, report_dir: Path, extra_notes: list | None = None
) -> dict:
    """Compute metrics, render charts, and write a Markdown report for one
    symbol's backtest. Returns the metrics dict.
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics = summarize(trades, equity)

    plot_equity_curve(equity, symbol, report_dir / f"{symbol}_equity_curve.png")
    plot_drawdown(equity, symbol, report_dir / f"{symbol}_drawdown.png")
    plot_trade_pnl_distribution(trades, symbol, report_dir / f"{symbol}_trade_pnl.png")

    lines = [
        f"# Backtest Report — {symbol}",
        "",
        f"- **Total trades:** {metrics['n_trades']}",
        f"- **Win rate:** {metrics['win_rate']:.1%}",
        f"- **Profit factor:** {metrics['profit_factor']:.2f}",
        f"- **Sharpe ratio (annualized):** {metrics['sharpe_ratio']:.2f}",
        f"- **Sortino ratio (annualized):** {metrics['sortino_ratio']:.2f}",
        f"- **Max drawdown:** {metrics['max_drawdown_pct']:.2%} (${metrics['max_drawdown_dollars']:,.2f})",
        f"- **Total return:** {metrics['total_return_pct']:.2%}",
        f"- **Average trade return:** ${metrics['avg_trade_return_dollars']:,.2f}",
        f"- **Final equity:** ${metrics['final_equity']:,.2f}",
        "",
        "## Charts",
        f"![Equity Curve]({symbol}_equity_curve.png)",
        f"![Drawdown]({symbol}_drawdown.png)",
        f"![Trade P&L]({symbol}_trade_pnl.png)",
        "",
    ]
    if extra_notes:
        lines.append("## Notes")
        for n in extra_notes:
            lines.append(f"- {n}")
        lines.append("")

    if len(trades) > 0:
        lines.append("## Exit Reason Breakdown")
        lines.append("")
        counts = trades["exit_reason"].value_counts()
        for reason, count in counts.items():
            lines.append(f"- {reason}: {count}")

    report_path = report_dir / f"{symbol}_backtest_report.md"
    report_path.write_text("\n".join(lines))

    trades.to_csv(report_dir / f"{symbol}_trades.csv", index=False)

    return metrics
