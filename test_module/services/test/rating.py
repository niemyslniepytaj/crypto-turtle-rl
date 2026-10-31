import math
import empyrical as ep
import pandas as pd

class Rating:
    """Obliczanie metryk oceny strategii (PnL, drawdown, Sharpe, excess)."""
    def __init__(self):
        self.portfolio_values = {}

    @staticmethod
    def _finite(x, default=0.0):
        try:
            v = float(x)
        except (TypeError, ValueError):
            return default
        return v if math.isfinite(v) else default

    def rate(self, portfolio_values, transactions, closed_trades=None, holdings_history=None,
             balance_start=1000.0, periods_per_year=525_600):
        closed_trades = closed_trades or []
        holdings_history = holdings_history or []
        returns = pd.Series(portfolio_values).pct_change().dropna()

        # płaska krzywa (0 tx / cash) → empyrical daje NaN na Sharpe/Sortino/CAGR
        metrics = {
            "total_return": self._finite(ep.cum_returns_final(returns) * 100, 0.0),
            "cagr": self._finite(ep.cagr(returns, annualization=periods_per_year) * 100, 0.0),
            "sharpe_ratio": self._finite(ep.sharpe_ratio(returns, annualization=periods_per_year), 0.0),
            "sortino_ratio": self._finite(ep.sortino_ratio(returns, annualization=periods_per_year), 0.0),
            "max_drawdown": self._finite(ep.max_drawdown(returns) * 100, 0.0),
            "calmar_ratio": self._finite(ep.calmar_ratio(returns, annualization=periods_per_year), 0.0),
            "growth_since_start": ((portfolio_values[-1] / portfolio_values[0]) - 1) * 100,
            "max_portfolio_growth": ((max(portfolio_values) / portfolio_values[0]) - 1) * 100,
            "max_portfolio_drop": ((min(portfolio_values) / portfolio_values[0]) - 1) * 100,
            "transaction_count": transactions,
            "final_portfolio_value": portfolio_values[-1]
        }

        ticks_per_day = periods_per_year / 365.0
        n_ticks = max(len(portfolio_values), 1)
        days = n_ticks / ticks_per_day
        metrics["avg_transactions_per_day"] = transactions / days if days > 0 else 0.0

        wins = [t["pnl_usd"] for t in closed_trades if t.get("pnl_usd", 0) > 0]
        losses = [t["pnl_usd"] for t in closed_trades if t.get("pnl_usd", 0) < 0]
        n_closed = len(closed_trades)

        if n_closed == 0:
            metrics["profit_factor"] = None
            metrics["win_rate"] = None
            metrics["avg_win"] = None
            metrics["avg_loss"] = None
        else:
            sum_wins = sum(wins)
            sum_losses = abs(sum(losses))
            metrics["profit_factor"] = (sum_wins / sum_losses) if sum_losses > 0 else None
            metrics["win_rate"] = len(wins) / n_closed
            metrics["avg_win"] = (sum_wins / len(wins)) if wins else None
            metrics["avg_loss"] = (sum(losses) / len(losses)) if losses else None

        if balance_start and balance_start > 0 and n_closed > 0:
            notional = sum(
                t["qty"] * t["sell_price"] + t["qty"] * t["buy_price"]
                for t in closed_trades
            )
            metrics["turnover"] = notional / balance_start
        else:
            metrics["turnover"] = 0.0 if n_closed == 0 else None

        if holdings_history:
            cash_ticks = sum(1 for h in holdings_history if sum(h) <= 1e-9)
            metrics["time_in_cash"] = cash_ticks / len(holdings_history)
        else:
            metrics["time_in_cash"] = None

        metrics["closed_trades_count"] = n_closed

        for key, value in metrics.items():
            if isinstance(value, (float, int)) and value is not None:
                metrics[key] = round(value, 4)

        metrics["score"] = self.compute_score(metrics)
        return metrics

    def compute_score(self, metrics):
        sharpe = metrics.get("sharpe_ratio") or 0
        sortino = metrics.get("sortino_ratio") or 0
        cagr = metrics.get("cagr") or 0
        max_dd = metrics.get("max_drawdown") or 0

        sharpe_score = min(max(sharpe / 2.0, 0), 1)
        sortino_score = min(max(sortino / 3.0, 0), 1)
        cagr_score = min(max((cagr / 100) * 5, 0), 1)
        dd_score = 1 - min(abs(max_dd / 100) / 0.5, 1)

        final_score = (
            0.4 * sharpe_score +
            0.3 * sortino_score +
            0.2 * cagr_score +
            0.1 * dd_score
        )
        return round(final_score * 100, 2)
