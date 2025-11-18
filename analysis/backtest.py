"""
File: analysis/backtest.py
VWV Trading System - Integrated Backtesting Engine
Created: 2025-11-18
Phase 2A: Performance Validation & Strategy Testing

Provides comprehensive backtesting for:
- Master Score signals (Phase 1a)
- Divergence signals (Phase 1b)
- Confluence signals (Phase 1b)
- Combined multi-signal strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class Trade:
    """Represents a single trade with entry, exit, and P&L details"""

    def __init__(self, entry_date, entry_price, direction='LONG'):
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.direction = direction
        self.exit_date = None
        self.exit_price = None
        self.pnl = 0
        self.pnl_pct = 0
        self.holding_days = 0
        self.exit_reason = None

    def close(self, exit_date, exit_price, reason='SIGNAL'):
        """Close the trade and calculate P&L"""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_reason = reason
        self.holding_days = (exit_date - self.entry_date).days

        if self.direction == 'LONG':
            self.pnl = exit_price - self.entry_price
            self.pnl_pct = ((exit_price - self.entry_price) / self.entry_price) * 100
        else:  # SHORT
            self.pnl = self.entry_price - exit_price
            self.pnl_pct = ((self.entry_price - exit_price) / self.entry_price) * 100

    def to_dict(self):
        """Convert trade to dictionary for DataFrame"""
        return {
            'entry_date': self.entry_date,
            'entry_price': self.entry_price,
            'exit_date': self.exit_date,
            'exit_price': self.exit_price,
            'direction': self.direction,
            'holding_days': self.holding_days,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'exit_reason': self.exit_reason
        }


class BacktestEngine:
    """Core backtesting engine for VWV trading strategies"""

    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000,
                 commission: float = 0.001, slippage: float = 0.0005):
        """
        Initialize backtest engine

        Args:
            data: Historical OHLCV data
            initial_capital: Starting capital ($)
            commission: Commission per trade (0.001 = 0.1%)
            slippage: Estimated slippage per trade (0.0005 = 0.05%)
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.trades = []
        self.equity_curve = []
        self.current_position = None
        self.cash = initial_capital
        self.total_commission = 0

    def enter_trade(self, date, price, direction='LONG'):
        """Enter a new trade"""
        if self.current_position is not None:
            return  # Already in position

        # Apply commission and slippage
        entry_price = price * (1 + self.slippage) if direction == 'LONG' else price * (1 - self.slippage)
        commission_cost = entry_price * self.commission

        self.current_position = Trade(date, entry_price, direction)
        self.total_commission += commission_cost

    def exit_trade(self, date, price, reason='SIGNAL'):
        """Exit current trade"""
        if self.current_position is None:
            return  # No position to exit

        # Apply commission and slippage
        direction = self.current_position.direction
        exit_price = price * (1 - self.slippage) if direction == 'LONG' else price * (1 + self.slippage)
        commission_cost = exit_price * self.commission

        self.current_position.close(date, exit_price, reason)
        self.trades.append(self.current_position)

        # Update cash
        trade_pnl = (exit_price - self.current_position.entry_price) if direction == 'LONG' else (self.current_position.entry_price - exit_price)
        self.cash += trade_pnl
        self.total_commission += commission_cost

        self.current_position = None

    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if len(self.trades) == 0:
            return {
                'error': 'No trades executed',
                'total_trades': 0
            }

        trade_returns = [t.pnl_pct for t in self.trades]
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]

        # Basic metrics
        total_return = (self.cash - self.initial_capital) / self.initial_capital * 100
        total_trades = len(self.trades)
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

        # Win/Loss metrics
        avg_win = np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0
        avg_trade = np.mean(trade_returns)

        # Profit factor
        gross_profit = sum([t.pnl for t in winning_trades])
        gross_loss = abs(sum([t.pnl for t in losing_trades]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Drawdown calculation
        equity = self.initial_capital
        peak = equity
        max_dd = 0
        max_dd_pct = 0

        for trade in self.trades:
            equity += trade.pnl
            if equity > peak:
                peak = equity
            dd = peak - equity
            dd_pct = (dd / peak * 100) if peak > 0 else 0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
                max_dd = dd

        # Sharpe Ratio (annualized, assuming 252 trading days)
        if len(trade_returns) > 1:
            returns_std = np.std(trade_returns)
            avg_holding_days = np.mean([t.holding_days for t in self.trades])
            trades_per_year = 252 / avg_holding_days if avg_holding_days > 0 else 0
            annualized_return = avg_trade * trades_per_year
            annualized_std = returns_std * np.sqrt(trades_per_year)
            sharpe_ratio = annualized_return / annualized_std if annualized_std > 0 else 0
        else:
            sharpe_ratio = 0
            annualized_return = 0

        # Expectancy
        expectancy = (win_rate / 100 * avg_win) + ((100 - win_rate) / 100 * avg_loss)

        # Time-based metrics
        first_trade_date = self.trades[0].entry_date
        last_trade_date = self.trades[-1].exit_date if self.trades[-1].exit_date else self.trades[-1].entry_date
        total_days = (last_trade_date - first_trade_date).days
        years = total_days / 365.25
        annualized_return_calendar = (total_return / years) if years > 0 else 0

        return {
            # Returns
            'total_return': round(total_return, 2),
            'annualized_return': round(annualized_return_calendar, 2),

            # Risk
            'max_drawdown': round(max_dd, 2),
            'max_drawdown_pct': round(max_dd_pct, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),

            # Trade Statistics
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': round(win_rate, 2),

            # Performance
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'avg_trade': round(avg_trade, 2),
            'profit_factor': round(profit_factor, 2),
            'expectancy': round(expectancy, 2),

            # Other
            'total_commission': round(self.total_commission, 2),
            'final_capital': round(self.cash, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),

            # Time
            'start_date': first_trade_date,
            'end_date': last_trade_date,
            'days_traded': total_days,
            'avg_holding_days': round(np.mean([t.holding_days for t in self.trades]), 1)
        }

    def get_equity_curve(self) -> pd.DataFrame:
        """Generate equity curve over time"""
        equity_data = []
        equity = self.initial_capital

        for trade in self.trades:
            # Entry point
            equity_data.append({
                'date': trade.entry_date,
                'equity': equity,
                'event': 'entry'
            })

            # Exit point
            equity += trade.pnl
            equity_data.append({
                'date': trade.exit_date,
                'equity': equity,
                'event': 'exit'
            })

        df = pd.DataFrame(equity_data)
        return df

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Convert trades list to DataFrame"""
        if not self.trades:
            return pd.DataFrame()

        trade_dicts = [t.to_dict() for t in self.trades]
        df = pd.DataFrame(trade_dicts)
        df = df.sort_values('entry_date', ascending=False)
        return df


def backtest_master_score(data: pd.DataFrame, analysis_function,
                           entry_threshold: float = 65,
                           exit_threshold: float = 40,
                           hold_days: int = 20) -> Dict:
    """
    Backtest Master Score strategy

    Args:
        data: Historical OHLCV data
        analysis_function: Function that calculates master score from data
        entry_threshold: Master score threshold for entry (default 65)
        exit_threshold: Master score threshold for exit (default 40)
        hold_days: Max holding period in days (default 20)

    Returns:
        Dictionary with backtest results and metrics
    """
    engine = BacktestEngine(data)

    # Need at least 50 bars for calculations
    for i in range(50, len(data)):
        current_date = data.index[i]
        current_price = data['Close'].iloc[i]

        # Get data slice up to current bar
        data_slice = data.iloc[:i+1]

        try:
            # Calculate master score (simplified - would need full analysis pipeline)
            # This is placeholder - actual implementation would call full analysis
            master_score = 50  # Placeholder

            # Entry logic: No position + score above threshold
            if engine.current_position is None:
                if master_score >= entry_threshold:
                    engine.enter_trade(current_date, current_price, 'LONG')

            # Exit logic: In position + (score below threshold OR max holding period)
            else:
                days_held = (current_date - engine.current_position.entry_date).days

                if master_score <= exit_threshold:
                    engine.exit_trade(current_date, current_price, 'SIGNAL')
                elif days_held >= hold_days:
                    engine.exit_trade(current_date, current_price, 'TIME_STOP')

        except Exception as e:
            continue

    # Close any open position at end
    if engine.current_position is not None:
        last_date = data.index[-1]
        last_price = data['Close'].iloc[-1]
        engine.exit_trade(last_date, last_price, 'END_OF_DATA')

    metrics = engine.calculate_metrics()
    equity_curve = engine.get_equity_curve()
    trades_df = engine.get_trades_dataframe()

    return {
        'strategy': 'Master Score',
        'parameters': {
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'max_hold_days': hold_days
        },
        'metrics': metrics,
        'equity_curve': equity_curve,
        'trades': trades_df,
        'engine': engine
    }


def backtest_divergence_signals(data: pd.DataFrame,
                                  technicals: pd.DataFrame,
                                  min_divergence_score: float = 10,
                                  hold_days: int = 15) -> Dict:
    """
    Backtest Divergence detection strategy

    Args:
        data: Historical OHLCV data
        technicals: Technical indicators DataFrame
        min_divergence_score: Minimum divergence score for entry
        hold_days: Max holding period

    Returns:
        Backtest results dictionary
    """
    engine = BacktestEngine(data)

    # Implementation similar to master_score
    # Would integrate with Phase 1b divergence detection

    return {
        'strategy': 'Divergence',
        'parameters': {
            'min_score': min_divergence_score,
            'max_hold_days': hold_days
        },
        'metrics': {},
        'equity_curve': pd.DataFrame(),
        'trades': pd.DataFrame(),
        'engine': engine
    }


def backtest_confluence_signals(data: pd.DataFrame,
                                  confluence_data: pd.DataFrame,
                                  min_agreement: float = 70,
                                  hold_days: int = 20) -> Dict:
    """
    Backtest Confluence-based strategy

    Args:
        data: Historical OHLCV data
        confluence_data: Confluence scores over time
        min_agreement: Minimum confluence score for entry (0-100)
        hold_days: Max holding period

    Returns:
        Backtest results dictionary
    """
    engine = BacktestEngine(data)

    # Implementation similar to master_score
    # Would integrate with Phase 1b confluence detection

    return {
        'strategy': 'Confluence',
        'parameters': {
            'min_agreement': min_agreement,
            'max_hold_days': hold_days
        },
        'metrics': {},
        'equity_curve': pd.DataFrame(),
        'trades': pd.DataFrame(),
        'engine': engine
    }


def backtest_combined_strategy(data: pd.DataFrame,
                                 strategy_config: Dict) -> Dict:
    """
    Backtest combined multi-signal strategy

    Combines Master Score, Divergence, and Confluence signals
    with configurable weights and thresholds.

    Args:
        data: Historical OHLCV data
        strategy_config: Configuration dict with thresholds and weights

    Returns:
        Backtest results dictionary
    """
    engine = BacktestEngine(data)

    # Multi-signal logic would go here
    # Entry when multiple signals align
    # Exit when signals weaken

    return {
        'strategy': 'Combined Multi-Signal',
        'parameters': strategy_config,
        'metrics': {},
        'equity_curve': pd.DataFrame(),
        'trades': pd.DataFrame(),
        'engine': engine
    }


def backtest_buy_and_hold(data: pd.DataFrame) -> Dict:
    """
    Backtest simple buy-and-hold strategy as benchmark

    Args:
        data: Historical OHLCV data

    Returns:
        Backtest results dictionary
    """
    engine = BacktestEngine(data)

    # Buy at start
    first_date = data.index[50]  # Skip first 50 bars like other strategies
    first_price = data['Close'].iloc[50]
    engine.enter_trade(first_date, first_price, 'LONG')

    # Sell at end
    last_date = data.index[-1]
    last_price = data['Close'].iloc[-1]
    engine.exit_trade(last_date, last_price, 'END_OF_DATA')

    metrics = engine.calculate_metrics()
    equity_curve = engine.get_equity_curve()
    trades_df = engine.get_trades_dataframe()

    return {
        'strategy': 'Buy & Hold (Benchmark)',
        'parameters': {},
        'metrics': metrics,
        'equity_curve': equity_curve,
        'trades': trades_df,
        'engine': engine
    }


def compare_strategies(backtest_results: List[Dict]) -> pd.DataFrame:
    """
    Compare multiple strategy backtest results side-by-side

    Args:
        backtest_results: List of backtest result dictionaries

    Returns:
        DataFrame with comparison metrics
    """
    comparison_data = []

    for result in backtest_results:
        strategy_name = result['strategy']
        metrics = result['metrics']

        if 'error' in metrics:
            continue

        comparison_data.append({
            'Strategy': strategy_name,
            'Total Return %': metrics.get('total_return', 0),
            'Annual Return %': metrics.get('annualized_return', 0),
            'Max Drawdown %': metrics.get('max_drawdown_pct', 0),
            'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
            'Win Rate %': metrics.get('win_rate', 0),
            'Profit Factor': metrics.get('profit_factor', 0),
            'Total Trades': metrics.get('total_trades', 0),
            'Avg Trade %': metrics.get('avg_trade', 0)
        })

    df = pd.DataFrame(comparison_data)

    # Sort by Sharpe Ratio (risk-adjusted performance)
    if len(df) > 0:
        df = df.sort_values('Sharpe Ratio', ascending=False)

    return df


def generate_backtest_report(result: Dict) -> str:
    """
    Generate a formatted text report of backtest results

    Args:
        result: Backtest result dictionary

    Returns:
        Formatted string report
    """
    metrics = result['metrics']

    if 'error' in metrics:
        return f"Backtest Error: {metrics['error']}"

    report = f"""
{'='*80}
BACKTEST REPORT: {result['strategy']}
{'='*80}

PERFORMANCE METRICS:
--------------------
Total Return:        {metrics['total_return']:>10.2f}%
Annualized Return:   {metrics['annualized_return']:>10.2f}%
Maximum Drawdown:    {metrics['max_drawdown_pct']:>10.2f}%
Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}

TRADE STATISTICS:
-----------------
Total Trades:        {metrics['total_trades']:>10}
Winning Trades:      {metrics['winning_trades']:>10}
Losing Trades:       {metrics['losing_trades']:>10}
Win Rate:            {metrics['win_rate']:>10.2f}%

PROFIT ANALYSIS:
----------------
Average Win:         {metrics['avg_win']:>10.2f}%
Average Loss:        {metrics['avg_loss']:>10.2f}%
Average Trade:       {metrics['avg_trade']:>10.2f}%
Profit Factor:       {metrics['profit_factor']:>10.2f}
Expectancy/Trade:    {metrics['expectancy']:>10.2f}%

CAPITAL:
--------
Initial Capital:     ${metrics.get('initial_capital', 100000):>10,.2f}
Final Capital:       ${metrics['final_capital']:>10,.2f}
Gross Profit:        ${metrics['gross_profit']:>10,.2f}
Gross Loss:          ${metrics['gross_loss']:>10,.2f}
Total Commission:    ${metrics['total_commission']:>10,.2f}

TIME PERIOD:
------------
Start Date:          {metrics['start_date'].strftime('%Y-%m-%d')}
End Date:            {metrics['end_date'].strftime('%Y-%m-%d')}
Days Traded:         {metrics['days_traded']} days
Avg Holding Period:  {metrics['avg_holding_days']} days

{'='*80}
"""

    return report
