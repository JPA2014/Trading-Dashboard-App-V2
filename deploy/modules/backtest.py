"""
backtest.py - Backtesting engine for trading dashboard.
Tests options trading strategies against historical OHLCV data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats

from .data import get_ohlcv, get_options_chain
from .indicators import calculate_all_indicators


def get_historical_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for backtesting.
    
    Args:
        ticker: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        stock_data = get_ohlcv(ticker, period="max", force_refresh=True)
        
        # Filter by date range
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        stock_data = stock_data[(stock_data.index >= start) & (stock_data.index <= end)]
        
        return stock_data
    except Exception as e:
        raise ValueError(f"Could not fetch historical data: {e}")


def generate_signals(df: pd.DataFrame, strategy: str = "standard") -> pd.DataFrame:
    """
    Generate trading signals based on technical indicators.
    
    Args:
        df: DataFrame with OHLCV data
        strategy: Signal generation strategy ("standard", "aggressive", "conservative")
    
    Returns:
        DataFrame with signal columns added
    """
    indicators = calculate_all_indicators(df)
    
    df = df.copy()
    df["sma_20"] = indicators["sma_20"]
    df["sma_50"] = indicators["sma_50"]
    df["ema_9"] = indicators["ema_9"]
    df["rsi"] = indicators["rsi"]
    df["macd"] = indicators["macd"]
    df["macd_signal"] = indicators["macd_signal"]
    df["macd_histogram"] = indicators["macd_histogram"]
    
    # Generate signals based on strategy
    if strategy == "standard":
        # Standard: price above 50 SMA, RSI 40-60, MACD positive
        df["signal"] = 0
        df.loc[
            (df["Close"] > df["sma_50"]) &
            (df["rsi"] >= 40) & (df["rsi"] <= 60) &
            (df["macd_histogram"] > 0),
            "signal"
        ] = 1
    elif strategy == "aggressive":
        # Aggressive: price above 20 SMA, RSI 30-70, MACD crossing
        df["signal"] = 0
        df.loc[
            (df["Close"] > df["sma_20"]) &
            (df["rsi"] >= 30) & (df["rsi"] <= 70) &
            (df["macd"] > df["macd_signal"]),
            "signal"
        ] = 1
    elif strategy == "conservative":
        # Conservative: price above 200 SMA, RSI 45-55, strong MACD
        df["signal"] = 0
        df.loc[
            (df["Close"] > df["sma_50"]) &
            (df["rsi"] >= 45) & (df["rsi"] <= 55) &
            (df["macd_histogram"] > 0) &
            (df["macd_histogram"] > df["macd_histogram"].shift(1)),
            "signal"
        ] = 1
    
    return df


def simulate_option_trade(entry_price: float, exit_price: float,
                          quantity: int = 1, option_type: str = "call",
                          is_long: bool = True) -> Dict:
    """
    Simulate an options trade and calculate P&L.
    
    Args:
        entry_price: Entry price per contract
        exit_price: Exit price per contract
        quantity: Number of contracts
        option_type: "call" or "put"
        is_long: True for long position, False for short
    
    Returns:
        Dictionary with trade results
    """
    multiplier = 100  # Options contracts are 100 shares
    
    if is_long:
        pnl = (exit_price - entry_price) * quantity * multiplier
        cost = entry_price * quantity * multiplier
    else:
        pnl = (entry_price - exit_price) * quantity * multiplier
        cost = 0  # Short positions have no upfront cost (but unlimited risk)
    
    return {
        "entry_price": entry_price,
        "exit_price": exit_price,
        "quantity": quantity,
        "option_type": option_type,
        "is_long": is_long,
        "cost": cost,
        "pnl": pnl,
        "return_pct": (pnl / cost * 100) if cost > 0 else 0
    }


def backtest_strategy(ticker: str, start_date: str, end_date: str,
                     strategy: str = "standard",
                     option_type: str = "call",
                     holding_days: int = 5,
                     initial_capital: float = 10000) -> Dict:
    """
    Backtest an options trading strategy.
    
    Args:
        ticker: Stock symbol
        start_date: Backtest start date (YYYY-MM-DD)
        end_date: Backtest end date (YYYY-MM-DD)
        strategy: Signal strategy ("standard", "aggressive", "conservative")
        option_type: "call" or "put"
        holding_days: Days to hold each position
        initial_capital: Starting capital
    
    Returns:
        Dictionary with backtest results and metrics
    """
    try:
        # Get historical data
        df = get_historical_data(ticker, start_date, end_date)
        
        if df.empty or len(df) < 100:
            return {"error": "Insufficient historical data"}
        
        # Generate signals
        df = generate_signals(df, strategy)
        
        # Find signal dates
        signal_dates = df[df["signal"] == 1].index
        
        if len(signal_dates) == 0:
            return {
                "error": "No trading signals generated",
                "trades": [],
                "metrics": {}
            }
        
        trades = []
        capital = initial_capital
        capital_history = [initial_capital]
        dates_history = [df.index[0]]
        
        for i, signal_date in enumerate(signal_dates):
            if i >= len(signal_dates) - 1:
                break
            
            # Calculate holding period
            next_signal_idx = df.index.get_loc(signal_date)
            exit_idx = min(next_signal_idx + holding_days, len(df) - 1)
            
            if exit_idx >= len(df):
                continue
            
            exit_date = df.index[exit_idx]
            
            # Get entry and exit prices
            entry_price = df.loc[signal_date, "Close"]
            exit_price = df.loc[exit_date, "Close"]
            
            # Simulate percentage move (simplified options pricing)
            # Real options pricing would need IV and Greeks
            price_change_pct = (exit_price - entry_price) / entry_price
            
            if option_type == "call":
                option_return = max(0, price_change_pct) * 2  # Leverage effect
            else:
                option_return = max(0, -price_change_pct) * 2
            
            # Simulate entry/exit option prices (percentage of underlying)
            entry_opt_price = entry_price * 0.03  # ~3% of stock price
            exit_opt_price = entry_opt_price * (1 + option_return)
            
            # Calculate position size (use 10% of capital per trade)
            position_size = int((capital * 0.1) / (entry_opt_price * 100))
            if position_size < 1:
                position_size = 1
            
            # Simulate trade
            trade_pnl = (exit_opt_price - entry_opt_price) * position_size * 100
            trade_return = (trade_pnl / (entry_opt_price * position_size * 100)) * 100
            
            # Update capital
            capital += trade_pnl
            capital_history.append(capital)
            dates_history.append(exit_date)
            
            trades.append({
                "entry_date": signal_date.strftime("%Y-%m-%d"),
                "exit_date": exit_date.strftime("%Y-%m-%d"),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "option_entry": entry_opt_price,
                "option_exit": exit_opt_price,
                "position_size": position_size,
                "pnl": trade_pnl,
                "return_pct": trade_return,
                "capital_after": capital
            })
        
        if not trades:
            return {
                "error": "No completed trades",
                "trades": [],
                "metrics": {}
            }
        
        # Calculate metrics
        metrics = calculate_metrics(trades, capital_history, dates_history, initial_capital)
        
        return {
            "ticker": ticker,
            "strategy": strategy,
            "period": f"{start_date} to {end_date}",
            "initial_capital": initial_capital,
            "final_capital": capital,
            "trades": trades,
            "capital_history": list(zip(dates_history, capital_history)),
            "metrics": metrics
        }
    except Exception as e:
        return {"error": str(e)}


def calculate_metrics(trades: List[Dict], capital_history: List[float],
                     dates_history: List[datetime], initial_capital: float) -> Dict:
    """
    Calculate performance metrics from backtest results.
    
    Args:
        trades: List of trade dictionaries
        capital_history: List of capital values over time
        dates_history: Corresponding dates for capital history
        initial_capital: Starting capital
    
    Returns:
        Dictionary with performance metrics
    """
    if not trades:
        return {}
    
    trades_df = pd.DataFrame(trades)
    capital_series = pd.Series(capital_history, index=dates_history)
    
    # Basic metrics
    total_return = capital_history[-1] - initial_capital
    total_return_pct = (total_return / initial_capital) * 100
    
    num_trades = len(trades)
    winning_trades = len(trades_df[trades_df["pnl"] > 0])
    losing_trades = len(trades_df[trades_df["pnl"] <= 0])
    win_rate = (winning_trades / num_trades * 100) if num_trades > 0 else 0
    
    # P&L metrics
    avg_win = trades_df[trades_df["pnl"] > 0]["pnl"].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df["pnl"] <= 0]["pnl"].mean() if losing_trades > 0 else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    
    # Max drawdown
    cummax = capital_series.cummax()
    drawdown = (capital_series - cummax) / cummax * 100
    max_drawdown = drawdown.min()
    
    # Sharpe ratio
    if len(capital_series) > 1:
        returns = capital_series.pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Trade duration
    trades_df["duration"] = pd.to_datetime(trades_df["exit_date"]) - pd.to_datetime(trades_df["entry_date"])
    avg_duration = trades_df["duration"].mean()
    
    return {
        "total_return": round(total_return, 2),
        "total_return_pct": round(total_return_pct, 2),
        "num_trades": num_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": round(win_rate, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "inf",
        "max_drawdown": round(max_drawdown, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "avg_duration_days": avg_duration.days if hasattr(avg_duration, 'days') else 0,
        "best_trade": round(trades_df["pnl"].max(), 2),
        "worst_trade": round(trades_df["pnl"].min(), 2)
    }


def create_equity_curve(trades: List[Dict], dates: List[datetime],
                        initial_capital: float) -> pd.DataFrame:
    """
    Create equity curve data from backtest trades.
    
    Args:
        trades: List of trade dictionaries
        dates: List of dates
        initial_capital: Starting capital
    
    Returns:
        DataFrame with equity curve data
    """
    if not trades:
        return pd.DataFrame()
    
    equity_data = [{"date": dates[0], "equity": initial_capital, "drawdown": 0}]
    
    peak = initial_capital
    
    for trade, date in zip(trades, dates[1:]):
        equity = trade["capital_after"]
        peak = max(peak, equity)
        drawdown = ((equity - peak) / peak) * 100
        
        equity_data.append({
            "date": date,
            "equity": equity,
            "drawdown": round(drawdown, 2)
        })
    
    return pd.DataFrame(equity_data)


def compare_strategies(ticker: str, start_date: str, end_date: str,
                      strategies: List[str] = None) -> pd.DataFrame:
    """
    Compare multiple strategies for a ticker.
    
    Args:
        ticker: Stock symbol
        start_date: Backtest start date
        end_date: Backtest end date
        strategies: List of strategies to compare
    
    Returns:
        DataFrame comparing strategy performance
    """
    if strategies is None:
        strategies = ["standard", "aggressive", "conservative"]
    
    results = []
    
    for strategy in strategies:
        result = backtest_strategy(ticker, start_date, end_date, strategy)
        
        if "error" not in result and result.get("metrics"):
            metrics = result["metrics"]
            results.append({
                "Strategy": strategy.capitalize(),
                "Total Return %": metrics["total_return_pct"],
                "Win Rate %": metrics["win_rate"],
                "Sharpe Ratio": metrics["sharpe_ratio"],
                "Max Drawdown %": metrics["max_drawdown"],
                "Profit Factor": metrics["profit_factor"],
                "Num Trades": metrics["num_trades"]
            })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Testing backtest.py module...")
    
    # Run a simple backtest
    print("\nRunning backtest on AAPL...")
    
    result = backtest_strategy(
        ticker="AAPL",
        start_date="2023-01-01",
        end_date="2023-12-31",
        strategy="standard",
        option_type="call",
        holding_days=5,
        initial_capital=10000
    )
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"\nBacktest Results for {result['ticker']}:")
        print(f"  Period: {result['period']}")
        print(f"  Initial Capital: ${result['initial_capital']:,.2f}")
        print(f"  Final Capital: ${result['final_capital']:,.2f}")
        
        metrics = result["metrics"]
        print(f"\nPerformance Metrics:")
        print(f"  Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"  Win Rate: {metrics['win_rate']:.2f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"  Profit Factor: {metrics['profit_factor']}")
        print(f"  Number of Trades: {metrics['num_trades']}")
        
        if result["trades"]:
            print(f"\nLast 3 Trades:")
            for trade in result["trades"][-3:]:
                print(f"  {trade['entry_date']} -> {trade['exit_date']}: ${trade['pnl']:.2f}")
    
    # Compare strategies
    print("\nComparing strategies...")
    comparison = compare_strategies(
        ticker="SPY",
        start_date="2023-06-01",
        end_date="2023-12-31",
        strategies=["standard", "aggressive", "conservative"]
    )
    
    if not comparison.empty:
        print("\nStrategy Comparison:")
        print(comparison.to_string(index=False))
    
    print("\nbacktest.py module test complete!")
