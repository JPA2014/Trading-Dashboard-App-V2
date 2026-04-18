"""
indicators.py - Technical indicators and Plotly charting for trading dashboard.
Calculates SMA, EMA, VWAP, RSI, MACD, Bollinger Bands, ATR, and IV Rank.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from .data import get_ohlcv, get_options_chain, get_current_price


def calculate_sma(df: pd.DataFrame, period: int, column: str = "Close") -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        df: DataFrame with OHLCV data
        period: Number of periods for SMA
        column: Column to calculate SMA on
    
    Returns:
        Series with SMA values
    """
    return df[column].rolling(window=period, min_periods=1).mean()


def calculate_ema(df: pd.DataFrame, period: int, column: str = "Close") -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Args:
        df: DataFrame with OHLCV data
        period: Number of periods for EMA
        column: Column to calculate EMA on
    
    Returns:
        Series with EMA values
    """
    return df[column].ewm(span=period, adjust=False, min_periods=1).mean()


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume Weighted Average Price.
    Requires High, Low, Close, and Volume columns.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        Series with VWAP values
    """
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    cumulative_tp_vol = (typical_price * df["Volume"]).cumsum()
    cumulative_vol = df["Volume"].cumsum()
    vwap = cumulative_tp_vol / cumulative_vol
    return vwap


def calculate_rsi(df: pd.DataFrame, period: int = 14, column: str = "Close") -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    Args:
        df: DataFrame with OHLCV data
        period: RSI period (default 14)
        column: Column to calculate RSI on
    
    Returns:
        Series with RSI values (0-100)
    """
    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, 
                   signal: int = 9, column: str = "Close") -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        df: DataFrame with OHLCV data
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)
        column: Column to calculate MACD on
    
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    ema_fast = calculate_ema(df, fast, column)
    ema_slow = calculate_ema(df, slow, column)
    
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(pd.DataFrame({"Close": macd_line}), signal, "Close")
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, 
                               num_std: float = 2.0, column: str = "Close") -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        df: DataFrame with OHLCV data
        period: Moving average period (default 20)
        num_std: Number of standard deviations (default 2.0)
        column: Column to calculate on
    
    Returns:
        Tuple of (Upper band, Middle band, Lower band)
    """
    middle = calculate_sma(df, period, column)
    std = df[column].rolling(window=period, min_periods=1).std()
    
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    return upper, middle, lower


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range.
    
    Args:
        df: DataFrame with OHLCV data (requires High, Low, Close)
        period: ATR period (default 14)
    
    Returns:
        Series with ATR values
    """
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()
    
    return atr


def calculate_iv_rank(ticker: str, lookback: int = 30) -> float:
    """
    Calculate IV Rank based on historical IV data.
    IV Rank = (Current IV - Lowest IV) / (Highest IV - Lowest IV) * 100
    
    Args:
        ticker: Stock symbol
        lookback: Number of days for IV history
    
    Returns:
        IV Rank as percentage (0-100)
    """
    from .data import get_iv_history
    
    try:
        iv_df = get_iv_history(ticker, lookback_days=lookback)
        
        if iv_df.empty or "IV" not in iv_df.columns:
            return 50.0  # Default to middle if no data
        
        current_iv = iv_df["IV"].iloc[-1]
        low_iv = iv_df["IV"].min()
        high_iv = iv_df["IV"].max()
        
        if high_iv == low_iv:
            return 50.0
        
        iv_rank = ((current_iv - low_iv) / (high_iv - low_iv)) * 100
        return round(iv_rank, 2)
    except Exception:
        return 50.0


def detect_golden_death_cross(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Detect golden cross and death cross signals.
    Golden Cross: 50 SMA crosses above 200 SMA
    Death Cross: 50 SMA crosses below 200 SMA
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        Dictionary with 'golden_cross' and 'death_cross' dates (or None)
    """
    sma_50 = calculate_sma(df, 50)
    sma_200 = calculate_sma(df, 200)
    
    # Calculate crossover signals
    prev_diff = sma_50.shift(1) - sma_200.shift(1)
    curr_diff = sma_50 - sma_200
    
    # Golden cross: diff changed from negative to positive
    golden = None
    death = None
    
    for i in range(1, len(df)):
        if prev_diff.iloc[i] < 0 and curr_diff.iloc[i] > 0:
            golden = df.index[i].strftime("%Y-%m-%d")
        elif prev_diff.iloc[i] > 0 and curr_diff.iloc[i] < 0:
            death = df.index[i].strftime("%Y-%m-%d")
    
    return {"golden_cross": golden, "death_cross": death}


def calculate_all_indicators(df: pd.DataFrame, ticker: str = None) -> Dict[str, pd.Series]:
    """
    Calculate all technical indicators for a DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        ticker: Optional ticker for IV Rank calculation
    
    Returns:
        Dictionary of all indicator Series
    """
    indicators = {}
    
    # Moving averages
    indicators["sma_20"] = calculate_sma(df, 20)
    indicators["sma_50"] = calculate_sma(df, 50)
    indicators["sma_200"] = calculate_sma(df, 200)
    indicators["ema_9"] = calculate_ema(df, 9)
    indicators["ema_21"] = calculate_ema(df, 21)
    
    # VWAP
    indicators["vwap"] = calculate_vwap(df)
    
    # RSI
    indicators["rsi"] = calculate_rsi(df)
    
    # MACD
    macd, signal, histogram = calculate_macd(df)
    indicators["macd"] = macd
    indicators["macd_signal"] = signal
    indicators["macd_histogram"] = histogram
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df)
    indicators["bb_upper"] = bb_upper
    indicators["bb_middle"] = bb_middle
    indicators["bb_lower"] = bb_lower
    
    # ATR
    indicators["atr"] = calculate_atr(df)
    
    # Crossover signals
    crossovers = detect_golden_death_cross(df)
    indicators["golden_cross"] = crossovers["golden_cross"]
    indicators["death_cross"] = crossovers["death_cross"]
    
    # IV Rank (if ticker provided)
    if ticker:
        indicators["iv_rank"] = calculate_iv_rank(ticker)
    
    return indicators


def create_candlestick_chart(df: pd.DataFrame, ticker: str = "Unknown",
                             indicators: Dict[str, pd.Series] = None,
                             show_sma: bool = True,
                             show_ema: bool = True,
                             show_vwap: bool = True,
                             show_bb: bool = True,
                             show_volume: bool = True) -> go.Figure:
    """
    Create an interactive Plotly candlestick chart with technical indicators.
    
    Args:
        df: DataFrame with OHLCV data
        ticker: Stock symbol for title
        indicators: Dictionary of calculated indicators
        show_sma: Show SMA lines
        show_ema: Show EMA lines
        show_vwap: Show VWAP line
        show_bb: Show Bollinger Bands
        show_volume: Show volume subplot
    
    Returns:
        Plotly Figure object
    """
    if indicators is None:
        indicators = calculate_all_indicators(df, ticker)
    
    # Determine subplot configuration
    row_count = 2 if show_volume else 1
    row_heights = [0.7, 0.3] if show_volume else [1.0]
    
    fig = make_subplots(
        rows=row_count, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=("", "Volume"),
        specs=[[{"secondary_y": True}], [{}]]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350"
        ),
        row=1, col=1
    )
    
    # SMA lines
    if show_sma:
        fig.add_trace(
            go.Scatter(x=df.index, y=indicators["sma_20"], name="SMA 20",
                      line=dict(color="#2196F3", width=1), opacity=0.8),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=indicators["sma_50"], name="SMA 50",
                      line=dict(color="#FF9800", width=1.5), opacity=0.8),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=indicators["sma_200"], name="SMA 200",
                      line=dict(color="#9C27B0", width=1.5), opacity=0.8),
            row=1, col=1
        )
    
    # EMA lines
    if show_ema:
        fig.add_trace(
            go.Scatter(x=df.index, y=indicators["ema_9"], name="EMA 9",
                      line=dict(color="#00BCD4", width=1), opacity=0.8),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=indicators["ema_21"], name="EMA 21",
                      line=dict(color="#E91E63", width=1), opacity=0.8),
            row=1, col=1
        )
    
    # VWAP
    if show_vwap:
        fig.add_trace(
            go.Scatter(x=df.index, y=indicators["vwap"], name="VWAP",
                      line=dict(color="#FFEB3B", width=2), opacity=0.9),
            row=1, col=1
        )
    
    # Bollinger Bands
    if show_bb:
        fig.add_trace(
            go.Scatter(x=df.index, y=indicators["bb_upper"], name="BB Upper",
                      line=dict(color="rgba(156, 39, 176, 0.3)", width=1),
                      fill=None, opacity=0.5),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=indicators["bb_lower"], name="BB Lower",
                      line=dict(color="rgba(156, 39, 176, 0.3)", width=1),
                      fill="tonexty", fillcolor="rgba(156, 39, 176, 0.1)", opacity=0.3),
            row=1, col=1
        )
    
    # Volume bars
    if show_volume:
        colors = ["#26a69a" if df["Close"].iloc[i] >= df["Open"].iloc[i] 
                  else "#ef5350" for i in range(len(df))]
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=colors,
                   opacity=0.7),
            row=2, col=1
        )
    
    # Golden/Death cross markers
    if indicators.get("golden_cross"):
        gc_date = pd.to_datetime(indicators["golden_cross"])
        if gc_date in df.index:
            gc_price = df.loc[gc_date, "Close"]
            fig.add_annotation(
                x=gc_date, y=gc_price,
                text="GOLDEN CROSS",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                ax=-40,
                ay=-40,
                font=dict(color="green", size=12)
            )
    
    if indicators.get("death_cross"):
        dc_date = pd.to_datetime(indicators["death_cross"])
        if dc_date in df.index:
            dc_price = df.loc[dc_date, "Close"]
            fig.add_annotation(
                x=dc_date, y=dc_price,
                text="DEATH CROSS",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                ax=-40,
                ay=40,
                font=dict(color="red", size=12)
            )
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} - Technical Analysis",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=700,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )
    
    fig.update_xaxes(row=1, col=1, showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(row=1, col=1, showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    
    return fig


def create_rsi_chart(df: pd.DataFrame, indicators: Dict[str, pd.Series] = None) -> go.Figure:
    """
    Create RSI subplot chart.
    
    Args:
        df: DataFrame with OHLCV data
        indicators: Pre-calculated indicators
    
    Returns:
        Plotly Figure object
    """
    if indicators is None:
        indicators = calculate_all_indicators(df)
    
    fig = go.Figure()
    
    rsi = indicators["rsi"]
    
    # RSI line
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi,
        name="RSI (14)",
        line=dict(color="#9C27B0", width=2)
    ))
    
    # Overbought line (70)
    fig.add_trace(go.Scatter(
        x=df.index, y=[70] * len(df),
        name="Overbought (70)",
        line=dict(color="rgba(239, 83, 80, 0.5)", width=1, dash="dash")
    ))
    
    # Oversold line (30)
    fig.add_trace(go.Scatter(
        x=df.index, y=[30] * len(df),
        name="Oversold (30)",
        line=dict(color="rgba(38, 166, 154, 0.5)", width=1, dash="dash")
    ))
    
    # Color regions
    fig.add_trace(go.Scatter(
        x=df.index.tolist() + df.index.tolist()[::-1],
        y=[100] * len(df) + [70] * len(df),
        fill="toself",
        fillcolor="rgba(239, 83, 80, 0.1)",
        line=dict(color="rgba(239, 83, 80, 0.1)"),
        name="Overbought Zone"
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index.tolist() + df.index.tolist()[::-1],
        y=[30] * len(df) + [0] * len(df),
        fill="toself",
        fillcolor="rgba(38, 166, 154, 0.1)",
        line=dict(color="rgba(38, 166, 154, 0.1)"),
        name="Oversold Zone"
    ))
    
    fig.update_layout(
        title="RSI (14)",
        template="plotly_dark",
        height=250,
        yaxis=dict(range=[0, 100], tickvals=[0, 30, 50, 70, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    return fig


def create_macd_chart(df: pd.DataFrame, indicators: Dict[str, pd.Series] = None) -> go.Figure:
    """
    Create MACD subplot chart.
    
    Args:
        df: DataFrame with OHLCV data
        indicators: Pre-calculated indicators
    
    Returns:
        Plotly Figure object
    """
    if indicators is None:
        indicators = calculate_all_indicators(df)
    
    fig = go.Figure()
    
    macd = indicators["macd"]
    signal = indicators["macd_signal"]
    histogram = indicators["macd_histogram"]
    
    # Histogram bars (color based on value)
    colors = ["#26a69a" if val >= 0 else "#ef5350" for val in histogram]
    fig.add_trace(go.Bar(
        x=df.index, y=histogram,
        name="Histogram",
        marker_color=colors,
        opacity=0.7
    ))
    
    # MACD line
    fig.add_trace(go.Scatter(
        x=df.index, y=macd,
        name="MACD",
        line=dict(color="#2196F3", width=2)
    ))
    
    # Signal line
    fig.add_trace(go.Scatter(
        x=df.index, y=signal,
        name="Signal",
        line=dict(color="#FF9800", width=2)
    ))
    
    fig.update_layout(
        title="MACD (12, 26, 9)",
        template="plotly_dark",
        height=250,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    return fig


def get_signal_status(rsi: float) -> Tuple[str, str]:
    """
    Get RSI status and color.
    
    Args:
        rsi: RSI value
    
    Returns:
        Tuple of (status text, color)
    """
    if rsi >= 70:
        return "Overbought", "#ef5350"
    elif rsi <= 30:
        return "Oversold", "#26a69a"
    else:
        return "Neutral", "#9E9E9E"


def scan_for_signals(ticker: str, df: pd.DataFrame = None) -> Dict:
    """
    Scan for trading signals on a ticker.
    
    Args:
        ticker: Stock symbol
        df: Optional pre-loaded OHLCV data
    
    Returns:
        Dictionary with signal indicators and scores
    """
    if df is None:
        df = get_ohlcv(ticker, period="3mo")
    
    indicators = calculate_all_indicators(df, ticker)
    
    current_price = df["Close"].iloc[-1]
    price_above_ema21 = current_price > indicators["ema_9"].iloc[-1]
    price_above_sma50 = current_price > indicators["sma_50"].iloc[-1]
    
    rsi = indicators["rsi"].iloc[-1]
    macd_hist = indicators["macd_histogram"].iloc[-1]
    prev_macd_hist = indicators["macd_histogram"].iloc[-2]
    
    iv_rank = indicators.get("iv_rank", 50)
    
    # Calculate signals
    signals = {
        "price_above_ema21": price_above_ema21,
        "price_above_sma50": price_above_sma50,
        "rsi_neutral": 40 <= rsi <= 60,
        "rsi_oversold": rsi <= 40,
        "rsi_overbought": rsi >= 60,
        "macd_positive": macd_hist > 0,
        "macd_turning_positive": prev_macd_hist < 0 and macd_hist > 0,
        "iv_rank_low": iv_rank < 30,
        "iv_rank": iv_rank,
        "golden_cross_active": indicators.get("golden_cross") is not None,
        "current_price": current_price,
        "rsi": rsi,
        "ema_21": indicators["ema_9"].iloc[-1],
        "sma_50": indicators["sma_50"].iloc[-1],
        "golden_cross_date": indicators.get("golden_cross"),
        "death_cross_date": indicators.get("death_cross")
    }
    
    return signals


if __name__ == "__main__":
    print("Testing indicators.py module...")
    
    # Test with AAPL
    df = get_ohlcv("AAPL", period="3mo")
    print(f"Loaded {len(df)} days of data")
    
    # Calculate all indicators
    indicators = calculate_all_indicators(df, "AAPL")
    print(f"\nLatest values:")
    print(f"  SMA 20: {indicators['sma_20'].iloc[-1]:.2f}")
    print(f"  SMA 50: {indicators['sma_50'].iloc[-1]:.2f}")
    print(f"  EMA 9:  {indicators['ema_9'].iloc[-1]:.2f}")
    print(f"  RSI:    {indicators['rsi'].iloc[-1]:.2f}")
    print(f"  MACD:   {indicators['macd'].iloc[-1]:.4f}")
    print(f"  ATR:    {indicators['atr'].iloc[-1]:.2f}")
    print(f"  IV Rank: {indicators.get('iv_rank', 'N/A')}")
    
    # Check for crossovers
    print(f"\nCrossover signals:")
    print(f"  Golden Cross: {indicators['golden_cross']}")
    print(f"  Death Cross: {indicators['death_cross']}")
    
    # Test signal scanning
    signals = scan_for_signals("AAPL", df)
    print(f"\nSignal scan for AAPL:")
    print(f"  Price above EMA21: {signals['price_above_ema21']}")
    print(f"  Price above SMA50: {signals['price_above_sma50']}")
    print(f"  RSI: {signals['rsi']:.2f} ({get_signal_status(signals['rsi'])[0]})")
    print(f"  MACD positive: {signals['macd_positive']}")
    print(f"  IV Rank: {signals['iv_rank']:.1f}%")
    
    print("\nindicators.py module test complete!")
