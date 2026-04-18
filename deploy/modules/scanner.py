"""
scanner.py - Options buy signal scanner for trading dashboard.
Scans watchlist tickers for high-probability options buy setups.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from .data import (
    get_watchlist, get_ohlcv, get_options_chain, get_current_price
)
from .indicators import calculate_all_indicators, scan_for_signals
from .gex import calculate_gex_profile, get_near_term_gex


def calculate_signal_score(ticker: str, strike: float, option_type: str,
                           expiration: str, df: pd.DataFrame = None) -> Dict:
    """
    Calculate signal score for a potential options trade.
    
    Scoring criteria (7 total signals):
    1. IV Rank below 30 (+1)
    2. Price above 21 EMA (+1)
    3. Price above 50 SMA (+1)
    4. MACD histogram turning positive (+1)
    5. GEX is negative (+1)
    6. RSI between 40-60 (+1)
    7. Open interest above 500 on flagged strike (+1)
    
    Args:
        ticker: Stock symbol
        strike: Option strike price
        option_type: "call" or "put"
        expiration: Expiration date string
        df: Pre-loaded OHLCV DataFrame
    
    Returns:
        Dictionary with score and individual signal results
    """
    if df is None:
        try:
            df = get_ohlcv(ticker, period="3mo")
        except Exception:
            return {"error": "Could not load price data", "score": 0, "signals": {}}
    
    try:
        indicators = calculate_all_indicators(df, ticker)
        current_price = df["Close"].iloc[-1]
        
        signals = {}
        score = 0
        
        # Signal 1: IV Rank below 30
        iv_rank = indicators.get("iv_rank", 50)
        signals["iv_rank_below_30"] = iv_rank < 30
        if signals["iv_rank_below_30"]:
            score += 1
        
        # Signal 2: Price above 21 EMA
        ema_21 = indicators["ema_9"].iloc[-1]
        signals["price_above_ema21"] = current_price > ema_21
        if signals["price_above_ema21"]:
            score += 1
        
        # Signal 3: Price above 50 SMA
        sma_50 = indicators["sma_50"].iloc[-1]
        signals["price_above_sma50"] = current_price > sma_50
        if signals["price_above_sma50"]:
            score += 1
        
        # Signal 4: MACD histogram turning positive
        macd_hist = indicators["macd_histogram"]
        current_hist = macd_hist.iloc[-1]
        prev_hist = macd_hist.iloc[-2] if len(macd_hist) > 1 else 0
        signals["macd_turning_positive"] = prev_hist < 0 and current_hist > 0
        if signals["macd_turning_positive"]:
            score += 1
        
        # Signal 5: GEX is negative
        try:
            gex_data = get_near_term_gex(ticker, days_to_expiry=30)
            net_gex = gex_data.get("net_gex", 0)
            signals["gex_negative"] = net_gex < 0
            gex_value = net_gex
        except Exception:
            signals["gex_negative"] = False
            gex_value = 0
        if signals["gex_negative"]:
            score += 1
        
        # Signal 6: RSI between 40-60
        rsi = indicators["rsi"].iloc[-1]
        signals["rsi_neutral"] = 40 <= rsi <= 60
        if signals["rsi_neutral"]:
            score += 1
        
        # Signal 7: Open interest above 500
        try:
            calls, puts, _ = get_options_chain(ticker, expiration)
            option_df = calls if option_type == "call" else puts
            strike_oi = option_df[option_df["strike"] == strike]["openInterest"].sum()
            signals["high_open_interest"] = strike_oi > 500
            open_interest = strike_oi
        except Exception:
            signals["high_open_interest"] = False
            open_interest = 0
        if signals["high_open_interest"]:
            score += 1
        
        # Additional data for the result
        try:
            calls, puts, _ = get_options_chain(ticker, expiration)
            option_df = calls if option_type == "call" else puts
            strike_row = option_df[option_df["strike"] == strike]
            
            if not strike_row.empty:
                option_price = strike_row["lastPrice"].iloc[0] if pd.notna(strike_row["lastPrice"].iloc[0]) else 0
                volume = strike_row["volume"].iloc[0] if pd.notna(strike_row["volume"].iloc[0]) else 0
                implied_vol = strike_row["impliedVolatility"].iloc[0] if "impliedVolatility" in strike_row.columns and pd.notna(strike_row["impliedVolatility"].iloc[0]) else 0
            else:
                option_price = 0
                volume = 0
                implied_vol = 0
        except Exception:
            option_price = 0
            volume = 0
            implied_vol = 0
        
        return {
            "ticker": ticker,
            "current_price": current_price,
            "strike": strike,
            "expiration": expiration,
            "option_type": option_type,
            "score": score,
            "max_score": 7,
            "iv_rank": round(iv_rank, 1),
            "gex_value": round(gex_value, 0),
            "rsi": round(rsi, 1),
            "open_interest": int(open_interest),
            "option_price": round(option_price, 2),
            "volume": int(volume),
            "implied_vol": round(implied_vol * 100, 1) if implied_vol else 0,
            "signals": signals,
            "flag": "BUY" if score >= 5 else ("WATCH" if score >= 4 else "NEUTRAL")
        }
    except Exception as e:
        return {"error": str(e), "score": 0, "signals": {}}


def find_optimal_strike(ticker: str, option_type: str, expiration: str,
                        df: pd.DataFrame = None) -> Optional[float]:
    """
    Find the optimal strike price for an options trade based on signals.
    
    Args:
        ticker: Stock symbol
        option_type: "call" or "put"
        expiration: Expiration date string
        df: Pre-loaded OHLCV DataFrame
    
    Returns:
        Optimal strike price or None
    """
    try:
        calls, puts, _ = get_options_chain(ticker, expiration)
        option_df = calls if option_type == "call" else puts
        
        if option_df.empty:
            return None
        
        if df is None:
            df = get_ohlcv(ticker, period="3mo")
        
        current_price = df["Close"].iloc[-1]
        
        # Filter strikes by open interest (> 100)
        option_df = option_df[option_df["openInterest"] > 100]
        
        if option_df.empty:
            return None
        
        if option_type == "call":
            # For calls, prefer slightly OTM strikes
            otm_strikes = option_df[option_df["strike"] >= current_price * 0.95]
            if not otm_strikes.empty:
                # Find strike with best balance of OI and distance
                best_strike = otm_strikes.loc[otm_strikes["openInterest"].idxmax()]
                return float(best_strike["strike"])
            return float(option_df["strike"].iloc[len(option_df) // 2])
        else:
            # For puts, prefer slightly OTM strikes
            otm_strikes = option_df[option_df["strike"] <= current_price * 1.05]
            if not otm_strikes.empty:
                best_strike = otm_strikes.loc[otm_strikes["openInterest"].idxmax()]
                return float(best_strike["strike"])
            return float(option_df["strike"].iloc[len(option_df) // 2])
    except Exception:
        return None


def scan_watchlist(watchlist: List[str] = None, min_score: int = 4,
                   days_to_expiry: int = 30) -> pd.DataFrame:
    """
    Scan the watchlist for options buy signals.
    
    Args:
        watchlist: List of tickers to scan (uses stored watchlist if None)
        min_score: Minimum signal score to include in results
        days_to_expiry: Days to expiration for options
    
    Returns:
        DataFrame with scan results sorted by score
    """
    if watchlist is None:
        watchlist = get_watchlist()
    
    if not watchlist:
        return pd.DataFrame()
    
    results = []
    errors = []
    
    # Get nearest expiration
    try:
        _, _, expirations = get_options_chain(watchlist[0])
        if expirations:
            # Find expiration within days_to_expiry
            target_date = datetime.now() + timedelta(days=days_to_expiry)
            expiration = None
            
            for exp in expirations:
                exp_date = datetime.strptime(exp, "%Y-%m-%d")
                if 0 <= (exp_date - datetime.now()).days <= days_to_expiry:
                    expiration = exp
                    break
            
            if expiration is None:
                expiration = expirations[0]
        else:
            expiration = None
    except Exception as e:
        errors.append(f"Could not get expiration dates: {e}")
        expiration = None
    
    for ticker in watchlist:
        try:
            # Get price data
            df = get_ohlcv(ticker, period="3mo")
            current_price = df["Close"].iloc[-1]
            
            # Try both call and put options
            for option_type in ["call", "put"]:
                if expiration is None:
                    continue
                
                # Find optimal strike
                strike = find_optimal_strike(ticker, option_type, expiration, df)
                
                if strike is None:
                    continue
                
                # Calculate signal score
                signal_result = calculate_signal_score(
                    ticker, strike, option_type, expiration, df
                )
                
                if "error" not in signal_result and signal_result["score"] >= min_score:
                    results.append(signal_result)
        except Exception as e:
            errors.append(f"{ticker}: {str(e)}")
    
    if not results:
        return pd.DataFrame()
    
    # Create results DataFrame
    df_results = pd.DataFrame(results)
    
    # Add color coding for flag
    def get_flag_color(score):
        if score >= 6:
            return "green"
        elif score >= 5:
            return "yellow"
        elif score >= 4:
            return "orange"
        return "gray"
    
    df_results["flag_color"] = df_results["score"].apply(get_flag_color)
    
    # Sort by score descending
    df_results = df_results.sort_values("score", ascending=False)
    
    return df_results


def get_scan_summary(results_df: pd.DataFrame) -> Dict:
    """
    Get summary statistics from scan results.
    
    Args:
        results_df: DataFrame from scan_watchlist
    
    Returns:
        Dictionary with summary statistics
    """
    if results_df.empty:
        return {
            "total_signals": 0,
            "buy_signals": 0,
            "watch_signals": 0,
            "avg_score": 0,
            "tickers_scanned": 0
        }
    
    return {
        "total_signals": len(results_df),
        "buy_signals": len(results_df[results_df["score"] >= 6]),
        "watch_signals": len(results_df[(results_df["score"] >= 4) & (results_df["score"] < 6)]),
        "avg_score": round(results_df["score"].mean(), 1),
        "tickers_scanned": results_df["ticker"].nunique(),
        "call_signals": len(results_df[results_df["option_type"] == "call"]),
        "put_signals": len(results_df[results_df["option_type"] == "put"])
    }


def create_scan_chart(results_df: pd.DataFrame) -> Dict[str, any]:
    """
    Create visualization data for scan results.
    
    Args:
        results_df: DataFrame from scan_watchlist
    
    Returns:
        Dictionary with chart data
    """
    if results_df.empty:
        return {}
    
    # Score distribution
    score_counts = results_df["score"].value_counts().sort_index()
    
    # Signals by ticker
    ticker_scores = results_df.groupby("ticker")["score"].max().sort_values(ascending=False)
    
    # Option type distribution
    option_dist = results_df["option_type"].value_counts()
    
    return {
        "score_distribution": score_counts.to_dict(),
        "ticker_scores": ticker_scores.to_dict(),
        "option_distribution": option_dist.to_dict(),
        "avg_iv_rank": round(results_df["iv_rank"].mean(), 1),
        "avg_rsi": round(results_df["rsi"].mean(), 1),
        "total_oi": results_df["open_interest"].sum()
    }


if __name__ == "__main__":
    print("Testing scanner.py module...")
    
    # Add some test tickers to watchlist if needed
    from .data import add_to_watchlist, get_watchlist
    
    watchlist = get_watchlist()
    print(f"\nCurrent watchlist: {watchlist}")
    
    if not watchlist:
        # Add some test tickers
        for ticker in ["SPY", "QQQ", "AAPL", "TSLA"]:
            add_to_watchlist(ticker)
        watchlist = get_watchlist()
        print(f"Watchlist after adding: {watchlist}")
    
    # Run scan
    print("\nScanning watchlist...")
    results = scan_watchlist(watchlist, min_score=4)
    
    if not results.empty:
        print(f"\nFound {len(results)} signals:")
        print(results[["ticker", "strike", "expiration", "option_type", "score", "flag"]].to_string())
        
        # Get summary
        summary = get_scan_summary(results)
        print(f"\nScan Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    else:
        print("\nNo signals found meeting the criteria.")
    
    print("\nscanner.py module test complete!")
