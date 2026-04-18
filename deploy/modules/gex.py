"""
gex.py - Gamma Exposure (GEX) Calculator for trading dashboard.
Calculates dealer gamma by strike and identifies max pain and gamma flip levels.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from plotly import graph_objects as go

from .data import get_options_chain, get_current_price, get_ohlcv
from .indicators import calculate_all_indicators


def calculate_gex_profile(ticker: str, expiration: str = None) -> Dict:
    """
    Calculate Gamma Exposure profile for a ticker.
    
    GEX Formula: For each strike, calculate:
    - Net gamma from calls and puts
    - Normalize by distance from current price
    
    Args:
        ticker: Stock symbol
        expiration: Specific expiration date (YYYY-MM-DD)
    
    Returns:
        Dictionary with gex data, max_pain, gamma_flip, and strike_data
    """
    try:
        calls, puts, expirations = get_options_chain(ticker, expiration)
        
        if calls.empty or puts.empty:
            return {"error": "No options data available"}
        
        if expiration is None:
            expiration = expirations[0]
        
        current_price = get_current_price(ticker)
        
        # Filter for liquid options (open interest > 50)
        calls_filtered = calls[calls["openInterest"] > 50].copy()
        puts_filtered = puts[puts["openInterest"] > 50].copy()
        
        # Calculate gamma exposure for each strike
        # GEX = Open Interest * Absolute Delta * 0.01 (scaling factor)
        # For OTM calls: delta approaches 1 as price increases
        # For OTM puts: delta approaches 0 as price decreases (negative delta)
        
        calls_filtered["gamma_exposure"] = 0.0
        puts_filtered["gamma_exposure"] = 0.0
        
        # Simplified gamma calculation using option delta approximation
        # Real GEX requires actual delta values from broker
        
        for idx, row in calls_filtered.iterrows():
            strike = row["strike"]
            oi = row["openInterest"]
            price = row["lastPrice"] if pd.notna(row["lastPrice"]) else 0.01
            
            # Approximate delta based on Moneyness
            moneyness = current_price / strike
            if moneyness > 1.1:  # Deep ITM
                delta = 1.0
            elif moneyness > 1.0:  # ITM
                delta = 0.75
            elif moneyness > 0.95:  # ATM
                delta = 0.5
            elif moneyness > 0.9:  # Slightly OTM
                delta = 0.25
            else:  # OTM
                delta = 0.05
            
            calls_filtered.at[idx, "gamma_exposure"] = oi * delta * 0.01 * price
            calls_filtered.at[idx, "delta_approx"] = delta
        
        for idx, row in puts_filtered.iterrows():
            strike = row["strike"]
            oi = row["openInterest"]
            price = row["lastPrice"] if pd.notna(row["lastPrice"]) else 0.01
            
            # Approximate delta for puts (negative)
            moneyness = strike / current_price
            if moneyness > 1.1:  # Deep ITM (OTM puts)
                delta = -0.05
            elif moneyness > 1.0:  # ITM puts
                delta = -0.25
            elif moneyness > 0.95:  # ATM
                delta = -0.5
            elif moneyness > 0.9:  # Slightly OTM
                delta = -0.75
            else:  # Deep OTM
                delta = -1.0
            
            puts_filtered.at[idx, "gamma_exposure"] = oi * abs(delta) * 0.01 * price
            puts_filtered.at[idx, "delta_approx"] = delta
        
        # Combine calls and puts gamma
        all_gamma = pd.concat([
            calls_filtered[["strike", "gamma_exposure", "openInterest", "volume"]].assign(type="call"),
            puts_filtered[["strike", "gamma_exposure", "openInterest", "volume"]].assign(type="put")
        ])
        
        # Calculate net gamma by strike
        gamma_by_strike = all_gamma.groupby("strike").agg({
            "gamma_exposure": "sum",
            "openInterest": "sum",
            "volume": "sum"
        }).reset_index()
        
        # Calculate max pain
        max_pain = calculate_max_pain(ticker, calls, puts, current_price)
        
        # Calculate gamma flip level (where net gamma = 0)
        gamma_flip = find_gamma_flip(gamma_by_strike)
        
        # Calculate total GEX
        total_call_gamma = calls_filtered["gamma_exposure"].sum()
        total_put_gamma = puts_filtered["gamma_exposure"].sum()
        net_gex = total_call_gamma - total_put_gamma
        
        return {
            "ticker": ticker,
            "expiration": expiration,
            "current_price": current_price,
            "max_pain": max_pain,
            "gamma_flip": gamma_flip,
            "total_call_gamma": total_call_gamma,
            "total_put_gamma": total_put_gamma,
            "net_gex": net_gex,
            "gamma_by_strike": gamma_by_strike.to_dict("records"),
            "calls": calls_filtered.to_dict("records"),
            "puts": puts_filtered.to_dict("records"),
            "expirations": expirations
        }
    except Exception as e:
        return {"error": str(e)}


def calculate_max_pain(ticker: str, calls: pd.DataFrame, puts: pd.DataFrame, 
                        current_price: float) -> float:
    """
    Calculate the Max Pain strike price.
    
    Max Pain: The strike price at which the maximum number of options 
    expire worthless (both calls and puts lose value).
    
    Args:
        ticker: Stock symbol
        calls: Calls DataFrame
        puts: Puts DataFrame
        current_price: Current stock price
    
    Returns:
        Max pain strike price
    """
    # Get unique strikes
    strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
    
    pain_by_strike = {}
    
    for strike in strikes:
        # Calculate pain for calls (call holders lose when price falls)
        call_pain = 0
        for _, row in calls.iterrows():
            if row["strike"] > strike:
                # ITM call: intrinsic value = price - strike
                intrinsic = max(0, current_price - row["strike"])
                call_pain += (row.get("lastPrice", 0.01) - intrinsic) * row.get("openInterest", 0)
        
        # Calculate pain for puts (put holders lose when price rises)
        put_pain = 0
        for _, row in puts.iterrows():
            if row["strike"] < strike:
                # ITM put: intrinsic value = strike - price
                intrinsic = max(0, row["strike"] - current_price)
                put_pain += (row.get("lastPrice", 0.01) - intrinsic) * row.get("openInterest", 0)
        
        pain_by_strike[strike] = call_pain + put_pain
    
    if not pain_by_strike:
        return current_price
    
    return min(pain_by_strike, key=pain_by_strike.get)


def find_gamma_flip(gamma_df: pd.DataFrame) -> Optional[float]:
    """
    Find the strike price where net gamma flips from positive to negative.
    
    Args:
        gamma_df: DataFrame with strike and gamma_exposure columns
    
    Returns:
        Gamma flip strike or None
    """
    if gamma_df.empty or len(gamma_df) < 2:
        return None
    
    gamma_df = gamma_df.sort_values("strike").reset_index(drop=True)
    
    for i in range(1, len(gamma_df)):
        prev_gex = gamma_df["gamma_exposure"].iloc[i-1]
        curr_gex = gamma_df["gamma_exposure"].iloc[i]
        
        # Check for sign change
        if prev_gex >= 0 and curr_gex < 0:
            # Linear interpolation to find exact flip point
            prev_strike = gamma_df["strike"].iloc[i-1]
            curr_strike = gamma_df["strike"].iloc[i]
            
            ratio = abs(prev_gex) / (abs(prev_gex) + abs(curr_gex))
            flip_strike = prev_strike + ratio * (curr_strike - prev_strike)
            return round(flip_strike, 2)
        elif prev_gex < 0 and curr_gex >= 0:
            prev_strike = gamma_df["strike"].iloc[i-1]
            curr_strike = gamma_df["strike"].iloc[i]
            
            ratio = abs(curr_gex) / (abs(prev_gex) + abs(curr_gex))
            flip_strike = prev_strike + ratio * (curr_strike - prev_strike)
            return round(flip_strike, 2)
    
    return None


def create_gex_chart(gex_data: Dict) -> go.Figure:
    """
    Create a Plotly bar chart showing GEX profile by strike.
    
    Args:
        gex_data: Output from calculate_gex_profile
    
    Returns:
        Plotly Figure object
    """
    if "error" in gex_data:
        fig = go.Figure()
        fig.add_annotation(text=gex_data["error"], showarrow=False)
        return fig
    
    gamma_by_strike = gex_data["gamma_by_strike"]
    
    if not gamma_by_strike:
        fig = go.Figure()
        fig.add_annotation(text="No gamma data available", showarrow=False)
        return fig
    
    df = pd.DataFrame(gamma_by_strike)
    df = df.sort_values("strike")
    
    # Color bars based on positive/negative gamma
    colors = ["#26a69a" if g >= 0 else "#ef5350" for g in df["gamma_exposure"]]
    
    fig = go.Figure()
    
    # GEX bars
    fig.add_trace(go.Bar(
        x=df["strike"],
        y=df["gamma_exposure"],
        marker_color=colors,
        name="Gamma Exposure"
    ))
    
    # Add vertical lines for max pain, current price, and gamma flip
    current_price = gex_data["current_price"]
    max_pain = gex_data["max_pain"]
    gamma_flip = gex_data.get("gamma_flip")
    
    # Max pain line (blue dashed)
    fig.add_vline(
        x=max_pain, 
        line_dash="dash", 
        line_color="#2196F3",
        annotation_text=f"Max Pain: ${max_pain:.2f}",
        annotation_position="top"
    )
    
    # Current price line (white solid)
    fig.add_vline(
        x=current_price,
        line_dash="solid",
        line_color="white",
        annotation_text=f"Current: ${current_price:.2f}",
        annotation_position="bottom"
    )
    
    # Gamma flip line (yellow dashed)
    if gamma_flip:
        fig.add_vline(
            x=gamma_flip,
            line_dash="dot",
            line_color="#FFEB3B",
            annotation_text=f"Gamma Flip: ${gamma_flip:.2f}",
            annotation_position="top"
        )
    
    # Summary annotation
    net_gex = gex_data["net_gex"]
    gex_status = "POSITIVE" if net_gex >= 0 else "NEGATIVE"
    summary_text = f"Net GEX: ${net_gex:,.0f} ({gex_status})"
    
    fig.update_layout(
        title=f"{gex_data['ticker']} Gamma Exposure - {gex_data['expiration']}<br><sup>{summary_text}</sup>",
        xaxis_title="Strike Price",
        yaxis_title="Gamma Exposure ($)",
        template="plotly_dark",
        height=500,
        showlegend=False,
        hovermode="x unified"
    )
    
    return fig


def get_gex_summary(gex_data: Dict) -> pd.DataFrame:
    """
    Get GEX summary as a DataFrame for display.
    
    Args:
        gex_data: Output from calculate_gex_profile
    
    Returns:
        DataFrame with GEX summary
    """
    if "error" in gex_data:
        return pd.DataFrame([{"Error": gex_data["error"]}])
    
    return pd.DataFrame([
        {"Metric": "Ticker", "Value": gex_data["ticker"]},
        {"Metric": "Expiration", "Value": gex_data["expiration"]},
        {"Metric": "Current Price", "Value": f"${gex_data['current_price']:.2f}"},
        {"Metric": "Max Pain Strike", "Value": f"${gex_data['max_pain']:.2f}"},
        {"Metric": "Gamma Flip Level", "Value": f"${gex_data.get('gamma_flip', 'N/A'):.2f}" if gex_data.get('gamma_flip') else "N/A"},
        {"Metric": "Total Call Gamma", "Value": f"${gex_data['total_call_gamma']:,.0f}"},
        {"Metric": "Total Put Gamma", "Value": f"${gex_data['total_put_gamma']:,.0f}"},
        {"Metric": "Net GEX", "Value": f"${gex_data['net_gex']:,.0f}"},
        {"Metric": "GEX Status", "Value": "POSITIVE" if gex_data['net_gex'] >= 0 else "NEGATIVE"}
    ])


def get_near_term_gex(ticker: str, days_to_expiry: int = 7) -> Dict:
    """
    Get GEX for the nearest expiration within specified days.
    
    Args:
        ticker: Stock symbol
        days_to_expiry: Maximum days until expiration
    
    Returns:
        GEX data dictionary
    """
    try:
        _, _, expirations = get_options_chain(ticker)
        
        from datetime import datetime, timedelta
        
        target_date = datetime.now() + timedelta(days=days_to_expiry)
        
        # Find closest expiration
        closest_exp = None
        min_days = float('inf')
        
        for exp in expirations:
            exp_date = datetime.strptime(exp, "%Y-%m-%d")
            days_diff = (exp_date - datetime.now()).days
            
            if 0 <= days_diff <= days_to_expiry and days_diff < min_days:
                min_days = days_diff
                closest_exp = exp
        
        if closest_exp:
            return calculate_gex_profile(ticker, closest_exp)
        
        # Fallback to nearest expiration
        return calculate_gex_profile(ticker, expirations[0])
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    print("Testing gex.py module...")
    
    # Test with SPY for realistic options data
    try:
        gex_data = calculate_gex_profile("SPY")
        
        if "error" not in gex_data:
            print(f"\nGEX Summary for {gex_data['ticker']}:")
            print(gex_data)
            
            summary = get_gex_summary(gex_data)
            print("\nGEX Summary Table:")
            print(summary.to_string(index=False))
            
            print(f"\nMax Pain: ${gex_data['max_pain']:.2f}")
            print(f"Gamma Flip: ${gex_data.get('gamma_flip', 'N/A')}")
            print(f"Net GEX: ${gex_data['net_gex']:,.0f}")
        else:
            print(f"Error: {gex_data['error']}")
    except Exception as e:
        print(f"Error calculating GEX: {e}")
    
    print("\ngex.py module test complete!")
