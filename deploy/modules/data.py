"""
data.py - Data fetching, caching, and rate limiting for trading dashboard.
All data pulled from yfinance with SQLite caching and request throttling.
"""

import sqlite3
import time
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from functools import wraps
import pandas as pd
import yfinance as yf
import requests
from dotenv import load_dotenv

load_dotenv()

# Database path - use temp directory on Streamlit Cloud
if os.path.exists("/mount"):
    DATA_DIR = "/tmp/trading_dashboard_data"
else:
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "cache.db")

# Rate limiting: minimum seconds between requests
RATE_LIMIT_SECONDS = 0.5
_last_request_time = 0

# Cache duration in minutes (configurable via .env)
CACHE_DURATION = int(os.getenv("CACHE_DURATION", "15"))


def _get_db_connection():
    """Create a connection to the SQLite cache database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Create tables if they don't exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            ticker TEXT,
            data_type TEXT,
            fetched_at TEXT,
            data TEXT,
            PRIMARY KEY (ticker, data_type)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY,
            ticker TEXT UNIQUE,
            position INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            strike REAL,
            expiration TEXT,
            option_type TEXT,
            quantity INTEGER,
            entry_price REAL,
            current_price REAL,
            entry_date TEXT,
            status TEXT
        )
    """)
    conn.commit()
    return conn


def _rate_limit():
    """Enforce rate limiting between requests to avoid being rate limited."""
    global _last_request_time
    current_time = time.time()
    elapsed = current_time - _last_request_time
    if elapsed < RATE_LIMIT_SECONDS:
        time.sleep(RATE_LIMIT_SECONDS - elapsed)
    _last_request_time = time.time()


def _is_market_hours() -> bool:
    """Check if current time is during market hours (9:30am - 4pm ET Mon-Fri)."""
    now = datetime.now()
    weekday = now.weekday()
    hour = now.hour
    minute = now.minute
    
    # Market hours: Mon-Fri (0-4), 9:30am - 4:00pm ET
    # Simplified check (doesn't account for holidays)
    if weekday >= 5:  # Saturday or Sunday
        return False
    current_minutes = hour * 60 + minute
    market_open = 9 * 60 + 30  # 9:30 AM
    market_close = 16 * 60     # 4:00 PM
    return market_open <= current_minutes < market_close


def _get_cache_age(ticker: str, data_type: str) -> Optional[int]:
    """Get the age of cached data in minutes. Returns None if not cached."""
    conn = _get_db_connection()
    try:
        cursor = conn.execute(
            "SELECT fetched_at FROM cache WHERE ticker = ? AND data_type = ?",
            (ticker.upper(), data_type)
        )
        row = cursor.fetchone()
        if row:
            fetched_at = datetime.fromisoformat(row["fetched_at"])
            age = (datetime.now() - fetched_at).total_seconds() / 60
            return int(age)
        return None
    finally:
        conn.close()


def _write_cache(ticker: str, data_type: str, data: pd.DataFrame):
    """Write data to cache database."""
    conn = _get_db_connection()
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (ticker TEXT, data_type TEXT, "
            "fetched_at TEXT, data TEXT, PRIMARY KEY (ticker, data_type))"
        )
        conn.execute(
            "INSERT OR REPLACE INTO cache (ticker, data_type, fetched_at, data) "
            "VALUES (?, ?, ?, ?)",
            (ticker.upper(), data_type, datetime.now().isoformat(), data.to_json())
        )
        conn.commit()
    finally:
        conn.close()


def _read_cache(ticker: str, data_type: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """Read data from cache if valid. Returns None if not cached or expired."""
    if force_refresh:
        return None
    
    age = _get_cache_age(ticker, data_type)
    if age is None:
        return None
    
    # During market hours, invalidate cache older than CACHE_DURATION
    if _is_market_hours() and age > CACHE_DURATION:
        return None
    
    # Outside market hours, allow cache up to 24 hours
    if not _is_market_hours() and age > 1440:
        return None
    
    conn = _get_db_connection()
    try:
        cursor = conn.execute(
            "SELECT data FROM cache WHERE ticker = ? AND data_type = ?",
            (ticker.upper(), data_type)
        )
        row = cursor.fetchone()
        if row:
            return pd.read_json(row["data"])
        return None
    finally:
        conn.close()


def get_ohlcv(ticker: str, period: str = "1y", force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch OHLCV data for a ticker with caching.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
        force_refresh: Bypass cache if True
    
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
    """
    cache = _read_cache(ticker, f"ohlcv_{period}", force_refresh)
    if cache is not None:
        return cache
    
    _rate_limit()
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        
        _write_cache(ticker, f"ohlcv_{period}", df)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to fetch OHLCV for {ticker}: {str(e)}")


def get_options_chain(ticker: str, expiration: str = None, force_refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Fetch options chain data for a ticker with caching.
    
    Args:
        ticker: Stock symbol
        expiration: Specific expiration date (YYYY-MM-DD), or None for all
        force_refresh: Bypass cache if True
    
    Returns:
        Tuple of (calls DataFrame, puts DataFrame, list of available expirations)
    """
    cache = _read_cache(ticker, f"options_{expiration or 'all'}", force_refresh)
    if cache is not None:
        return (
            pd.DataFrame(cache.get("calls", [])),
            pd.DataFrame(cache.get("puts", [])),
            cache.get("expirations", [])
        )
    
    _rate_limit()
    try:
        stock = yf.Ticker(ticker)
        
        # Get available expirations
        expirations = list(stock.options)
        if not expirations:
            raise ValueError(f"No options data available for {ticker}")
        
        if expiration is None:
            expiration = expirations[0]
        
        if expiration not in expirations:
            raise ValueError(f"Expiration {expiration} not available. Choose from: {expirations}")
        
        # Fetch options chain
        opt = stock.option_chain(expiration)
        calls = opt.calls
        puts = opt.puts
        
        cache_data = {
            "calls": calls.to_dict("records"),
            "puts": puts.to_dict("records"),
            "expirations": expirations
        }
        
        _write_cache(ticker, f"options_{expiration or 'all'}", pd.DataFrame([cache_data]))
        return calls, puts, expirations
    except Exception as e:
        raise RuntimeError(f"Failed to fetch options for {ticker}: {str(e)}")


def get_current_price(ticker: str, force_refresh: bool = False) -> float:
    """
    Get current/last known price for a ticker.
    
    Args:
        ticker: Stock symbol
        force_refresh: Bypass cache if True
    
    Returns:
        Current price as float
    """
    cache = _read_cache(ticker, "current_price", force_refresh)
    if cache is not None and not cache.empty:
        return float(cache.iloc[0]["close"])
    
    _rate_limit()
    try:
        stock = yf.Ticker(ticker)
        info = stock.fast_info
        price = info.last_price
        
        if price is None:
            # Fallback to history
            df = get_ohlcv(ticker, period="5d", force_refresh=force_refresh)
            price = df["Close"].iloc[-1]
        
        _write_cache(ticker, "current_price", pd.DataFrame([{"close": price}]))
        return float(price)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch price for {ticker}: {str(e)}")


def get_market_overview(tickers: List[str], force_refresh: bool = False) -> pd.DataFrame:
    """
    Get market overview data for multiple tickers.
    
    Args:
        tickers: List of stock symbols
        force_refresh: Bypass cache if True
    
    Returns:
        DataFrame with ticker, price, change, volume for each ticker
    """
    results = []
    
    for ticker in tickers:
        try:
            _rate_limit()
            stock = yf.Ticker(ticker)
            info = stock.fast_info
            
            price = info.last_price or 0
            prev_close = info.previous_close or 0
            change = price - prev_close
            change_pct = (change / prev_close * 100) if prev_close else 0
            
            # Get volume from history
            history = get_ohlcv(ticker, period="5d", force_refresh=force_refresh)
            volume = history["Volume"].iloc[-1] if not history.empty else 0
            
            results.append({
                "Ticker": ticker.upper(),
                "Price": round(price, 2),
                "Change": round(change, 2),
                "Change %": round(change_pct, 2),
                "Volume": int(volume)
            })
        except Exception as e:
            results.append({
                "Ticker": ticker.upper(),
                "Price": None,
                "Change": None,
                "Change %": None,
                "Volume": None,
                "Error": str(e)
            })
    
    return pd.DataFrame(results)


def get_iv_history(ticker: str, lookback_days: int = 30, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch IV history by scraping/simulating from options chains.
    Note: yfinance doesn't provide historical IV, so this uses OTM put/call
    implied volatility from options chain snapshots where available.
    
    Args:
        ticker: Stock symbol
        lookback_days: Number of days to look back
        force_refresh: Bypass cache if True
    
    Returns:
        DataFrame with date and IV values
    """
    # For now, return simulated IV data based on price history
    # Real IV history would require paid data sources
    cache = _read_cache(ticker, f"iv_history_{lookback_days}", force_refresh)
    if cache is not None:
        return cache
    
    # Get price history to derive synthetic IV
    df = get_ohlcv(ticker, period="3mo", force_refresh=force_refresh)
    
    # Simulate IV using historical volatility as proxy
    # This is an approximation - real IV requires options pricing
    returns = df["Close"].pct_change().dropna()
    hv = returns.rolling(window=20).std() * (252 ** 0.5) * 100  # Annualized
    
    iv_df = pd.DataFrame({
        "Date": df.index[-len(hv.dropna()):],
        "IV": hv.dropna().values
    }).reset_index(drop=True)
    
    _write_cache(ticker, f"iv_history_{lookback_days}", iv_df)
    return iv_df


def get_ticker_info(ticker: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Get fundamental info for a ticker.
    
    Args:
        ticker: Stock symbol
        force_refresh: Bypass cache if True
    
    Returns:
        Dictionary of ticker information
    """
    cache = _read_cache(ticker, "info", force_refresh)
    if cache is not None:
        return cache.to_dict("records")[0] if not cache.empty else {}
    
    _rate_limit()
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        info_df = pd.DataFrame([info])
        _write_cache(ticker, "info", info_df)
        return info
    except Exception as e:
        return {"error": str(e)}


# Watchlist management functions
def get_watchlist() -> List[str]:
    """Get the user's watchlist from database."""
    conn = _get_db_connection()
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS watchlist (id INTEGER PRIMARY KEY, "
            "ticker TEXT UNIQUE, position INTEGER)"
        )
        cursor = conn.execute("SELECT ticker FROM watchlist ORDER BY position")
        return [row["ticker"] for row in cursor.fetchall()]
    finally:
        conn.close()


def add_to_watchlist(ticker: str) -> bool:
    """Add a ticker to the watchlist. Returns True if successful."""
    ticker = ticker.upper().strip()
    conn = _get_db_connection()
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS watchlist (id INTEGER PRIMARY KEY, "
            "ticker TEXT UNIQUE, position INTEGER)"
        )
        # Get max position
        cursor = conn.execute("SELECT MAX(position) as max_pos FROM watchlist")
        max_pos = cursor.fetchone()["max_pos"] or 0
        
        conn.execute(
            "INSERT OR IGNORE INTO watchlist (ticker, position) VALUES (?, ?)",
            (ticker, max_pos + 1)
        )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def remove_from_watchlist(ticker: str) -> bool:
    """Remove a ticker from the watchlist. Returns True if successful."""
    ticker = ticker.upper().strip()
    conn = _get_db_connection()
    try:
        conn.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker,))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def reorder_watchlist(tickers: List[str]) -> bool:
    """Reorder the watchlist. Takes list of tickers in new order."""
    conn = _get_db_connection()
    try:
        conn.execute("DELETE FROM watchlist")
        for i, ticker in enumerate(tickers):
            conn.execute(
                "INSERT INTO watchlist (ticker, position) VALUES (?, ?)",
                (ticker.upper(), i)
            )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


# Paper trading account functions
def get_paper_account() -> Dict[str, Any]:
    """Get paper trading account summary."""
    conn = _get_db_connection()
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS paper_trades (id INTEGER PRIMARY KEY, "
            "ticker TEXT, strike REAL, expiration TEXT, option_type TEXT, "
            "quantity INTEGER, entry_price REAL, current_price REAL, "
            "entry_date TEXT, status TEXT)"
        )
        cursor = conn.execute(
            "SELECT SUM((current_price - entry_price) * quantity) as total_pnl, "
            "COUNT(*) as total_trades FROM paper_trades WHERE status = 'open'"
        )
        row = cursor.fetchone()
        
        cursor2 = conn.execute("SELECT SUM((current_price - entry_price) * quantity) as closed_pnl FROM paper_trades WHERE status = 'closed'")
        closed_row = cursor2.fetchone()
        
        return {
            "open_pnl": row["total_pnl"] or 0,
            "closed_pnl": closed_row["closed_pnl"] or 0,
            "total_pnl": (row["total_pnl"] or 0) + (closed_row["closed_pnl"] or 0),
            "open_trades": row["total_trades"] or 0
        }
    finally:
        conn.close()


def add_paper_trade(ticker: str, strike: float, expiration: str, 
                    option_type: str, quantity: int, entry_price: float) -> bool:
    """Add a new paper trade."""
    conn = _get_db_connection()
    try:
        conn.execute(
            "INSERT INTO paper_trades "
            "(ticker, strike, expiration, option_type, quantity, entry_price, current_price, entry_date, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ticker.upper(), strike, expiration, option_type.lower(), quantity,
             entry_price, entry_price, datetime.now().isoformat(), "open")
        )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def close_paper_trade(trade_id: int, exit_price: float) -> bool:
    """Close a paper trade."""
    conn = _get_db_connection()
    try:
        conn.execute(
            "UPDATE paper_trades SET status = 'closed', current_price = ? WHERE id = ?",
            (exit_price, trade_id)
        )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


if __name__ == "__main__":
    # Test data fetching
    print("Testing data.py module...")
    
    # Test watchlist
    print("\nWatchlist:", get_watchlist())
    
    # Test adding to watchlist
    add_to_watchlist("AAPL")
    add_to_watchlist("TSLA")
    add_to_watchlist("SPY")
    print("Watchlist after add:", get_watchlist())
    
    # Test OHLCV fetch
    try:
        df = get_ohlcv("AAPL", period="1mo")
        print(f"\nAAPL OHLCV shape: {df.shape}")
        print(df.tail())
    except Exception as e:
        print(f"Error fetching AAPL: {e}")
    
    # Test options chain
    try:
        calls, puts, expirations = get_options_chain("AAPL")
        print(f"\nAAPL options expirations: {expirations[:3]}...")
        print(f"Calls shape: {calls.shape}")
    except Exception as e:
        print(f"Error fetching options: {e}")
    
    print("\ndata.py module test complete!")
