"""
alerts.py - Twilio text alerts for trading dashboard.
Sends SMS notifications when high-probability trade setups are detected.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dotenv import load_dotenv
import sqlite3

load_dotenv()

# Twilio configuration from .env
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
MY_PHONE_NUMBER = os.getenv("MY_PHONE_NUMBER")

# Database for tracking sent alerts
ALERT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cache.db")


def _get_alert_db():
    """Get database connection for alert tracking."""
    conn = sqlite3.connect(ALERT_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_alert_table():
    """Initialize the alert tracking table."""
    conn = _get_alert_db()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alert_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                strike REAL,
                expiration TEXT,
                option_type TEXT,
                signal_score INTEGER,
                sent_at TEXT NOT NULL,
                message TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()


def was_alert_sent_today(ticker: str) -> bool:
    """
    Check if an alert was already sent for this ticker today.
    
    Args:
        ticker: Stock symbol
    
    Returns:
        True if alert was sent today, False otherwise
    """
    _init_alert_table()
    
    conn = _get_alert_db()
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM alert_history WHERE ticker = ? AND DATE(sent_at) = ?",
            (ticker.upper(), today)
        )
        row = cursor.fetchone()
        return row["count"] > 0 if row else False
    finally:
        conn.close()


def log_alert(ticker: str, strike: float, expiration: str, option_type: str,
              signal_score: int, message: str) -> bool:
    """
    Log a sent alert to the database.
    
    Args:
        ticker: Stock symbol
        strike: Option strike price
        expiration: Expiration date
        option_type: "call" or "put"
        signal_score: Signal score (out of 7)
        message: The message that was sent
    
    Returns:
        True if logged successfully
    """
    _init_alert_table()
    
    conn = _get_alert_db()
    try:
        conn.execute(
            "INSERT INTO alert_history (ticker, strike, expiration, option_type, signal_score, sent_at, message) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ticker.upper(), strike, expiration, option_type, signal_score, 
             datetime.now().isoformat(), message)
        )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def get_alert_history(days: int = 7) -> List[Dict]:
    """
    Get history of sent alerts.
    
    Args:
        days: Number of days to look back
    
    Returns:
        List of alert dictionaries
    """
    _init_alert_table()
    
    conn = _get_alert_db()
    try:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        cursor = conn.execute(
            "SELECT * FROM alert_history WHERE sent_at >= ? ORDER BY sent_at DESC",
            (cutoff,)
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def send_twilio_sms(message: str) -> Dict:
    """
    Send an SMS message via Twilio.
    
    Args:
        message: The message to send
    
    Returns:
        Dictionary with success status and details
    """
    # Check if Twilio is configured
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, MY_PHONE_NUMBER]):
        return {
            "success": False,
            "error": "Twilio credentials not configured. Please check your .env file.",
            "simulated": True,
            "message": message
        }
    
    # Check for placeholder values
    if "your_" in TWILIO_ACCOUNT_SID or "your_" in TWILIO_AUTH_TOKEN:
        return {
            "success": False,
            "error": "Twilio credentials are placeholders. Please update your .env file.",
            "simulated": True,
            "message": message
        }
    
    try:
        from twilio.rest import Client
        
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        twilio_message = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=MY_PHONE_NUMBER
        )
        
        return {
            "success": True,
            "message_sid": twilio_message.sid,
            "status": twilio_message.status,
            "message": message
        }
    except ImportError:
        return {
            "success": False,
            "error": "Twilio library not installed. Run: pip install twilio",
            "simulated": True,
            "message": message
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "simulated": True,
            "message": message
        }


def send_trade_alert(ticker: str, strike: float, expiration: str,
                     option_type: str, signal_score: int, 
                     current_price: float = None) -> Dict:
    """
    Send a trade alert for a high-probability setup.
    Limits to one alert per ticker per trading day.
    
    Args:
        ticker: Stock symbol
        strike: Option strike price
        expiration: Expiration date
        option_type: "call" or "put"
        signal_score: Signal score (out of 7)
        current_price: Current stock price (optional)
    
    Returns:
        Dictionary with success status and details
    """
    # Check if already alerted today
    if was_alert_sent_today(ticker):
        return {
            "success": False,
            "error": f"Alert already sent for {ticker} today",
            "suppressed": True
        }
    
    # Check minimum score threshold (6 or 7 out of 7)
    if signal_score < 6:
        return {
            "success": False,
            "error": f"Score {signal_score} below threshold (6)",
            "suppressed": True
        }
    
    # Build message
    price_str = f"${current_price:.2f}" if current_price else "N/A"
    exp_display = datetime.strptime(expiration, "%Y-%m-%d").strftime("%b %d") if expiration else "N/A"
    
    message = (
        f"📈 TRADE ALERT\n"
        f"Ticker: {ticker}\n"
        f"Type: {option_type.upper()}\n"
        f"Strike: ${strike:.2f}\n"
        f"Exp: {exp_display}\n"
        f"Price: {price_str}\n"
        f"Score: {signal_score}/7"
    )
    
    # Send the alert
    result = send_twilio_sms(message)
    
    # Log if successful
    if result.get("success"):
        log_alert(ticker, strike, expiration, option_type, signal_score, message)
    
    return result


def send_scanner_summary(scanner_results: List[Dict]) -> Dict:
    """
    Send a summary of scanner results.
    
    Args:
        scanner_results: List of signal dictionaries from scanner
    
    Returns:
        Dictionary with success status
    """
    if not scanner_results:
        return {"success": False, "error": "No results to summarize"}
    
    # Filter to high-score signals only
    high_score = [r for r in scanner_results if r.get("score", 0) >= 6]
    
    if not high_score:
        return {"success": False, "error": "No high-score signals to report"}
    
    # Build summary message
    message = f"📊 SCANNER SUMMARY\n"
    message += f"{len(high_score)} HIGH-PROBABILITY SETUPS:\n\n"
    
    for result in high_score[:5]:  # Max 5 in one message
        message += (
            f"• {result['ticker']} {result['option_type'].upper()}\n"
            f"  Strike: ${result['strike']:.2f}\n"
            f"  Score: {result['score']}/7\n\n"
        )
    
    if len(high_score) > 5:
        message += f"...and {len(high_score) - 5} more\n"
    
    return send_twilio_sms(message)


def test_alert_config() -> Dict:
    """
    Test the Twilio configuration by sending a test message.
    
    Returns:
        Dictionary with test results
    """
    test_message = (
        f"🔔 TEST ALERT\n"
        f"Trading Dashboard alerts are configured correctly!\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    
    return send_twilio_sms(test_message)


def get_alert_stats() -> Dict:
    """
    Get statistics about sent alerts.
    
    Returns:
        Dictionary with alert statistics
    """
    _init_alert_table()
    
    conn = _get_alert_db()
    try:
        # Total alerts
        cursor = conn.execute("SELECT COUNT(*) as count FROM alert_history")
        total = cursor.fetchone()["count"]
        
        # Today's alerts
        today = datetime.now().strftime("%Y-%m-%d")
        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM alert_history WHERE DATE(sent_at) = ?",
            (today,)
        )
        today_count = cursor.fetchone()["count"]
        
        # Unique tickers alerted
        cursor = conn.execute("SELECT COUNT(DISTINCT ticker) as count FROM alert_history")
        unique_tickers = cursor.fetchone()["count"]
        
        # Last alert
        cursor = conn.execute(
            "SELECT * FROM alert_history ORDER BY sent_at DESC LIMIT 1"
        )
        last_alert = dict(cursor.fetchone()) if cursor.fetchone() else None
        
        return {
            "total_alerts": total,
            "today_alerts": today_count,
            "unique_tickers": unique_tickers,
            "last_alert": last_alert
        }
    finally:
        conn.close()


if __name__ == "__main__":
    print("Testing alerts.py module...")
    
    # Check configuration
    print("\nTwilio Configuration:")
    print(f"  Account SID configured: {'yes' if TWILIO_ACCOUNT_SID else 'no'}")
    print(f"  Auth Token configured: {'yes' if TWILIO_AUTH_TOKEN else 'no'}")
    print(f"  From Phone configured: {'yes' if TWILIO_PHONE_NUMBER else 'no'}")
    print(f"  To Phone configured: {'yes' if MY_PHONE_NUMBER else 'no'}")
    
    # Get alert stats
    stats = get_alert_stats()
    print(f"\nAlert Statistics:")
    print(f"  Total alerts: {stats['total_alerts']}")
    print(f"  Today's alerts: {stats['today_alerts']}")
    print(f"  Unique tickers: {stats['unique_tickers']}")
    
    # Test sending (simulated if not configured)
    print("\nTesting alert send...")
    result = test_alert_config()
    
    if result.get("success"):
        print("✓ Test alert sent successfully!")
    elif result.get("simulated"):
        print(f"✓ Simulated mode: {result['message']}")
    else:
        print(f"✗ Failed: {result.get('error')}")
    
    # Test trade alert (should be suppressed if score < 6)
    print("\nTesting trade alert (score 5 - should be suppressed)...")
    result = send_trade_alert("AAPL", 150.0, "2024-12-20", "call", 5)
    print(f"  Result: {result}")
    
    print("\nalerts.py module test complete!")
