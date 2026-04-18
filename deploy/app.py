"""
Trading Dashboard - Personal Options Trading Analysis Tool
Built with Streamlit, Plotly, and yfinance
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from plotly import graph_objects as go
import plotly.express as px

# Add modules to path
sys.path.insert(0, os.path.dirname(__file__))

from modules import data, indicators, gex, scanner, alerts, backtest

# Data directory - use temp on Streamlit Cloud
if os.path.exists("/mount"):
    DATA_DIR = "/tmp/trading_dashboard_data"
else:
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
    }
    .stDataFrame {
        background-color: #262730;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }
    .buy-signal {
        background-color: rgba(38, 166, 154, 0.2);
        color: #26a69a;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .watch-signal {
        background-color: rgba(255, 235, 59, 0.2);
        color: #ffeb3b;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .positive {
        color: #26a69a;
    }
    .negative {
        color: #ef5350;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        "watchlist": data.get_watchlist(),
        "paper_mode": True,
        "last_scan_time": None,
        "scan_results": pd.DataFrame()
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# Sidebar configuration
def render_sidebar():
    """Render sidebar with settings and watchlist management."""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Trading mode toggle
        st.session_state.paper_mode = st.toggle(
            "Paper Trading Mode",
            value=st.session_state.paper_mode,
            help="Toggle between paper trading and live tracking"
        )
        
        mode_label = "📝 Paper Trading" if st.session_state.paper_mode else "🔴 Live Mode"
        st.info(mode_label)
        
        st.divider()
        
        # Watchlist management
        st.header("📋 Watchlist")
        
        # Add ticker
        col1, col2 = st.columns([2, 1])
        with col1:
            new_ticker = st.text_input("Add ticker", placeholder="AAPL", key="new_ticker_input")
        with col2:
            if st.button("Add", key="add_ticker_btn"):
                if new_ticker:
                    if data.add_to_watchlist(new_ticker):
                        st.session_state.watchlist = data.get_watchlist()
                        st.success(f"Added {new_ticker.upper()}")
                        st.rerun()
                    else:
                        st.error("Could not add ticker")
        
        # Display watchlist
        if st.session_state.watchlist:
            st.write("**Your Tickers:**")
            for ticker in st.session_state.watchlist:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(f"• {ticker}")
                with col2:
                    if st.button("×", key=f"remove_{ticker}"):
                        data.remove_from_watchlist(ticker)
                        st.session_state.watchlist = data.get_watchlist()
                        st.rerun()
        else:
            st.info("Add tickers to your watchlist to get started")
        
        st.divider()
        
        # Alert configuration
        st.header("🔔 Alerts")
        if st.button("Test Alert Config", key="test_alert"):
            result = alerts.test_alert_config()
            if result.get("success"):
                st.success("Test alert sent!")
            elif result.get("simulated"):
                st.info("Simulated mode - alert not sent")
            else:
                st.error(result.get("error", "Failed"))
        
        alert_stats = alerts.get_alert_stats()
        st.caption(f"Alerts today: {alert_stats['today_alerts']}")
        st.caption(f"Total alerts: {alert_stats['total_alerts']}")
        
        st.divider()
        
        st.caption("Data cached in SQLite")
        
        if st.button("Clear Cache", key="clear_cache"):
            st.info("Cache cleared on next fetch")


render_sidebar()

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Market Overview",
    "📈 Chart View",
    "🎯 GEX Profile",
    "🔍 Options Scanner",
    "📝 Trade Journal",
    "🔄 Backtester"
])


# Tab 1: Market Overview
with tab1:
    st.header("Market Overview")
    
    if not st.session_state.watchlist:
        st.info("Add tickers to your watchlist to see market overview")
    else:
        # Refresh button
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("🔄 Refresh Prices", key="refresh_overview"):
                st.cache_data.clear()
        
        with col2:
            try:
                overview_df = data.get_market_overview(st.session_state.watchlist, force_refresh=True)
                
                # Format display
                def format_change(val):
                    if pd.isna(val):
                        return "N/A"
                    sign = "+" if val > 0 else ""
                    return f"{sign}{val:.2f}%"
                
                overview_df["Change %"] = overview_df["Change %"].apply(format_change)
                
                # Color code changes
                def color_change(val):
                    if isinstance(val, str) and "+" in val:
                        return "color: #26a69a"
                    elif isinstance(val, str) and "%" in val and "-" in val:
                        return "color: #ef5350"
                    return ""
                
                st.dataframe(
                    overview_df.style.applymap(color_change, subset=["Change %"]),
                    use_container_width=True,
                    hide_index=True
                )
            except Exception as e:
                st.error(f"Error loading market data: {e}")
        
        # Daily movers section
        st.divider()
        st.subheader("Top Movers")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Top Gainers**")
            try:
                gainers = data.get_market_overview(st.session_state.watchlist)
                gainers = gainers.dropna(subset=["Change"])
                gainers = gainers.nlargest(3, "Change")
                for _, row in gainers.iterrows():
                    st.metric(row["Ticker"], f"${row['Price']:.2f}", f"+{row['Change']:.2f}")
            except:
                st.write("No data")
        
        with col2:
            st.write("**Top Losers**")
            try:
                losers = data.get_market_overview(st.session_state.watchlist)
                losers = losers.dropna(subset=["Change"])
                losers = losers.nsmallest(3, "Change")
                for _, row in losers.iterrows():
                    st.metric(row["Ticker"], f"${row['Price']:.2f}", f"{row['Change']:.2f}")
            except:
                st.write("No data")


# Tab 2: Chart View
with tab2:
    st.header("Technical Analysis Chart")
    
    # Ticker selection
    col1, col2 = st.columns([1, 3])
    with col1:
        chart_ticker = st.selectbox(
            "Select Ticker",
            st.session_state.watchlist if st.session_state.watchlist else ["AAPL"],
            key="chart_ticker"
        )
    
    with col2:
        period_options = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y"}
        selected_period = st.selectbox("Time Period", list(period_options.keys()), index=3)
    
    # Indicator toggles
    st.subheader("Indicators")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_sma = st.checkbox("SMA (20, 50, 200)", value=True)
    with col2:
        show_ema = st.checkbox("EMA (9, 21)", value=True)
    with col3:
        show_vwap = st.checkbox("VWAP", value=True)
    with col4:
        show_bb = st.checkbox("Bollinger Bands", value=True)
    
    try:
        # Load data and calculate indicators
        df = data.get_ohlcv(chart_ticker, period=period_options[selected_period])
        ind = indicators.calculate_all_indicators(df, chart_ticker)
        
        # Create main chart
        fig = indicators.create_candlestick_chart(
            df, chart_ticker, ind,
            show_sma=show_sma,
            show_ema=show_ema,
            show_vwap=show_vwap,
            show_bb=show_bb
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # RSI and MACD charts
        col1, col2 = st.columns(2)
        with col1:
            rsi_fig = indicators.create_rsi_chart(df, ind)
            st.plotly_chart(rsi_fig, use_container_width=True)
        
        with col2:
            macd_fig = indicators.create_macd_chart(df, ind)
            st.plotly_chart(macd_fig, use_container_width=True)
        
        # Signal summary
        st.divider()
        st.subheader("Signal Summary")
        
        signals = indicators.scan_for_signals(chart_ticker, df)
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            rsi_val = signals["rsi"]
            rsi_status, rsi_color = indicators.get_signal_status(rsi_val)
            st.metric("RSI (14)", f"{rsi_val:.1f}", rsi_status)
        
        with col2:
            st.metric("Price", f"${signals['current_price']:.2f}")
        
        with col3:
            iv_rank = signals.get("iv_rank", 0)
            st.metric("IV Rank", f"{iv_rank:.1f}%")
        
        with col4:
            st.metric("SMA 50", f"${signals['sma_50']:.2f}")
        
        with col5:
            st.metric("EMA 21", f"${signals['ema_21']:.2f}")
        
        with col6:
            gc = signals.get("golden_cross_date")
            dc = signals.get("death_cross_date")
            if gc:
                st.success(f"✅ Golden Cross: {gc}")
            elif dc:
                st.error(f"❌ Death Cross: {dc}")
            else:
                st.info("No crossover")
        
        # Signal checklist
        with st.expander("Signal Details"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Price Signals:**")
                st.write(f"✓ Price above EMA21: {signals['price_above_ema21']}")
                st.write(f"✓ Price above SMA50: {signals['price_above_sma50']}")
            
            with col2:
                st.write("**Momentum Signals:**")
                st.write(f"✓ RSI neutral (40-60): {signals['rsi_neutral']}")
                st.write(f"✓ MACD positive: {signals['macd_positive']}")
                st.write(f"✓ MACD turning positive: {signals['macd_turning_positive']}")
    
    except Exception as e:
        st.error(f"Error loading chart data: {e}")


# Tab 3: GEX Profile
with tab3:
    st.header("Gamma Exposure Profile")
    
    # Ticker and expiration selection
    col1, col2 = st.columns([1, 2])
    with col1:
        gex_ticker = st.selectbox(
            "Select Ticker",
            st.session_state.watchlist if st.session_state.watchlist else ["SPY"],
            key="gex_ticker"
        )
    
    with col2:
        try:
            _, _, expirations = data.get_options_chain(gex_ticker)
            exp_options = {exp: datetime.strptime(exp, "%Y-%m-%d").strftime("%b %d, %Y") 
                          for exp in expirations[:5]}
            selected_exp = st.selectbox("Expiration", list(exp_options.keys()))
        except Exception as e:
            st.error(f"Could not load expirations: {e}")
            selected_exp = None
    
    if selected_exp:
        try:
            gex_data = gex.calculate_gex_profile(gex_ticker, selected_exp)
            
            if "error" in gex_data:
                st.error(gex_data["error"])
            else:
                # GEX Chart
                gex_fig = gex.create_gex_chart(gex_data)
                st.plotly_chart(gex_fig, use_container_width=True)
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${gex_data['current_price']:.2f}")
                with col2:
                    st.metric("Max Pain", f"${gex_data['max_pain']:.2f}")
                with col3:
                    gex_val = gex_data["net_gex"]
                    gex_status = "Positive" if gex_val >= 0 else "Negative"
                    st.metric("Net GEX", f"${gex_val:,.0f}", gex_status)
                with col4:
                    flip = gex_data.get("gamma_flip")
                    flip_str = f"${flip:.2f}" if flip else "N/A"
                    st.metric("Gamma Flip", flip_str)
                
                # Detailed GEX table
                st.divider()
                st.subheader("Gamma by Strike")
                
                if gex_data.get("gamma_by_strike"):
                    gex_df = pd.DataFrame(gex_data["gamma_by_strike"])
                    gex_df = gex_df.sort_values("strike")
                    
                    # Filter to strikes near current price
                    current = gex_data["current_price"]
                    gex_df = gex_df[
                        (gex_df["strike"] >= current * 0.9) & 
                        (gex_df["strike"] <= current * 1.1)
                    ]
                    
                    st.dataframe(
                        gex_df.style.background_gradient(subset=["gamma_exposure"]),
                        use_container_width=True,
                        hide_index=True
                    )
        
        except Exception as e:
            st.error(f"Error calculating GEX: {e}")


# Tab 4: Options Scanner
with tab4:
    st.header("Options Buy Signal Scanner")
    
    # Scanner settings
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        min_score = st.slider("Minimum Score", 4, 7, 5)
    with col2:
        days_to_exp = st.slider("Days to Expiry", 7, 60, 30)
    with col3:
        show_all = st.checkbox("Show all signals", value=True)
    
    # Scan button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🔍 Run Scanner", key="run_scanner", type="primary"):
            with st.spinner("Scanning watchlist..."):
                try:
                    results = scanner.scan_watchlist(
                        st.session_state.watchlist,
                        min_score=min_score if not show_all else 4,
                        days_to_expiry=days_to_exp
                    )
                    st.session_state.scan_results = results
                    st.session_state.last_scan_time = datetime.now()
                except Exception as e:
                    st.error(f"Scanner error: {e}")
    
    # Display last scan time
    if st.session_state.last_scan_time:
        st.caption(f"Last scan: {st.session_state.last_scan_time.strftime('%H:%M:%S')}")
    
    # Display results
    if not st.session_state.scan_results.empty:
        results = st.session_state.scan_results
        
        # Summary
        summary = scanner.get_scan_summary(results)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Signals", summary["total_signals"])
        with col2:
            st.metric("Buy Signals", summary["buy_signals"], delta_color="normal")
        with col3:
            st.metric("Watch Signals", summary["watch_signals"])
        with col4:
            st.metric("Avg Score", f"{summary['avg_score']:.1f}")
        
        st.divider()
        
        # Filter options
        filter_col1, filter_col2 = st.columns([1, 1])
        with filter_col1:
            option_filter = st.multiselect(
                "Filter by Type",
                options=["call", "put"],
                default=["call", "put"]
            )
        with filter_col2:
            score_filter = st.slider("Min Score", 4, 7, min_score)
        
        # Apply filters
        filtered = results[results["score"] >= score_filter]
        if option_filter:
            filtered = filtered[filtered["option_type"].isin(option_filter)]
        
        # Sort options
        sort_col, sort_dir = st.columns([1, 1])
        with sort_col:
            sort_by = st.selectbox("Sort by", ["score", "iv_rank", "rsi", "open_interest"])
        with sort_dir:
            ascending = st.checkbox("Ascending")
        
        filtered = filtered.sort_values(sort_by, ascending=ascending)
        
        # Display table
        st.subheader("Signal Results")
        
        display_cols = [
            "ticker", "current_price", "strike", "expiration",
            "option_type", "score", "iv_rank", "rsi", "gex_value", "open_interest"
        ]
        
        display_df = filtered[display_cols].copy()
        display_df.columns = [
            "Ticker", "Price", "Strike", "Expiration", "Type",
            "Score", "IV Rank", "RSI", "GEX", "Open Int"
        ]
        
        # Color code score
        def score_color(score):
            if score >= 6:
                return "background-color: rgba(38, 166, 154, 0.3)"
            elif score >= 5:
                return "background-color: rgba(255, 235, 59, 0.3)"
            return ""
        
        st.dataframe(
            display_df.style.applymap(score_color, subset=["Score"]),
            use_container_width=True,
            hide_index=True
        )
        
        # Alert button for high-scoring signals
        st.divider()
        st.subheader("Send Alerts")
        
        high_signals = filtered[filtered["score"] >= 6]
        if not high_signals.empty:
            for _, signal in high_signals.iterrows():
                col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
                with col1:
                    st.write(f"**{signal['ticker']}** {signal['option_type'].upper()} ${signal['strike']:.2f}")
                with col2:
                    st.write(f"Score: {signal['score']}/7")
                with col3:
                    st.write(f"IV: {signal['iv_rank']:.0f}%")
                with col4:
                    st.write(f"GEX: ${signal['gex_value']:,.0f}")
                with col5:
                    if st.button("📱 Alert", key=f"alert_{signal['ticker']}_{signal['strike']}"):
                        result = alerts.send_trade_alert(
                            signal["ticker"],
                            signal["strike"],
                            signal["expiration"],
                            signal["option_type"],
                            signal["score"],
                            signal["current_price"]
                        )
                        if result.get("success"):
                            st.success("Alert sent!")
                        elif result.get("suppressed"):
                            st.info(result.get("error", "Alert suppressed"))
                        else:
                            st.error(result.get("error", "Failed to send"))
        else:
            st.info("No signals with score 6+ to alert")
    
    else:
        st.info("Click 'Run Scanner' to analyze your watchlist for trade setups")


# Tab 5: Trade Journal
with tab5:
    st.header("Trade Journal")
    
    # Trade logging form
    with st.expander("➕ Log New Trade", expanded=False):
        with st.form("trade_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                trade_date = st.date_input("Date", datetime.now())
                trade_ticker = st.text_input("Ticker", placeholder="AAPL")
            with col2:
                strike = st.number_input("Strike Price", min_value=0.0, step=0.5, format="%.2f")
                option_type = st.selectbox("Type", ["call", "put"])
            with col3:
                expiration = st.date_input("Expiration")
                quantity = st.number_input("Quantity", min_value=1, step=1, value=1)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                entry_price = st.number_input("Entry Price", min_value=0.0, step=0.01, format="%.2f")
            with col2:
                exit_price = st.number_input("Exit Price (leave 0 if open)", min_value=0.0, step=0.01, format="%.2f", value=0.0)
            with col3:
                notes = st.text_area("Notes")
            
            submitted = st.form_submit_button("Log Trade")
            if submitted:
                if trade_ticker and strike > 0:
                    # Calculate P&L
                    multiplier = 100
                    pnl = (exit_price - entry_price) * quantity * multiplier if exit_price > 0 else 0
                    
                    # Log to CSV
                    trade_log_path = os.path.join(DATA_DIR, "trades_log.csv")
                    
                    trade_record = pd.DataFrame([{
                        "date": trade_date.strftime("%Y-%m-%d"),
                        "ticker": trade_ticker.upper(),
                        "strike": strike,
                        "expiration": expiration.strftime("%Y-%m-%d"),
                        "option_type": option_type,
                        "quantity": quantity,
                        "entry_price": entry_price,
                        "exit_price": exit_price if exit_price > 0 else "",
                        "pnl": pnl if exit_price > 0 else "",
                        "notes": notes
                    }])
                    
                    if os.path.exists(trade_log_path):
                        trade_record.to_csv(trade_log_path, mode="a", header=False, index=False)
                    else:
                        trade_record.to_csv(trade_log_path, mode="w", header=True, index=False)
                    
                    st.success("Trade logged!")
                else:
                    st.error("Please fill in required fields")
    
    # Load and display trade journal
    trade_log_path = os.path.join(DATA_DIR, "trades_log.csv")
    
    if os.path.exists(trade_log_path):
        try:
            journal_df = pd.read_csv(trade_log_path)
            
            if not journal_df.empty:
                # Filters
                col1, col2 = st.columns([1, 1])
                with col1:
                    ticker_filter = st.multiselect(
                        "Filter by Ticker",
                        journal_df["ticker"].unique(),
                        default=journal_df["ticker"].unique()[:3]
                    )
                with col2:
                    date_range = st.date_input(
                        "Date Range",
                        value=(datetime.now() - timedelta(days=30), datetime.now())
                    )
                
                # Apply filters
                filtered_journal = journal_df[journal_df["ticker"].isin(ticker_filter)]
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    filtered_journal = filtered_journal[
                        (pd.to_datetime(filtered_journal["date"]) >= pd.to_datetime(start_date)) &
                        (pd.to_datetime(filtered_journal["date"]) <= pd.to_datetime(end_date))
                    ]
                
                # Summary
                col1, col2, col3, col4 = st.columns(4)
                
                total_trades = len(filtered_journal)
                closed_trades = filtered_journal[filtered_journal["pnl"].notna()]
                total_pnl = closed_trades["pnl"].sum() if not closed_trades.empty else 0
                win_rate = (closed_trades["pnl"] > 0).sum() / len(closed_trades) * 100 if not closed_trades.empty else 0
                
                with col1:
                    st.metric("Total Trades", total_trades)
                with col2:
                    st.metric("Closed Trades", len(closed_trades))
                with col3:
                    st.metric("Total P&L", f"${total_pnl:.2f}", delta_color="normal")
                with col4:
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                
                st.divider()
                
                # Journal table
                st.dataframe(
                    filtered_journal.sort_values("date", ascending=False),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Export button
                csv = filtered_journal.to_csv(index=False)
                st.download_button(
                    "📥 Export CSV",
                    csv,
                    f"trade_journal_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
            else:
                st.info("No trades logged yet")
        except Exception as e:
            st.error(f"Error loading trade journal: {e}")
    else:
        st.info("No trades logged yet. Add your first trade above!")
    
    # Paper trading section
    st.divider()
    st.subheader("Paper Trading Account")
    
    paper_account = data.get_paper_account()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Open P&L", f"${paper_account['open_pnl']:.2f}", delta_color="normal")
    with col2:
        st.metric("Closed P&L", f"${paper_account['closed_pnl']:.2f}", delta_color="normal")
    with col3:
        st.metric("Total P&L", f"${paper_account['total_pnl']:.2f}", delta_color="normal")
    
    # Paper trade form
    with st.expander("➕ Open Paper Trade"):
        with st.form("paper_trade_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                pt_ticker = st.text_input("Ticker", key="pt_ticker")
            with col2:
                pt_strike = st.number_input("Strike", min_value=0.0, step=0.5, format="%.2f")
                pt_type = st.selectbox("Type", ["call", "put"], key="pt_type")
            with col3:
                pt_exp = st.date_input("Expiration", key="pt_exp")
                pt_qty = st.number_input("Quantity", min_value=1, value=1, key="pt_qty")
            
            pt_entry = st.number_input("Entry Price", min_value=0.0, step=0.01, format="%.2f", key="pt_entry")
            
            if st.form_submit_button("Open Paper Trade"):
                if pt_ticker and pt_strike > 0 and pt_entry > 0:
                    if data.add_paper_trade(pt_ticker, pt_strike, pt_exp.strftime("%Y-%m-%d"), pt_type, pt_qty, pt_entry):
                        st.success("Paper trade opened!")
                    else:
                        st.error("Failed to open trade")
                else:
                    st.error("Please fill all fields")


# Tab 6: Backtester
with tab6:
    st.header("Strategy Backtester")
    
    # Backtest configuration
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        bt_ticker = st.selectbox(
            "Ticker",
            st.session_state.watchlist if st.session_state.watchlist else ["AAPL"],
            key="bt_ticker"
        )
    with col2:
        strategy = st.selectbox("Strategy", ["standard", "aggressive", "conservative"])
    with col3:
        option_type = st.selectbox("Option Type", ["call", "put"])
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    with col3:
        holding_days = st.slider("Holding Period (days)", 1, 30, 5)
    
    initial_capital = st.number_input("Initial Capital", min_value=1000, value=10000, step=1000)
    
    if st.button("🚀 Run Backtest", type="primary", key="run_backtest"):
        with st.spinner("Running backtest..."):
            try:
                result = backtest.backtest_strategy(
                    bt_ticker, 
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                    strategy,
                    option_type,
                    holding_days,
                    initial_capital
                )
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    # Display metrics
                    metrics = result["metrics"]
                    
                    st.divider()
                    st.subheader("Performance Metrics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        delta_color = "normal" if metrics["total_return"] >= 0 else "inverse"
                        st.metric(
                            "Total Return", 
                            f"${metrics['total_return']:.2f}", 
                            f"{metrics['total_return_pct']:.2f}%",
                            delta_color=delta_color
                        )
                    with col2:
                        st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
                    with col3:
                        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                    with col4:
                        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Trades", metrics["num_trades"])
                    with col2:
                        st.metric("Win/Loss", f"{metrics['winning_trades']}/{metrics['losing_trades']}")
                    with col3:
                        st.metric("Avg Win", f"${metrics['avg_win']:.2f}")
                    with col4:
                        st.metric("Avg Loss", f"${metrics['avg_loss']:.2f}")
                    
                    # Equity curve
                    if result.get("capital_history"):
                        equity_df = backtest.create_equity_curve(
                            result["trades"],
                            [h[0] for h in result["capital_history"]],
                            initial_capital
                        )
                        
                        st.divider()
                        st.subheader("Equity Curve")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=equity_df["date"],
                            y=equity_df["equity"],
                            mode="lines",
                            name="Equity",
                            line=dict(color="#26a69a", width=2)
                        ))
                        
                        fig.update_layout(
                            template="plotly_dark",
                            height=400,
                            xaxis_title="Date",
                            yaxis_title="Account Value ($)"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Trade log
                    if result.get("trades"):
                        st.divider()
                        st.subheader("Trade Log")
                        
                        trades_df = pd.DataFrame(result["trades"])
                        st.dataframe(
                            trades_df.sort_values("entry_date", ascending=False),
                            use_container_width=True,
                            hide_index=True
                        )
            
            except Exception as e:
                st.error(f"Backtest error: {e}")
    
    # Strategy comparison
    st.divider()
    st.subheader("Compare Strategies")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        compare_ticker = st.selectbox(
            "Compare Ticker",
            st.session_state.watchlist if st.session_state.watchlist else ["SPY"],
            key="compare_ticker"
        )
    
    if st.button("Compare", key="compare_strategies"):
        with st.spinner("Comparing strategies..."):
            try:
                comparison = backtest.compare_strategies(
                    compare_ticker,
                    (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d"),
                    datetime.now().strftime("%Y-%m-%d")
                )
                
                if not comparison.empty:
                    st.dataframe(
                        comparison.set_index("Strategy"),
                        use_container_width=True
                    )
                else:
                    st.info("No comparison data available")
            except Exception as e:
                st.error(f"Comparison error: {e}")


# Footer
st.divider()
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Trading Dashboard | Data from Yahoo Finance | Not financial advice"
    "</div>",
    unsafe_allow_html=True
)


if __name__ == "__main__":
    st.run()
