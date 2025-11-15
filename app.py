"""
Interactive Web Interface for DJIA Mean-Reversion Trading Strategy
Built with Streamlit for non-technical users
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path
import os
import glob

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Page configuration
st.set_page_config(
    page_title="DJIA Mean-Reversion Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'trades_df' not in st.session_state:
    st.session_state.trades_df = None
if 'strategy_metrics' not in st.session_state:
    st.session_state.strategy_metrics = None
if 'dia_metrics' not in st.session_state:
    st.session_state.dia_metrics = None

def run_backtest(start_date, end_date, top_k, trading_cost_bps):
    """Run the backtest and return results"""
    try:
        # Import modules
        from data_loader import fetch_djia_data, load_saved_data, check_cached_data
        from signals import compute_past_return, compute_reversal_score
        from portfolio import generate_rebalance_dates, construct_weights
        from backtester import backtest
        from performance import compute_all_metrics, compute_rolling_12m_return
        from run_mean_reversion import create_benchmark_dia
        
        # Convert date objects to strings if needed
        if hasattr(start_date, 'strftime'):
            start_date = start_date.strftime('%Y-%m-%d')
        if hasattr(end_date, 'strftime'):
            end_date = end_date.strftime('%Y-%m-%d')
        
        # Find all available cache files
        cache_dir = Path('data')
        if not cache_dir.exists():
            cache_dir.mkdir(exist_ok=True)
        
        # Get all cache files
        all_cache_files = []
        for pattern in ['djia_prices*.pkl', 'djia_prices*.pickle']:
            all_cache_files.extend(list(cache_dir.glob(pattern)))
        
        # Also check specific known cache files
        known_caches = [
            'data/djia_prices_20151115_20251114.pkl',
            'data/djia_prices_20151115_20251115.pkl',
            'data/djia_prices.pkl'
        ]
        for cache_path in known_caches:
            cache_file = Path(cache_path)
            if cache_file.exists() and cache_file not in all_cache_files:
                all_cache_files.append(cache_file)
        
        # Resolve symlinks
        resolved_files = []
        for cache_file in all_cache_files:
            try:
                if cache_file.is_symlink():
                    resolved = cache_file.resolve()
                    if resolved.exists() and resolved not in resolved_files:
                        resolved_files.append(resolved)
                else:
                    resolved_files.append(cache_file)
            except:
                if cache_file.exists():
                    resolved_files.append(cache_file)
        
        all_cache_files = list(set(resolved_files))  # Remove duplicates
        
        prices_df = None
        data_path = None
        cache_used = False
        
        # Debug: Show what cache files we found
        if all_cache_files:
            with st.expander("🔍 Debug: Cache Files Found", expanded=False):
                st.write(f"Found {len(all_cache_files)} cache file(s):")
                for cf in all_cache_files:
                    st.write(f"  - {cf}")
        
        # Try to find a cache file that covers the requested range
        # Sort by modification time (newest first)
        all_cache_files.sort(key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True)
        
        for cache_file in all_cache_files:
            cache_path = str(cache_file)
            if os.path.exists(cache_path) and os.path.isfile(cache_path):
                try:
                    # Check if cache covers the range
                    cache_valid = check_cached_data(cache_path, start_date, end_date)
                    if cache_valid:
                        st.success(f"📦 Using cached data from {cache_file.name}")
                        prices_df = load_saved_data(cache_path)
                        # Filter to requested date range
                        if not prices_df.empty:
                            start_dt = pd.to_datetime(start_date)
                            end_dt = pd.to_datetime(end_date)
                            # Only filter if needed
                            if prices_df.index[0] > start_dt or prices_df.index[-1] < end_dt:
                                prices_df = prices_df.loc[start_dt:end_dt]
                            if not prices_df.empty and len(prices_df) > 0:
                                data_path = cache_path
                                cache_used = True
                                break
                except Exception as e:
                    with st.expander("⚠️ Cache Check Error", expanded=False):
                        st.write(f"Error checking {cache_file.name}: {str(e)}")
                    continue
        
        # If no cache found, try to fetch (but warn user)
        if prices_df is None or prices_df.empty:
            st.error("❌ No cached data found that covers the requested date range!")
            st.warning("⚠️ Attempting to fetch from Yahoo Finance (this may take a while and may hit rate limits)...")
            st.info("💡 Tip: Use dates 2015-11-15 to 2025-11-14 to use existing cached data")
            
            # Don't stop - just try to fetch
            cache_suffix = f"{start_date.replace('-', '')}_{end_date.replace('-', '')}"
            data_path = f'data/djia_prices_{cache_suffix}.pkl'
            try:
                prices_df = fetch_djia_data(start_date=start_date, end_date=end_date, save_path=data_path, batch_size=5)
            except Exception as fetch_error:
                st.error(f"❌ Failed to fetch data: {str(fetch_error)}")
                st.error("Please use dates 2015-11-15 to 2025-11-14 to use cached data, or try again later.")
                return None
        
        # Find DIA cache file (similar to DJIA cache logic)
        dia_cache_dir = Path('data')
        dia_cache_files = list(dia_cache_dir.glob('dia_prices*.pkl'))
        
        # Also check specific known cache files
        known_dia_caches = [
            'data/dia_prices_20151115_20251114.pkl',
            'data/dia_prices_20151115_20251115.pkl',
        ]
        for cache_path in known_dia_caches:
            cache_file = Path(cache_path)
            if cache_file.exists() and cache_file not in dia_cache_files:
                dia_cache_files.append(cache_file)
        
        # Resolve symlinks
        resolved_dia_files = []
        for cache_file in dia_cache_files:
            try:
                if cache_file.is_symlink():
                    resolved = cache_file.resolve()
                    if resolved.exists() and resolved not in resolved_dia_files:
                        resolved_dia_files.append(resolved)
                else:
                    resolved_dia_files.append(cache_file)
            except:
                if cache_file.exists():
                    resolved_dia_files.append(cache_file)
        
        dia_cache_files = list(set(resolved_dia_files))
        dia_cache_files.sort(key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True)
        
        # Use first valid DIA cache, or create new path
        dia_cache_path = None
        for cache_file in dia_cache_files:
            cache_path = str(cache_file)
            if os.path.exists(cache_path) and os.path.isfile(cache_path):
                if check_cached_data(cache_path, start_date, end_date):
                    dia_cache_path = cache_path
                    break
        
        # If no valid cache found, use date-based path
        if dia_cache_path is None:
            dia_cache_path = f'data/dia_prices_{start_date.replace("-", "")}_{end_date.replace("-", "")}.pkl'
        
        # Generate signals
        LOOKBACK_DAYS = 21  # Number of days to look back for returns
        past_returns = compute_past_return(prices_df, lookback_days=LOOKBACK_DAYS)
        reversal_scores = compute_reversal_score(past_returns)
        
        # Construct portfolio
        rebalance_dates = generate_rebalance_dates(prices_df, frequency='daily')
        weights_df = construct_weights(reversal_scores, rebalance_dates, top_k=top_k, daily_rebalance=True)
        
        # Run backtest
        results = backtest(prices_df, weights_df, trading_cost_bps=trading_cost_bps)
        
        # Compute metrics
        strategy_metrics = compute_all_metrics(
            results['net_return'],
            results['equity'],
            results['turnover']
        )
        
        # Get DIA benchmark
        benchmark_dia_equity = None
        try:
            # Try to find and use cached DIA data
            dia_cache_found = False
            for cache_file in dia_cache_files:
                cache_path = str(cache_file)
                if os.path.exists(cache_path) and os.path.isfile(cache_path):
                    if check_cached_data(cache_path, start_date, end_date):
                        st.info(f"📦 Using cached DIA data from {cache_file.name}")
                        benchmark_dia_equity = create_benchmark_dia(start_date, end_date, cache_path=cache_path)
                        dia_cache_found = True
                        break
            
            # If no cache found, try to create/fetch (but only if we have prices_df)
            if not dia_cache_found and prices_df is not None and not prices_df.empty:
                # Use the date-based cache path
                benchmark_dia_equity = create_benchmark_dia(start_date, end_date, cache_path=dia_cache_path)
            
            # Process DIA equity if we have it
            if benchmark_dia_equity is not None and not benchmark_dia_equity.empty:
                common_dates = results.index.intersection(benchmark_dia_equity.index)
                benchmark_dia_equity = benchmark_dia_equity.loc[common_dates]
                if len(benchmark_dia_equity) > 0:
                    benchmark_dia_equity = benchmark_dia_equity / benchmark_dia_equity.iloc[0]
                    dia_returns = benchmark_dia_equity.pct_change().dropna()
                    dia_metrics = compute_all_metrics(
                        dia_returns,
                        benchmark_dia_equity,
                        pd.Series(0.0, index=dia_returns.index)
                    )
                else:
                    dia_metrics = None
            else:
                dia_metrics = None
        except Exception as e:
            st.warning(f"⚠️ Could not load DIA benchmark: {str(e)}")
            dia_metrics = None
            benchmark_dia_equity = None
        
        # Create trades DataFrame
        trades_list = []
        common_dates = prices_df.index.intersection(weights_df.index)
        prices_aligned = prices_df.loc[common_dates]
        weights_aligned = weights_df.loc[common_dates]
        
        for i, date in enumerate(weights_aligned.index):
            day_weights = weights_aligned.loc[date]
            selected_tickers = day_weights[day_weights > 0].index.tolist()
            
            if i > 0:
                prev_date = weights_aligned.index[i - 1]
                prev_weights = weights_aligned.loc[prev_date]
                prev_tickers = prev_weights[prev_weights > 0].index.tolist()
                
                for ticker in prev_tickers:
                    prev_weight = prev_weights[ticker]
                    sell_price = prices_aligned.loc[date, ticker]
                    trades_list.append({
                        'Date': date,
                        'Action': 'SELL',
                        'Ticker': ticker,
                        'Weight': prev_weight,
                        'Price': sell_price
                    })
            
            for ticker in selected_tickers:
                weight = day_weights[ticker]
                buy_price = prices_aligned.loc[date, ticker]
                trades_list.append({
                    'Date': date,
                    'Action': 'BUY',
                    'Ticker': ticker,
                    'Weight': weight,
                    'Price': buy_price
                })
        
        trades_df = pd.DataFrame(trades_list)
        trades_df = trades_df.sort_values(['Date', 'Action', 'Ticker'])
        
        return {
            'results': results,
            'trades_df': trades_df,
            'strategy_metrics': strategy_metrics,
            'dia_metrics': dia_metrics,
            'benchmark_dia_equity': benchmark_dia_equity if (benchmark_dia_equity is not None and not benchmark_dia_equity.empty) else None,
            'prices_df': prices_df,
            'weights_df': weights_df
        }
    except Exception as e:
        st.error(f"Error running backtest: {str(e)}")
        return None

# Main UI
st.markdown('<h1 class="main-header">📈 DJIA Mean-Reversion Trading Strategy</h1>', unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Strategy Configuration")
    
    # Date range - default to cached data range
    st.subheader("Date Range")
    # Default to dates that match cached data (2015-11-15 to 2025-11-14)
    default_end = datetime(2025, 11, 14)  # Yesterday from cache
    default_start = datetime(2015, 11, 15)  # Start of cache
    
    st.info("💡 **Tip**: Use default dates (2015-11-15 to 2025-11-14) to use cached data instantly!")
    
    start_date = st.date_input(
        "Start Date",
        value=default_start,
        max_value=default_end,
        help="Select the start date for the backtest"
    )
    
    end_date = st.date_input(
        "End Date",
        value=default_end,
        max_value=datetime.now(),
        min_value=start_date,
        help="Select the end date for the backtest"
    )
    
    st.divider()
    
    # Strategy parameters
    st.subheader("Strategy Parameters")
    top_k = st.slider(
        "Number of Stocks to Select",
        min_value=1,
        max_value=30,
        value=10,
        help="Select the top K worst performing stocks each day"
    )
    
    trading_cost_bps = st.number_input(
        "Trading Cost (basis points)",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5,
        help="Trading cost per unit of turnover in basis points"
    )
    
    st.divider()
    
    # Run button
    run_button = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    **Strategy Description:**
    - Daily mean-reversion strategy on DJIA stocks
    - Selects worst performing stocks (21-day return)
    - Buys at close, sells next day at close
    - Equal-weighted positions
    """)

# Main content area
if run_button or st.session_state.backtest_results is not None:
    if run_button:
        with st.spinner("Running backtest... This may take a minute."):
            results = run_backtest(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                top_k,
                trading_cost_bps
            )
            
            if results:
                st.session_state.backtest_results = results
                st.session_state.trades_df = results['trades_df']
                st.session_state.strategy_metrics = results['strategy_metrics']
                st.session_state.dia_metrics = results['dia_metrics']
                st.success("✅ Backtest completed successfully!")
            else:
                st.error("❌ Backtest failed. Please check the configuration.")
    
    if st.session_state.backtest_results:
        results = st.session_state.backtest_results
        strategy_metrics = st.session_state.strategy_metrics
        dia_metrics = st.session_state.dia_metrics
        
        # Performance Metrics
        st.header("📊 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "CAGR",
                f"{strategy_metrics.get('CAGR', 0):.2%}",
                delta=f"{strategy_metrics.get('CAGR', 0) - (dia_metrics.get('CAGR', 0) if dia_metrics else 0):.2%}" if dia_metrics else None,
                delta_color="normal",
                help="Compound Annual Growth Rate"
            )
        
        with col2:
            st.metric(
                "Sharpe Ratio",
                f"{strategy_metrics.get('Sharpe Ratio', 0):.2f}",
                delta=f"{strategy_metrics.get('Sharpe Ratio', 0) - (dia_metrics.get('Sharpe Ratio', 0) if dia_metrics else 0):.2f}" if dia_metrics else None,
                delta_color="normal",
                help="Risk-adjusted return measure"
            )
        
        with col3:
            st.metric(
                "Max Drawdown",
                f"{strategy_metrics.get('Max Drawdown', 0):.2%}",
                delta=f"{strategy_metrics.get('Max Drawdown', 0) - (dia_metrics.get('Max Drawdown', 0) if dia_metrics else 0):.2%}" if dia_metrics else None,
                delta_color="inverse",
                help="Maximum peak-to-trough decline"
            )
        
        with col4:
            st.metric(
                "Volatility",
                f"{strategy_metrics.get('Annualized Volatility', 0):.2%}",
                delta=f"{strategy_metrics.get('Annualized Volatility', 0) - (dia_metrics.get('Annualized Volatility', 0) if dia_metrics else 0):.2%}" if dia_metrics else None,
                delta_color="inverse",
                help="Annualized volatility"
            )
        
        # Detailed metrics table
        st.subheader("Detailed Metrics")
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.write("**Strategy Metrics**")
            strategy_metrics_df = pd.DataFrame([
                {"Metric": "CAGR", "Value": f"{strategy_metrics.get('CAGR', 0):.2%}"},
                {"Metric": "Annualized Volatility", "Value": f"{strategy_metrics.get('Annualized Volatility', 0):.2%}"},
                {"Metric": "Sharpe Ratio", "Value": f"{strategy_metrics.get('Sharpe Ratio', 0):.2f}"},
                {"Metric": "Max Drawdown", "Value": f"{strategy_metrics.get('Max Drawdown', 0):.2%}"},
                {"Metric": "Calmar Ratio", "Value": f"{strategy_metrics.get('Calmar Ratio', 0):.2f}"},
                {"Metric": "Hit Rate", "Value": f"{strategy_metrics.get('Hit Rate', 0):.2%}"},
                {"Metric": "Average Turnover", "Value": f"{strategy_metrics.get('Average Turnover', 0):.4f}"},
            ])
            st.dataframe(strategy_metrics_df, use_container_width=True, hide_index=True)
        
        with metrics_col2:
            if dia_metrics:
                st.write("**DIA ETF (Benchmark) Metrics**")
                dia_metrics_df = pd.DataFrame([
                    {"Metric": "CAGR", "Value": f"{dia_metrics.get('CAGR', 0):.2%}"},
                    {"Metric": "Annualized Volatility", "Value": f"{dia_metrics.get('Annualized Volatility', 0):.2%}"},
                    {"Metric": "Sharpe Ratio", "Value": f"{dia_metrics.get('Sharpe Ratio', 0):.2f}"},
                    {"Metric": "Max Drawdown", "Value": f"{dia_metrics.get('Max Drawdown', 0):.2%}"},
                    {"Metric": "Calmar Ratio", "Value": f"{dia_metrics.get('Calmar Ratio', 0):.2f}"},
                    {"Metric": "Hit Rate", "Value": f"{dia_metrics.get('Hit Rate', 0):.2%}"},
                ])
                st.dataframe(dia_metrics_df, use_container_width=True, hide_index=True)
        
        # Equity Curve
        st.header("📈 Equity Curve")
        
        fig_equity = go.Figure()
        
        # Strategy equity
        fig_equity.add_trace(go.Scatter(
            x=results['results'].index,
            y=results['results']['equity'],
            mode='lines',
            name='Strategy',
            line=dict(color='#2E86AB', width=2)
        ))
        
        # DIA benchmark
        if results.get('benchmark_dia_equity') is not None:
            fig_equity.add_trace(go.Scatter(
                x=results['benchmark_dia_equity'].index,
                y=results['benchmark_dia_equity'].values,
                mode='lines',
                name='DIA ETF (Buy & Hold)',
                line=dict(color='#06A77D', width=2, dash='dash')
            ))
        
        fig_equity.update_layout(
            title="Equity Curve: Strategy vs Benchmark",
            xaxis_title="Date",
            yaxis_title="Equity (Normalized to 1.0)",
            hovermode='x unified',
            height=500,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig_equity, use_container_width=True)
        
        # Drawdown Chart
        st.header("📉 Drawdown Analysis")
        
        equity = results['results']['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='#E63946', width=1),
            fillcolor='rgba(230, 57, 70, 0.3)'
        ))
        
        fig_dd.update_layout(
            title="Drawdown Over Time",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_dd, use_container_width=True)
        
        # Monthly Returns Heatmap
        st.header("📅 Monthly Returns Heatmap")
        
        monthly_returns = (1 + results['results']['net_return']).resample('ME').prod() - 1
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        
        monthly_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        
        pivot_table = monthly_df.pivot(index='Year', columns='Month', values='Return')
        pivot_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Prepare text annotations with values
        z_values = pivot_table.values * 100
        text_values = []
        for row in z_values:
            text_row = []
            for val in row:
                if pd.isna(val):
                    text_row.append("")
                else:
                    # Format as percentage with 1 decimal place
                    text_row.append(f"{val:.1f}%")
            text_values.append(text_row)
        
        # Create heatmap with text annotations
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=z_values,
            x=pivot_table.columns,
            y=pivot_table.index.astype(str),
            colorscale='RdYlGn',
            text=text_values,
            texttemplate='%{text}',
            textfont={"size": 11, "color": "black"},
            colorbar=dict(title="Return (%)"),
            hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>',
            showscale=True
        ))
        
        fig_heatmap.update_layout(
            title="Monthly Returns Heatmap (%) - Values shown in each cell",
            xaxis_title="Month",
            yaxis_title="Year",
            height=600,
            yaxis=dict(autorange="reversed")  # Years from top to bottom
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Daily Trades
        st.header("💼 Daily Trades")
        
        st.write(f"Total trades: {len(st.session_state.trades_df):,}")
        
        # Date filter
        trade_dates = sorted(st.session_state.trades_df['Date'].unique())
        selected_date = st.selectbox(
            "Select Date to View Trades",
            options=trade_dates,
            index=len(trade_dates) - 1 if trade_dates else 0
        )
        
        if selected_date:
            daily_trades = st.session_state.trades_df[
                st.session_state.trades_df['Date'] == pd.to_datetime(selected_date)
            ]
            st.dataframe(daily_trades, use_container_width=True, hide_index=True)
        
        # Download buttons
        st.divider()
        st.subheader("📥 Download Results")
        
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            csv_trades = st.session_state.trades_df.to_csv(index=False)
            st.download_button(
                label="Download Trades CSV",
                data=csv_trades,
                file_name=f"daily_trades_{start_date}_{end_date}.csv",
                mime="text/csv"
            )
        
        with col_dl2:
            csv_results = results['results'].to_csv()
            st.download_button(
                label="Download Backtest Results CSV",
                data=csv_results,
                file_name=f"backtest_results_{start_date}_{end_date}.csv",
                mime="text/csv"
            )

else:
    # Welcome screen
    st.info("👈 **Configure your strategy in the sidebar and click 'Run Backtest' to get started!**")
    
    st.markdown("""
    ### 🎯 Strategy Overview
    
    This is a **daily mean-reversion trading strategy** on the 30 Dow Jones Industrial Average (DJIA) stocks.
    
    **How it works:**
    1. 📊 **Signal Generation**: Calculates 21-day returns for all DJIA stocks
    2. 📉 **Stock Selection**: Ranks stocks by worst performance (most negative returns)
    3. 🎯 **Portfolio Construction**: Selects top K worst performers with equal weights
    4. 💰 **Execution**: 
       - Buys selected stocks at market close each day
       - Sells all positions at next day's close
       - Repeats daily
    
    **Key Features:**
    - ✅ Fully automated backtesting
    - ✅ Performance metrics and visualizations
    - ✅ Comparison with DIA ETF benchmark
    - ✅ Detailed trade log export
    - ✅ Interactive charts and analysis
    
    **Use the sidebar to configure:**
    - Date range for backtesting
    - Number of stocks to select (K)
    - Trading costs
    """)

