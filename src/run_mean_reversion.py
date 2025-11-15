"""
Main script to run the cross-sectional mean-reversion backtest on DJIA stocks.
"""

import pandas as pd
import numpy as np
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import fetch_djia_data, load_saved_data
from signals import compute_past_return, compute_reversal_score
from portfolio import generate_rebalance_dates, construct_weights
from backtester import backtest
from performance import compute_all_metrics, compute_rolling_12m_return
from plots import (
    plot_equity_curve, plot_drawdown, plot_rolling_12m_return,
    plot_monthly_returns_heatmap, plot_daily_returns_distribution,
    plot_avg_weights, plot_selection_frequency
)


def create_benchmark_spy(
    start_date: str, 
    end_date: str, 
    cache_path: Optional[str] = None
) -> pd.Series:
    """
    Create SPY buy-and-hold benchmark using Yahoo Finance.
    
    Parameters:
    -----------
    start_date : str
        Start date
    end_date : str
        End date
    cache_path : str, optional
        Path to cache file for SPY data
    
    Returns:
    --------
    pd.Series
        SPY equity curve
    """
    import yfinance as yf
    from data_loader import check_cached_data, load_saved_data
    
    # Check cache first
    if cache_path and check_cached_data(cache_path, start_date, end_date):
        print(f"  Using cached SPY data from {cache_path}...")
        cached_data = load_saved_data(cache_path)
        
        # Handle both Series and DataFrame
        if isinstance(cached_data, pd.Series):
            cached_prices = cached_data
        elif isinstance(cached_data, pd.DataFrame):
            cached_prices = cached_data.iloc[:, 0] if len(cached_data.columns) > 0 else pd.Series(dtype=float)
        else:
            cached_prices = cached_data
        
        if len(cached_prices) == 0 or cached_prices.empty:
            print(f"  ⚠ Cached SPY data is empty")
        else:
            # Remove timezone if present
            if cached_prices.index.tz is not None:
                cached_prices.index = cached_prices.index.tz_localize(None)
            
            # Filter to requested date range
            cached_prices = cached_prices.loc[start_date:end_date]
            if len(cached_prices) > 0:
                returns = cached_prices.pct_change().dropna()
                equity = (1 + returns).cumprod()
                equity.index = pd.to_datetime(equity.index)
                return equity
    
    # Fetch from Yahoo Finance if not cached
    print("  Fetching SPY from Yahoo Finance...")
    spy = yf.Ticker('SPY')
    
    try:
        df = spy.history(start=start_date, end=end_date)
        
        if df.empty:
            return pd.Series(dtype=float)
        
        # Get close prices
        prices = df['Close']
        
        # Remove timezone to match strategy data
        if prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)
        
        # Save to cache if path provided
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)
            prices.to_pickle(cache_path)
            print(f"  Saved SPY data to {cache_path}")
        
        # Compute returns
        returns = prices.pct_change().dropna()
        
        # Compute equity curve
        equity = (1 + returns).cumprod()
        equity.index = pd.to_datetime(equity.index)
        
        return equity
    
    except Exception as e:
        print(f"  ⚠ Error fetching SPY: {e}")
        return pd.Series(dtype=float)


def create_benchmark_dia(
    start_date: str, 
    end_date: str, 
    cache_path: Optional[str] = None
) -> pd.Series:
    """
    Create DIA (DJIA ETF) buy-and-hold benchmark using Yahoo Finance.
    
    Parameters:
    -----------
    start_date : str
        Start date
    end_date : str
        End date
    cache_path : str, optional
        Path to cache file for DIA data
    
    Returns:
    --------
    pd.Series
        DIA equity curve
    """
    import yfinance as yf
    from data_loader import check_cached_data, load_saved_data
    
    # Check cache first
    if cache_path and check_cached_data(cache_path, start_date, end_date):
        print(f"  Using cached DIA data from {cache_path}...")
        cached_data = load_saved_data(cache_path)
        
        # Handle both Series and DataFrame
        if isinstance(cached_data, pd.Series):
            cached_prices = cached_data
        elif isinstance(cached_data, pd.DataFrame):
            cached_prices = cached_data.iloc[:, 0] if len(cached_data.columns) > 0 else pd.Series(dtype=float)
        else:
            cached_prices = cached_data
        
        if len(cached_prices) == 0 or cached_prices.empty:
            print(f"  ⚠ Cached DIA data is empty")
        else:
            # Remove timezone if present
            if cached_prices.index.tz is not None:
                cached_prices.index = cached_prices.index.tz_localize(None)
            
            # Filter to requested date range
            cached_prices = cached_prices.loc[start_date:end_date]
            if len(cached_prices) > 0:
                returns = cached_prices.pct_change().dropna()
                equity = (1 + returns).cumprod()
                equity.index = pd.to_datetime(equity.index)
                return equity
    
    # Fetch from Yahoo Finance if not cached
    print("  Fetching DIA from Yahoo Finance...")
    dia = yf.Ticker('DIA')
    
    try:
        df = dia.history(start=start_date, end=end_date)
        
        if df.empty:
            return pd.Series(dtype=float)
        
        # Get close prices
        prices = df['Close']
        
        # Remove timezone to match strategy data
        if prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)
        
        # Save to cache if path provided
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)
            prices.to_pickle(cache_path)
            print(f"  Saved DIA data to {cache_path}")
        
        # Compute returns
        returns = prices.pct_change().dropna()
        
        # Compute equity curve
        equity = (1 + returns).cumprod()
        equity.index = pd.to_datetime(equity.index)
        
        return equity
    
    except Exception as e:
        print(f"  ⚠ Error fetching DIA: {e}")
        return pd.Series(dtype=float)


def main():
    """
    Main execution function.
    """
    print("=" * 70)
    print("DJIA Cross-Sectional Mean-Reversion Backtest")
    print("=" * 70)
    print()
    
    # Configuration
    # Start date: 10 years ago from today
    start_date_obj = pd.Timestamp.today() - pd.DateOffset(years=10)
    START_DATE = start_date_obj.strftime('%Y-%m-%d')
    
    # Set END_DATE to yesterday to use cached data (avoids fetching from Yahoo Finance)
    END_DATE = (pd.Timestamp.today() - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
    
    print(f"Backtest Period: {START_DATE} to {END_DATE} (10 years)")
    print()
    FREQUENCY = 'daily'  # 'daily', 'weekly', or 'monthly'
    TOP_K = 10  # Number of stocks to select (worst performers)
    TRADING_COST_BPS = 5.0  # 5 basis points per unit of turnover
    BATCH_SIZE = 5  # Number of tickers to fetch per batch (to avoid rate limits)
    
    # Create cache file name based on date range
    cache_suffix = f"{START_DATE.replace('-', '')}_{END_DATE.replace('-', '')}"
    DATA_PATH = f'data/djia_prices_{cache_suffix}.pkl'
    DIA_CACHE_PATH = f'data/dia_prices_{cache_suffix}.pkl'
    SPY_CACHE_PATH = f'data/spy_prices_{cache_suffix}.pkl'
    
    PLOTS_DIR = 'plots'
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # ========================================================================
    # Step 1: Load Data
    # ========================================================================
    print("Step 1: Loading data...")
    from data_loader import check_cached_data
    
    if check_cached_data(DATA_PATH, START_DATE, END_DATE):
        print(f"Using cached data from {DATA_PATH}...")
        prices_df = load_saved_data(DATA_PATH)
        # Filter to requested date range if needed
        if prices_df.index[0] < pd.to_datetime(START_DATE) or (prices_df.index[-1] > pd.to_datetime(END_DATE)):
            prices_df = prices_df.loc[START_DATE:END_DATE]
    else:
        print("Fetching data from Yahoo Finance...")
        prices_df = fetch_djia_data(
            start_date=START_DATE,
            end_date=END_DATE,
            save_path=DATA_PATH,
            batch_size=BATCH_SIZE
        )
    print()
    
    # ========================================================================
    # Step 2: Generate Signals
    # ========================================================================
    print("Step 2: Generating signals...")
    LOOKBACK_DAYS = 21  # Number of days to look back for returns
    past_returns = compute_past_return(prices_df, lookback_days=LOOKBACK_DAYS)
    reversal_scores = compute_reversal_score(past_returns)
    print(f"Computed reversal scores for {len(reversal_scores)} dates (lookback: {LOOKBACK_DAYS} days)")
    print()
    
    # ========================================================================
    # Step 3: Construct Portfolio
    # ========================================================================
    print(f"Step 3: Constructing portfolio (frequency: {FREQUENCY}, top_k: {TOP_K})...")
    rebalance_dates = generate_rebalance_dates(prices_df, frequency=FREQUENCY)
    print(f"Generated {len(rebalance_dates)} rebalance dates")
    daily_rebalance = (FREQUENCY == 'daily')
    weights_df = construct_weights(reversal_scores, rebalance_dates, top_k=TOP_K, daily_rebalance=daily_rebalance)
    print(f"Portfolio weights constructed")
    print()
    
    # ========================================================================
    # Step 4: Run Backtest
    # ========================================================================
    print(f"Step 4: Running backtest (trading cost: {TRADING_COST_BPS} bps)...")
    results = backtest(
        prices_df,
        weights_df,
        trading_cost_bps=TRADING_COST_BPS
    )
    print(f"Backtest completed: {len(results)} trading days")
    print()
    
    # ========================================================================
    # Step 4.5: Export Daily Trades to CSV
    # ========================================================================
    if FREQUENCY == 'daily':
        print("Step 4.5: Exporting daily trades to CSV...")
        trades_list = []
        
        # Get prices aligned with weights
        common_dates = prices_df.index.intersection(weights_df.index)
        prices_aligned = prices_df.loc[common_dates]
        weights_aligned = weights_df.loc[common_dates]
        
        # For each day (except first), we:
        # 1. SELL all positions from previous day (at close of current day)
        # 2. BUY new positions for current day (at close of current day)
        for i, date in enumerate(weights_aligned.index):
            day_weights = weights_aligned.loc[date]
            selected_tickers = day_weights[day_weights > 0].index.tolist()
            
            # SELL: positions from previous day (if not first day)
            if i > 0:
                prev_date = weights_aligned.index[i - 1]
                prev_weights = weights_aligned.loc[prev_date]
                prev_tickers = prev_weights[prev_weights > 0].index.tolist()
                
                for ticker in prev_tickers:
                    prev_weight = prev_weights[ticker]
                    sell_price = prices_aligned.loc[date, ticker]  # Sell at today's close
                    trades_list.append({
                        'Date': date,
                        'Action': 'SELL',
                        'Ticker': ticker,
                        'Weight': prev_weight,
                        'Price': sell_price
                    })
            
            # BUY: new positions for today
            for ticker in selected_tickers:
                weight = day_weights[ticker]
                buy_price = prices_aligned.loc[date, ticker]  # Buy at today's close
                trades_list.append({
                    'Date': date,
                    'Action': 'BUY',
                    'Ticker': ticker,
                    'Weight': weight,
                    'Price': buy_price
                })
        
        # Create DataFrame and save to CSV
        trades_df = pd.DataFrame(trades_list)
        trades_df = trades_df.sort_values(['Date', 'Action', 'Ticker'])
        trades_csv_path = f'{PLOTS_DIR}/daily_trades.csv'
        trades_df.to_csv(trades_csv_path, index=False)
        print(f"  Exported {len(trades_df)} trade records to {trades_csv_path}")
        print(f"  Date range: {trades_df['Date'].min()} to {trades_df['Date'].max()}")
        print(f"  Sample: {len(trades_df[trades_df['Date'] == trades_df['Date'].iloc[0]])} trades on first day")
        print()
    
    # ========================================================================
    # Step 5: Create Benchmarks
    # ========================================================================
    print("Step 5: Creating benchmarks...")
    
    # DIA ETF benchmark (buy-and-hold)
    try:
        benchmark_dia_equity = create_benchmark_dia(
            START_DATE, 
            END_DATE,
            cache_path=DIA_CACHE_PATH
        )
        # Align DIA with strategy dates
        common_dates = results.index.intersection(benchmark_dia_equity.index)
        benchmark_dia_equity = benchmark_dia_equity.loc[common_dates]
        # Normalize to start at 1.0
        if len(benchmark_dia_equity) > 0:
            benchmark_dia_equity = benchmark_dia_equity / benchmark_dia_equity.iloc[0]
        print(f"DIA ETF benchmark created: {len(benchmark_dia_equity)} days")
    except Exception as e:
        print(f"Warning: Could not fetch DIA benchmark: {e}")
        benchmark_dia_equity = None
    
    # SPY benchmark (optional)
    try:
        benchmark_spy_equity = create_benchmark_spy(
            START_DATE, 
            END_DATE,
            cache_path=SPY_CACHE_PATH
        )
        # Align SPY with strategy dates
        common_dates = results.index.intersection(benchmark_spy_equity.index)
        benchmark_spy_equity = benchmark_spy_equity.loc[common_dates]
        # Normalize to start at 1.0
        if len(benchmark_spy_equity) > 0:
            benchmark_spy_equity = benchmark_spy_equity / benchmark_spy_equity.iloc[0]
    except Exception as e:
        print(f"Warning: Could not fetch SPY benchmark: {e}")
        benchmark_spy_equity = None
    
    print("Benchmarks created")
    print()
    
    # ========================================================================
    # Step 6: Compute Performance Metrics
    # ========================================================================
    print("Step 6: Computing performance metrics...")
    
    # Strategy metrics
    strategy_metrics = compute_all_metrics(
        results['net_return'],
        results['equity'],
        results['turnover']
    )
    
    # DIA metrics
    if benchmark_dia_equity is not None and len(benchmark_dia_equity) > 1:
        dia_returns = benchmark_dia_equity.pct_change().dropna()
        dia_metrics = compute_all_metrics(
            dia_returns,
            benchmark_dia_equity,
            pd.Series(0.0, index=dia_returns.index)  # No turnover for buy-and-hold
        )
    else:
        dia_metrics = {}
    
    print("Metrics computed")
    print()
    
    # ========================================================================
    # Step 7: Generate Plots
    # ========================================================================
    print("Step 7: Generating plots...")
    
    # Equity curve
    plot_equity_curve(
        results['equity'],
        None,
        benchmark_dia_equity,
        save_path=f'{PLOTS_DIR}/equity_curve.png'
    )
    
    # Drawdown
    plot_drawdown(
        results['equity'],
        save_path=f'{PLOTS_DIR}/drawdown.png'
    )
    
    # Rolling 12-month returns
    strategy_rolling_12m = compute_rolling_12m_return(results['net_return'])
    
    if benchmark_dia_equity is not None and len(benchmark_dia_equity) > 1:
        dia_rolling_12m = compute_rolling_12m_return(dia_returns)
    else:
        dia_rolling_12m = None
    
    plot_rolling_12m_return(
        strategy_rolling_12m,
        None,
        dia_rolling_12m,
        save_path=f'{PLOTS_DIR}/rolling_12m_return.png'
    )
    
    # Monthly returns heatmap
    plot_monthly_returns_heatmap(
        results['net_return'],
        save_path=f'{PLOTS_DIR}/monthly_returns_heatmap.png'
    )
    
    # Daily returns distribution
    plot_daily_returns_distribution(
        results['net_return'],
        save_path=f'{PLOTS_DIR}/daily_returns_distribution.png'
    )
    
    # Average weights
    plot_avg_weights(
        weights_df,
        save_path=f'{PLOTS_DIR}/avg_weights.png'
    )
    
    # Selection frequency
    plot_selection_frequency(
        weights_df,
        save_path=f'{PLOTS_DIR}/selection_frequency.png'
    )
    
    print("Plots generated and saved")
    print()
    
    # ========================================================================
    # Step 8: Print Performance Summary
    # ========================================================================
    print("=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print()
    
    print("STRATEGY METRICS:")
    print("-" * 70)
    for metric, value in strategy_metrics.items():
        if 'Ratio' in metric or 'Rate' in metric or 'Turnover' in metric:
            print(f"  {metric:25s}: {value:8.4f}")
        elif 'Drawdown' in metric or 'Loss' in metric or 'Return' in metric:
            print(f"  {metric:25s}: {value:8.4%}")
        else:
            print(f"  {metric:25s}: {value:8.4%}")
    print()
    
    if dia_metrics:
        print("DIA ETF METRICS (Buy & Hold):")
        print("-" * 70)
        for metric, value in dia_metrics.items():
            if 'Ratio' in metric or 'Rate' in metric or 'Turnover' in metric:
                print(f"  {metric:25s}: {value:8.4f}")
            elif 'Drawdown' in metric or 'Loss' in metric or 'Return' in metric:
                print(f"  {metric:25s}: {value:8.4%}")
            else:
                print(f"  {metric:25s}: {value:8.4%}")
        print()
    
    # Comparison table
    print("COMPARISON: Strategy vs DIA ETF (Buy & Hold):")
    print("-" * 70)
    print(f"{'Metric':<25s} {'Strategy':>15s} {'DIA ETF':>15s} {'Difference':>15s}")
    print("-" * 70)
    
    key_metrics = ['CAGR', 'Annualized Volatility', 'Sharpe Ratio', 'Max Drawdown']
    for metric in key_metrics:
        if metric in strategy_metrics and metric in dia_metrics:
            strat_val = strategy_metrics[metric]
            dia_val = dia_metrics[metric]
            diff = strat_val - dia_val
            
            if 'Ratio' in metric:
                print(f"{metric:<25s} {strat_val:>15.4f} {dia_val:>15.4f} {diff:>15.4f}")
            else:
                print(f"{metric:<25s} {strat_val:>15.4%} {dia_val:>15.4%} {diff:>15.4%}")
    print()
    
    # ========================================================================
    # Step 9: Interpretation Summary
    # ========================================================================
    print("=" * 70)
    print("INTERPRETATION SUMMARY")
    print("=" * 70)
    print()
    
    print("Strategy: Cross-Sectional Mean-Reversion on DJIA")
    print(f"- Frequency: {FREQUENCY} rebalancing")
    print(f"- Top K stocks: {TOP_K}")
    print(f"- Trading cost: {TRADING_COST_BPS} bps per unit turnover")
    print()
    
    print("Key Observations:")
    print(f"- Strategy CAGR: {strategy_metrics.get('CAGR', 0):.2%}")
    if dia_metrics:
        print(f"- DIA ETF CAGR: {dia_metrics.get('CAGR', 0):.2%}")
    print(f"- Strategy Sharpe: {strategy_metrics.get('Sharpe Ratio', 0):.2f}")
    if dia_metrics:
        print(f"- DIA ETF Sharpe: {dia_metrics.get('Sharpe Ratio', 0):.2f}")
    print(f"- Strategy Max Drawdown: {strategy_metrics.get('Max Drawdown', 0):.2%}")
    if dia_metrics:
        print(f"- DIA ETF Max Drawdown: {dia_metrics.get('Max Drawdown', 0):.2%}")
    print(f"- Strategy Hit Rate: {strategy_metrics.get('Hit Rate', 0):.2%}")
    print()
    
    print("Mean-reversion behavior:")
    print("- Strategy selects stocks that have fallen recently (oversold)")
    print("- Expectation: These stocks will revert to mean and recover")
    print("- Performance depends on persistence of mean-reversion in DJIA")
    print()
    
    print("=" * 70)
    print("Backtest complete! Check the 'plots' directory for visualizations.")
    print("=" * 70)


if __name__ == '__main__':
    main()

