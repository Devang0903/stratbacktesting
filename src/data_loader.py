"""
Data loading and cleaning module for DJIA components using Yahoo Finance.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional
import os
import time


# DJIA 30 components (as of 2024)
DJIA_TICKERS = [
    'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',
    'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
    'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
]


def fetch_djia_data(
    start_date: str = '2010-01-01',
    end_date: Optional[str] = None,
    save_path: Optional[str] = None,
    batch_size: int = 5
) -> pd.DataFrame:
    """
    Fetch daily OHLCV data for all DJIA components using Yahoo Finance.
    Uses batch/clustered fetching to avoid rate limiting and timeouts.
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format. If None, uses today's date.
    save_path : str, optional
        Path to save cleaned data (CSV or pickle). If None, doesn't save.
    batch_size : int
        Number of tickers to fetch in each batch (default: 5)
    
    Returns:
    --------
    pd.DataFrame
        Price matrix with DatetimeIndex (rows) and tickers (columns).
        Values are adjusted close prices.
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    
    print(f"Fetching DJIA data from {start_date} to {end_date} using batch retrieval...")
    print(f"  Batch size: {batch_size} tickers per batch")
    
    # Split tickers into batches
    batches = [DJIA_TICKERS[i:i + batch_size] for i in range(0, len(DJIA_TICKERS), batch_size)]
    print(f"  Split into {len(batches)} batches")
    
    price_dict = {}
    failed_tickers = []
    
    # Fetch data in batches
    for batch_idx, batch_tickers in enumerate(batches):
        print(f"\n  Batch {batch_idx + 1}/{len(batches)}: {', '.join(batch_tickers)}")
        
        # Add delay between batches
        if batch_idx > 0:
            time.sleep(3.0)  # 3 second delay between batches
        
        # Retry logic for batch
        max_retries = 3
        retry_delay = 5.0
        batch_fetched = False
        
        for attempt in range(max_retries):
            try:
                # Use yfinance download for batch fetching (more efficient)
                # This fetches all tickers in the batch at once
                df_batch = yf.download(
                    batch_tickers,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    threads=True,  # Use threading for parallel requests
                    auto_adjust=True  # Explicitly set to avoid warning
                )
                
                if df_batch.empty:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"    ⚠ Empty response, retrying batch {batch_idx + 1} after {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        failed_tickers.extend(batch_tickers)
                        print(f"    ✗ Batch {batch_idx + 1} returned no data")
                        break
                
                # Extract Close prices for each ticker
                # yf.download returns MultiIndex columns: (Adj Close, Close, High, Low, Open, Volume)
                if isinstance(df_batch.columns, pd.MultiIndex):
                    # Multiple tickers case - columns are (Metric, Ticker)
                    if 'Close' in df_batch.columns.levels[0]:
                        for ticker in batch_tickers:
                            if ticker in df_batch['Close'].columns:
                                price_dict[ticker] = df_batch['Close'][ticker]
                                print(f"    ✓ Fetched {ticker}")
                            else:
                                failed_tickers.append(ticker)
                                print(f"    ✗ {ticker} not in response")
                    else:
                        # Try 'Adj Close' if 'Close' not available
                        if 'Adj Close' in df_batch.columns.levels[0]:
                            for ticker in batch_tickers:
                                if ticker in df_batch['Adj Close'].columns:
                                    price_dict[ticker] = df_batch['Adj Close'][ticker]
                                    print(f"    ✓ Fetched {ticker} (Adj Close)")
                                else:
                                    failed_tickers.append(ticker)
                                    print(f"    ✗ {ticker} not in response")
                        else:
                            failed_tickers.extend(batch_tickers)
                            print(f"    ✗ No Close or Adj Close data found")
                else:
                    # Single ticker case - flat columns
                    if len(batch_tickers) == 1:
                        ticker = batch_tickers[0]
                        if 'Close' in df_batch.columns:
                            price_dict[ticker] = df_batch['Close']
                            print(f"    ✓ Fetched {ticker}")
                        elif 'Adj Close' in df_batch.columns:
                            price_dict[ticker] = df_batch['Adj Close']
                            print(f"    ✓ Fetched {ticker} (Adj Close)")
                        else:
                            failed_tickers.append(ticker)
                            print(f"    ✗ No Close data for {ticker}")
                    else:
                        # Unexpected structure
                        failed_tickers.extend(batch_tickers)
                        print(f"    ✗ Unexpected data structure")
                
                batch_fetched = True
                break
                
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = 'rate limit' in error_str or 'too many requests' in error_str or '429' in str(e)
                is_timeout = 'timeout' in error_str or 'timed out' in error_str
                is_missing = 'not found' in error_str or 'delisted' in error_str or 'missing' in error_str or '404' in str(e)
                
                # If it's a missing/delisted ticker, skip it gracefully
                if is_missing:
                    print(f"    ⚠ Skipping batch {batch_idx + 1} - some tickers may be missing/delisted: {e}")
                    # Try to fetch individual tickers that might work
                    for ticker in batch_tickers:
                        try:
                            stock = yf.Ticker(ticker)
                            df_single = stock.history(start=start_date, end=end_date)
                            if not df_single.empty and 'Close' in df_single.columns:
                                price_dict[ticker] = df_single['Close']
                                print(f"    ✓ Fetched {ticker} individually")
                            else:
                                failed_tickers.append(ticker)
                                print(f"    ✗ {ticker} - no data available")
                        except:
                            failed_tickers.append(ticker)
                            print(f"    ✗ {ticker} - failed to fetch")
                    break
                
                if attempt < max_retries - 1:
                    if is_rate_limit:
                        wait_time = retry_delay * (2 ** attempt) + 15  # Extra 15s for rate limits
                        print(f"    ⚠ Rate limited on batch {batch_idx + 1}! Waiting {wait_time}s...")
                    elif is_timeout:
                        wait_time = retry_delay * (2 ** attempt) + 5  # Extra 5s for timeouts
                        print(f"    ⚠ Timeout on batch {batch_idx + 1}! Waiting {wait_time}s...")
                    else:
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"    ⚠ Error on batch {batch_idx + 1}, retry {attempt+1}/{max_retries} after {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    # On final failure, try individual tickers
                    print(f"    ⚠ Batch failed, trying individual tickers...")
                    for ticker in batch_tickers:
                        try:
                            time.sleep(1.0)  # Small delay between individual requests
                            stock = yf.Ticker(ticker)
                            df_single = stock.history(start=start_date, end=end_date)
                            if not df_single.empty and 'Close' in df_single.columns:
                                price_dict[ticker] = df_single['Close']
                                print(f"    ✓ Fetched {ticker} individually")
                            else:
                                failed_tickers.append(ticker)
                                print(f"    ✗ {ticker} - no data available")
                        except Exception as e2:
                            failed_tickers.append(ticker)
                            print(f"    ✗ {ticker} - failed: {str(e2)[:50]}")
                    
                    # Longer delay after final failure
                    if is_rate_limit:
                        print(f"    ⏸ Waiting 30s before next batch due to rate limit...")
                        time.sleep(30.0)
                    elif is_timeout:
                        print(f"    ⏸ Waiting 10s before next batch due to timeout...")
                        time.sleep(10.0)
    
    if failed_tickers:
        print(f"\nWarning: Failed to fetch {len(failed_tickers)} tickers: {failed_tickers}")
    
    # Clean up the price dictionary - remove any None or empty series
    price_dict = {k: v for k, v in price_dict.items() if v is not None and not v.empty}
    
    # Combine into single DataFrame
    if not price_dict:
        raise ValueError("No data fetched. Please check your internet connection and date range.")
    
    price_df = pd.DataFrame(price_dict)
    
    if price_df.empty:
        raise ValueError("No data fetched. Please check your internet connection and date range.")
    
    # Warn if we're missing many tickers
    if len(failed_tickers) > 0:
        print(f"\nNote: Successfully fetched {len(price_dict)}/{len(DJIA_TICKERS)} tickers")
        print(f"      Missing tickers: {failed_tickers}")
        if len(price_dict) < len(DJIA_TICKERS) * 0.5:
            print(f"      Warning: Less than 50% of tickers fetched. Results may be unreliable.")
    
    # Ensure DatetimeIndex
    if not isinstance(price_df.index, pd.DatetimeIndex):
        price_df.index = pd.to_datetime(price_df.index)
    
    # Sort by date
    price_df = price_df.sort_index()
    
    # Forward-fill missing values (within reason)
    price_df = price_df.ffill(limit=5)
    
    # Drop early rows where too many tickers have NaNs
    # Keep rows where at least 80% of tickers have valid data
    min_valid_pct = 0.8
    min_valid_count = int(len(price_df.columns) * min_valid_pct)
    price_df = price_df.dropna(thresh=min_valid_count)
    
    # Drop any remaining columns with too many NaNs
    max_nan_pct = 0.1
    max_nan_count = int(len(price_df) * max_nan_pct)
    price_df = price_df.dropna(axis=1, thresh=len(price_df) - max_nan_count)
    
    # Final forward-fill for any remaining gaps
    price_df = price_df.ffill()
    
    # Drop any rows that still have NaNs
    price_df = price_df.dropna()
    
    print(f"Data loaded: {len(price_df)} trading days, {len(price_df.columns)} tickers")
    print(f"Date range: {price_df.index[0].date()} to {price_df.index[-1].date()}")
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        if save_path.endswith('.csv'):
            price_df.to_csv(save_path)
        elif save_path.endswith('.pkl') or save_path.endswith('.pickle'):
            price_df.to_pickle(save_path)
        print(f"Data saved to {save_path}")
    
    return price_df


def load_saved_data(file_path: str) -> pd.DataFrame:
    """
    Load previously saved price data.
    
    Parameters:
    -----------
    file_path : str
        Path to saved CSV or pickle file
    
    Returns:
    --------
    pd.DataFrame
        Price matrix
    """
    if file_path.endswith('.csv'):
        price_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    elif file_path.endswith('.pkl') or file_path.endswith('.pickle'):
        price_df = pd.read_pickle(file_path)
    else:
        raise ValueError("File must be .csv, .pkl, or .pickle")
    
    return price_df


def check_cached_data(file_path: str, start_date: str, end_date: Optional[str] = None) -> bool:
    """
    Check if cached data exists and covers the requested date range.
    
    Parameters:
    -----------
    file_path : str
        Path to cached data file
    start_date : str
        Requested start date
    end_date : str, optional
        Requested end date (None = today)
    
    Returns:
    --------
    bool
        True if cached data exists and covers the range, False otherwise
    """
    if not os.path.exists(file_path):
        return False
    
    try:
        cached_df = load_saved_data(file_path)
        if cached_df.empty:
            return False
        
        # Check date range
        cached_start = cached_df.index[0]
        cached_end = cached_df.index[-1]
        
        requested_start = pd.to_datetime(start_date)
        if end_date is None:
            requested_end = pd.Timestamp.today()
        else:
            requested_end = pd.to_datetime(end_date)
        
        # Simple logic: use cache if requested dates are within cached date range
        # Allow 1 day tolerance on both ends (markets might be closed, data might be 1 day off)
        tolerance = pd.Timedelta(days=1)
        
        # Check if requested start is after cached start (with tolerance)
        # and requested end is before cached end (with tolerance)
        if (cached_start <= requested_start + tolerance and 
            cached_end >= requested_end - tolerance):
            return True
        
        return False
    except Exception:
        return False
