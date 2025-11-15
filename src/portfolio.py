"""
Portfolio construction module for mean-reversion strategy.
"""

import pandas as pd
import numpy as np
from typing import Literal


def generate_rebalance_dates(
    price_df: pd.DataFrame,
    frequency: Literal['daily', 'weekly', 'monthly'] = 'weekly'
) -> pd.DatetimeIndex:
    """
    Generate rebalancing dates based on frequency.
    
    Parameters:
    -----------
    price_df : pd.DataFrame
        Price matrix with DatetimeIndex
    frequency : str
        'daily' for daily rebalancing (every trading day)
        'weekly' for weekly rebalancing (Mondays)
        'monthly' for monthly rebalancing (last trading day of month)
    
    Returns:
    --------
    pd.DatetimeIndex
        Rebalancing dates
    """
    dates = price_df.index
    
    if frequency == 'daily':
        # Rebalance every trading day
        rebalance_dates = dates
    elif frequency == 'weekly':
        # Rebalance on Mondays (or first trading day of week)
        rebalance_dates = dates[dates.weekday == 0]  # Monday = 0
        # If no Monday, use first day of week
        if len(rebalance_dates) == 0:
            # Group by week and take first day
            rebalance_dates = dates.to_series().groupby(
                dates.to_series().dt.to_period('W')
            ).first().index
    elif frequency == 'monthly':
        # Rebalance on last trading day of each month
        # Group by year-month period and get last date in each group
        dates_series = dates.to_series()
        monthly_groups = dates_series.groupby(dates_series.dt.to_period('M'))
        last_dates = monthly_groups.last()
        # Extract the actual dates (not the Period index)
        rebalance_dates = pd.DatetimeIndex(last_dates.values)
    else:
        raise ValueError(f"Frequency must be 'daily', 'weekly', or 'monthly', got {frequency}")
    
    # Ensure rebalance dates are within the price_df index
    rebalance_dates = rebalance_dates.intersection(dates)
    
    return rebalance_dates


def construct_weights(
    reversal_score_df: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    top_k: int = 5,
    daily_rebalance: bool = False
) -> pd.DataFrame:
    """
    Construct portfolio weights based on reversal scores.
    
    On each rebalance date:
    1. Rank stocks descending by reversal_score (most oversold first)
    2. Select top K stocks (worst performers)
    3. Assign equal weights: 1/K for selected, 0 otherwise
    
    Parameters:
    -----------
    reversal_score_df : pd.DataFrame
        Reversal scores matrix with DatetimeIndex
    rebalance_dates : pd.DatetimeIndex
        Dates to rebalance
    top_k : int
        Number of stocks to select (default: 5)
    daily_rebalance : bool
        If True, don't forward-fill weights (for daily rebalancing)
    
    Returns:
    --------
    pd.DataFrame
        Weights matrix with same index and columns as reversal_score_df
        Values are portfolio weights (sum to 1 on rebalance dates, 0 otherwise)
    """
    # Initialize weights matrix
    weights_df = pd.DataFrame(
        0.0,
        index=reversal_score_df.index,
        columns=reversal_score_df.columns
    )
    
    # Process each rebalance date
    for rebal_date in rebalance_dates:
        if rebal_date not in reversal_score_df.index:
            continue
        
        # Get reversal scores for this date
        scores = reversal_score_df.loc[rebal_date]
        
        # Drop NaN values
        valid_scores = scores.dropna()
        
        if len(valid_scores) < top_k:
            # If not enough valid stocks, use all available
            k = len(valid_scores)
            if k == 0:
                continue
        else:
            k = top_k
        
        # Rank descending (highest reversal_score = most oversold = worst performer)
        ranked = valid_scores.sort_values(ascending=False)
        
        # Select top K (worst performers)
        selected_tickers = ranked.head(k).index
        
        # Assign equal weights
        weight = 1.0 / k
        weights_df.loc[rebal_date, selected_tickers] = weight
    
    # Forward-fill weights between rebalance dates (unless daily rebalancing)
    if not daily_rebalance:
        weights_df = weights_df.ffill()
    
    # Fill any remaining NaNs with 0
    weights_df = weights_df.fillna(0)
    
    return weights_df

