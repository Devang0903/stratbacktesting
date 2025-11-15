"""
Signal generation module for mean-reversion strategy.
"""

import pandas as pd
import numpy as np


def compute_past_return(price_df: pd.DataFrame, lookback_days: int = 21) -> pd.DataFrame:
    """
    Compute past N-day returns for each stock.
    
    Parameters:
    -----------
    price_df : pd.DataFrame
        Price matrix with DatetimeIndex (rows) and tickers (columns)
    lookback_days : int
        Number of days to look back (default: 21)
    
    Returns:
    --------
    pd.DataFrame
        Past N-day returns: (Price(t) / Price(t-N)) - 1
        Same index and columns as price_df
    """
    # Shift by lookback_days periods and compute return
    past_returns = (price_df / price_df.shift(lookback_days)) - 1
    
    return past_returns


def compute_reversal_score(past_returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute reversal score as negative of past returns.
    
    High reversal_score = stock has fallen recently (oversold) → candidate to buy
    Low reversal_score = stock has risen recently (overbought) → do not buy
    
    Parameters:
    -----------
    past_returns_df : pd.DataFrame
        Past returns matrix
    
    Returns:
    --------
    pd.DataFrame
        Reversal scores: -1 * past_return
        Same index and columns as past_returns_df
    """
    reversal_scores = -1 * past_returns_df
    
    return reversal_scores

