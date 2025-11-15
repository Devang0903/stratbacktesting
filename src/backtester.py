"""
Core backtesting engine.
"""

import pandas as pd
import numpy as np


def backtest(
    prices_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    trading_cost_bps: float = 5.0,
    initial_capital: float = 1.0
) -> pd.DataFrame:
    """
    Run backtest given prices and weights.
    
    Parameters:
    -----------
    prices_df : pd.DataFrame
        Price matrix (adjusted close) with DatetimeIndex
    weights_df : pd.DataFrame
        Weights matrix with same index and columns as prices_df
    trading_cost_bps : float
        Trading cost in basis points per unit of turnover (default: 5 bps)
    initial_capital : float
        Initial capital (default: 1.0)
    
    Returns:
    --------
    pd.DataFrame
        Results with columns:
        - gross_return: daily gross portfolio return
        - net_return: daily net return (after costs)
        - turnover: daily turnover
        - equity: cumulative equity curve
        - num_positions: number of positions held
    """
    # Align indices
    common_dates = prices_df.index.intersection(weights_df.index)
    prices_df = prices_df.loc[common_dates]
    weights_df = weights_df.loc[common_dates]
    
    # Compute daily returns for each stock
    # For daily rebalancing: buy at close of day t, sell at close of day t+1
    # weights_df[t] = positions bought at close of day t
    # returns_df[t+1] = returns from day t to day t+1
    # portfolio_return[t+1] = sum(returns_df[t+1] * weights_df[t])
    
    returns_df = prices_df.pct_change()  # Daily returns (first row will be NaN)
    
    # Align: we need returns[t+1] matched with weights[t]
    # So shift returns forward by 1 day to align purchase day with return day
    returns_df_aligned = returns_df.shift(-1)
    
    # Remove last row (no next day return available for last day)
    if len(returns_df_aligned) > 0:
        returns_df_aligned = returns_df_aligned.iloc[:-1]
        weights_df = weights_df.iloc[:-1]
    
    # Compute portfolio gross returns
    # On day t+1, we get returns from positions bought on day t
    gross_returns = (returns_df_aligned * weights_df).sum(axis=1)
    gross_returns = gross_returns.fillna(0)
    
    # Compute turnover
    # Turnover = 0.5 * sum_i |w_i(t) - w_i(t-1)|
    turnover = 0.5 * weights_df.diff().abs().sum(axis=1)
    turnover = turnover.fillna(0)
    
    # Compute trading costs
    cost_bps = trading_cost_bps / 10000
    trading_costs = turnover * cost_bps
    
    # Compute net returns
    net_returns = gross_returns - trading_costs
    
    # Compute equity curve
    equity = (1 + net_returns).cumprod() * initial_capital
    
    # Number of positions held
    num_positions = (weights_df > 0).sum(axis=1)
    
    # Compile results
    results = pd.DataFrame({
        'gross_return': gross_returns,
        'net_return': net_returns,
        'turnover': turnover,
        'equity': equity,
        'num_positions': num_positions
    })
    
    return results

