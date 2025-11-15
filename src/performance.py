"""
Performance analytics module.
"""

import pandas as pd
import numpy as np
from typing import Union


def compute_cagr(equity_series: pd.Series) -> float:
    """
    Compute Compound Annual Growth Rate (CAGR).
    
    Parameters:
    -----------
    equity_series : pd.Series
        Equity curve values with DatetimeIndex
    
    Returns:
    --------
    float
        CAGR as decimal (e.g., 0.15 for 15%)
    """
    if len(equity_series) < 2:
        return 0.0
    
    start_value = equity_series.iloc[0]
    end_value = equity_series.iloc[-1]
    
    if start_value <= 0:
        return 0.0
    
    # Calculate number of years
    start_date = equity_series.index[0]
    end_date = equity_series.index[-1]
    years = (end_date - start_date).days / 365.25
    
    if years <= 0:
        return 0.0
    
    cagr = (end_value / start_value) ** (1 / years) - 1
    
    return cagr


def compute_annualized_volatility(returns_series: pd.Series) -> float:
    """
    Compute annualized volatility.
    
    Parameters:
    -----------
    returns_series : pd.Series
        Daily returns with DatetimeIndex
    
    Returns:
    --------
    float
        Annualized volatility as decimal
    """
    if len(returns_series) == 0:
        return 0.0
    
    # Assume 252 trading days per year
    trading_days_per_year = 252
    daily_vol = returns_series.std()
    annualized_vol = daily_vol * np.sqrt(trading_days_per_year)
    
    return annualized_vol


def compute_sharpe_ratio(returns_series: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Compute Sharpe ratio (assumes zero risk-free rate by default).
    
    Parameters:
    -----------
    returns_series : pd.Series
        Daily returns with DatetimeIndex
    risk_free_rate : float
        Annual risk-free rate (default: 0.0)
    
    Returns:
    --------
    float
        Sharpe ratio
    """
    if len(returns_series) == 0:
        return 0.0
    
    # Annualize mean return
    trading_days_per_year = 252
    mean_daily_return = returns_series.mean()
    annualized_return = mean_daily_return * trading_days_per_year
    
    # Annualized volatility
    vol = compute_annualized_volatility(returns_series)
    
    if vol == 0:
        return 0.0
    
    sharpe = (annualized_return - risk_free_rate) / vol
    
    return sharpe


def compute_max_drawdown(equity_series: pd.Series) -> float:
    """
    Compute maximum drawdown.
    
    Parameters:
    -----------
    equity_series : pd.Series
        Equity curve values with DatetimeIndex
    
    Returns:
    --------
    float
        Maximum drawdown as decimal (e.g., -0.25 for -25%)
    """
    if len(equity_series) == 0:
        return 0.0
    
    # Compute running maximum
    running_max = equity_series.expanding().max()
    
    # Compute drawdown
    drawdown = (equity_series - running_max) / running_max
    
    max_dd = drawdown.min()
    
    return max_dd


def compute_calmar_ratio(equity_series: pd.Series) -> float:
    """
    Compute Calmar ratio (CAGR / |max_drawdown|).
    
    Parameters:
    -----------
    equity_series : pd.Series
        Equity curve values with DatetimeIndex
    
    Returns:
    --------
    float
        Calmar ratio
    """
    cagr = compute_cagr(equity_series)
    max_dd = compute_max_drawdown(equity_series)
    
    if max_dd == 0:
        return 0.0
    
    calmar = cagr / abs(max_dd)
    
    return calmar


def compute_avg_turnover(turnover_series: pd.Series) -> float:
    """
    Compute average daily turnover.
    
    Parameters:
    -----------
    turnover_series : pd.Series
        Daily turnover values
    
    Returns:
    --------
    float
        Average daily turnover
    """
    return turnover_series.mean()


def compute_hit_rate(returns_series: pd.Series) -> float:
    """
    Compute hit rate (% of days with positive return).
    
    Parameters:
    -----------
    returns_series : pd.Series
        Daily returns
    
    Returns:
    --------
    float
        Hit rate as decimal (e.g., 0.55 for 55%)
    """
    if len(returns_series) == 0:
        return 0.0
    
    positive_days = (returns_series > 0).sum()
    hit_rate = positive_days / len(returns_series)
    
    return hit_rate


def compute_worst_1day_loss(returns_series: pd.Series) -> float:
    """
    Compute worst 1-day loss.
    
    Parameters:
    -----------
    returns_series : pd.Series
        Daily returns
    
    Returns:
    --------
    float
        Worst 1-day return (negative value)
    """
    return returns_series.min()


def compute_worst_1month_return(returns_series: pd.Series) -> float:
    """
    Compute worst 1-month return.
    
    Parameters:
    -----------
    returns_series : pd.Series
        Daily returns with DatetimeIndex
    
    Returns:
    --------
    float
        Worst 1-month return
    """
    # Resample to monthly and compute cumulative return
    monthly_returns = (1 + returns_series).resample('ME').prod() - 1
    
    return monthly_returns.min()


def compute_rolling_12m_return(returns_series: pd.Series) -> pd.Series:
    """
    Compute rolling 12-month return time series.
    
    Parameters:
    -----------
    returns_series : pd.Series
        Daily returns with DatetimeIndex
    
    Returns:
    --------
    pd.Series
        Rolling 12-month returns
    """
    # Compute cumulative returns
    cum_returns = (1 + returns_series).cumprod()
    
    # Rolling 12-month return
    rolling_12m = (cum_returns / cum_returns.shift(252)) - 1
    
    return rolling_12m


def compute_all_metrics(
    returns_series: pd.Series,
    equity_series: pd.Series,
    turnover_series: pd.Series
) -> dict:
    """
    Compute all performance metrics.
    
    Parameters:
    -----------
    returns_series : pd.Series
        Daily net returns
    equity_series : pd.Series
        Equity curve
    turnover_series : pd.Series
        Daily turnover
    
    Returns:
    --------
    dict
        Dictionary of all metrics
    """
    metrics = {
        'CAGR': compute_cagr(equity_series),
        'Annualized Volatility': compute_annualized_volatility(returns_series),
        'Sharpe Ratio': compute_sharpe_ratio(returns_series),
        'Max Drawdown': compute_max_drawdown(equity_series),
        'Calmar Ratio': compute_calmar_ratio(equity_series),
        'Average Turnover': compute_avg_turnover(turnover_series),
        'Hit Rate': compute_hit_rate(returns_series),
        'Worst 1-Day Loss': compute_worst_1day_loss(returns_series),
        'Worst 1-Month Return': compute_worst_1month_return(returns_series),
    }
    
    return metrics

