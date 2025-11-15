"""
Plotting utilities for backtest results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_equity_curve(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series = None,
    dia_equity: pd.Series = None,
    save_path: str = None
):
    """
    Plot equity curve for strategy and benchmarks.
    
    Parameters:
    -----------
    strategy_equity : pd.Series
        Strategy equity curve with DatetimeIndex
    benchmark_equity : pd.Series, optional
        Benchmark equity curve with DatetimeIndex (deprecated, kept for compatibility)
    dia_equity : pd.Series, optional
        DIA ETF equity curve with DatetimeIndex
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(strategy_equity.index, strategy_equity.values, 
            label='Strategy', linewidth=2, color='#2E86AB')
    
    # benchmark_equity is deprecated, kept for compatibility but not plotted
    
    if dia_equity is not None:
        ax.plot(dia_equity.index, dia_equity.values,
                label='DIA ETF (Buy & Hold)', linewidth=2, color='#06A77D', linestyle=':')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Equity', fontsize=12)
    ax.set_title('Equity Curve: Strategy vs Benchmarks', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved equity curve to {save_path}")
    
    plt.show()


def plot_drawdown(strategy_equity: pd.Series, save_path: str = None):
    """
    Plot drawdown curve.
    
    Parameters:
    -----------
    strategy_equity : pd.Series
        Strategy equity curve with DatetimeIndex
    save_path : str, optional
        Path to save figure
    """
    # Compute drawdown
    running_max = strategy_equity.expanding().max()
    drawdown = (strategy_equity - running_max) / running_max
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.fill_between(drawdown.index, drawdown.values, 0,
                     color='#F18F01', alpha=0.6, label='Drawdown')
    ax.plot(drawdown.index, drawdown.values, color='#C73E1D', linewidth=1.5)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown', fontsize=12)
    ax.set_title('Drawdown Curve', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved drawdown curve to {save_path}")
    
    plt.show()


def plot_rolling_12m_return(
    strategy_rolling: pd.Series,
    benchmark_rolling: pd.Series = None,
    dia_rolling: pd.Series = None,
    save_path: str = None
):
    """
    Plot rolling 12-month returns.
    
    Parameters:
    -----------
    strategy_rolling : pd.Series
        Rolling 12-month returns for strategy
    benchmark_rolling : pd.Series, optional
        Rolling 12-month returns for benchmark (deprecated, kept for compatibility)
    dia_rolling : pd.Series, optional
        Rolling 12-month returns for DIA ETF
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(strategy_rolling.index, strategy_rolling.values * 100,
            label='Strategy', linewidth=2, color='#2E86AB')
    
    # benchmark_rolling is deprecated, kept for compatibility but not plotted
    
    if dia_rolling is not None:
        ax.plot(dia_rolling.index, dia_rolling.values * 100,
                label='DIA ETF', linewidth=2, color='#06A77D', linestyle=':')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Rolling 12-Month Return (%)', fontsize=12)
    ax.set_title('Rolling 12-Month Returns', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved rolling 12m return to {save_path}")
    
    plt.show()


def plot_monthly_returns_heatmap(returns_series: pd.Series, save_path: str = None):
    """
    Plot calendar-style heatmap of monthly returns.
    
    Parameters:
    -----------
    returns_series : pd.Series
        Daily returns with DatetimeIndex
    save_path : str, optional
        Path to save figure
    """
    # Resample to monthly returns
    monthly_returns = (1 + returns_series).resample('ME').prod() - 1
    
    # Create year-month matrix
    monthly_returns.index = pd.to_datetime(monthly_returns.index)
    monthly_returns_df = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })
    
    # Pivot to create heatmap
    heatmap_data = monthly_returns_df.pivot(index='Year', columns='Month', values='Return')
    heatmap_data.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig, ax = plt.subplots(figsize=(14, max(6, len(heatmap_data) * 0.3)))
    
    # Create heatmap
    sns.heatmap(heatmap_data * 100, annot=True, fmt='.1f', cmap='RdYlGn',
                center=0, cbar_kws={'label': 'Monthly Return (%)'},
                ax=ax, linewidths=0.5, linecolor='gray')
    
    ax.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Year', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved monthly returns heatmap to {save_path}")
    
    plt.show()


def plot_daily_returns_distribution(returns_series: pd.Series, save_path: str = None):
    """
    Plot histogram of daily returns with basic stats.
    
    Parameters:
    -----------
    returns_series : pd.Series
        Daily returns
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    ax.hist(returns_series.values * 100, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
    
    # Add statistics
    mean_ret = returns_series.mean() * 100
    std_ret = returns_series.std() * 100
    median_ret = returns_series.median() * 100
    
    ax.axvline(mean_ret, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_ret:.3f}%')
    ax.axvline(median_ret, color='green', linestyle='--', linewidth=2, label=f'Median: {median_ret:.3f}%')
    
    ax.set_xlabel('Daily Return (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Daily Returns', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add text box with stats
    stats_text = f'Mean: {mean_ret:.4f}%\nStd: {std_ret:.4f}%\nMedian: {median_ret:.4f}%'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved daily returns distribution to {save_path}")
    
    plt.show()


def plot_avg_weights(weights_df: pd.DataFrame, save_path: str = None):
    """
    Plot bar chart of average weight per stock.
    
    Parameters:
    -----------
    weights_df : pd.DataFrame
        Weights matrix
    save_path : str, optional
        Path to save figure
    """
    avg_weights = weights_df.mean().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars = ax.bar(range(len(avg_weights)), avg_weights.values * 100,
                   color='#2E86AB', alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Stock', fontsize=12)
    ax.set_ylabel('Average Weight (%)', fontsize=12)
    ax.set_title('Average Portfolio Weight per Stock', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(avg_weights)))
    ax.set_xticklabels(avg_weights.index, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved average weights to {save_path}")
    
    plt.show()


def plot_selection_frequency(weights_df: pd.DataFrame, save_path: str = None):
    """
    Plot heatmap of stock selection frequency (how often each stock is selected).
    
    Parameters:
    -----------
    weights_df : pd.DataFrame
        Weights matrix
    save_path : str, optional
        Path to save figure
    """
    # Count how many times each stock has non-zero weight
    selection_freq = (weights_df > 0).sum()
    selection_pct = (selection_freq / len(weights_df)) * 100
    
    # Sort by frequency
    selection_pct = selection_pct.sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars = ax.bar(range(len(selection_pct)), selection_pct.values,
                   color='#F18F01', alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Stock', fontsize=12)
    ax.set_ylabel('Selection Frequency (%)', fontsize=12)
    ax.set_title('Stock Selection Frequency (How Often Selected as "Oversold")', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(selection_pct)))
    ax.set_xticklabels(selection_pct.index, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved selection frequency to {save_path}")
    
    plt.show()

