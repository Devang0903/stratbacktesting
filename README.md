# DJIA Cross-Sectional Mean-Reversion Trading Strategy

A complete quantitative trading backtesting system implementing a cross-sectional mean-reversion strategy on the 30 Dow Jones Industrial Average (DJIA) stocks.

## Strategy Overview

This strategy implements a **cross-sectional mean-reversion** approach:

- **Universe**: 30 DJIA stocks
- **Signal**: 5-day return reversal score (selects oversold stocks)
- **Portfolio**: Long-only, equal-weight top K stocks
- **Rebalancing**: Weekly or monthly
- **Constraints**: No shorting, no leverage, no derivatives

### Signal Generation

For each stock at date t:
1. Compute 5-day return: `5d_return(t) = (Price(t) / Price(t-5)) - 1`
2. Compute reversal score: `reversal_score(t) = -1 * 5d_return(t)`
3. Rank stocks by reversal score (highest = most oversold)
4. Select top K stocks (default: 5)
5. Assign equal weights: `1/K` for selected stocks, `0` otherwise

## Project Structure

```
TradingStrat/
├── data/                      # Data storage
│   └── djia_prices.pkl        # Cached price data
├── src/                       # Source code
│   ├── data_loader.py         # Data fetching and cleaning
│   ├── signals.py             # Signal generation
│   ├── portfolio.py           # Portfolio construction
│   ├── backtester.py          # Backtesting engine
│   ├── performance.py         # Performance metrics
│   ├── plots.py               # Visualization functions
│   └── run_mean_reversion.py  # Main execution script
├── plots/                     # Generated plots
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the backtest:

```bash
python src/run_mean_reversion.py
```

Or as a module:

```bash
python -m src.run_mean_reversion
```

## Configuration

Edit the configuration section in `src/run_mean_reversion.py`:

```python
START_DATE = '2010-01-01'      # Backtest start date
END_DATE = None                # None = today
FREQUENCY = 'weekly'           # 'weekly' or 'monthly'
TOP_K = 5                      # Number of stocks to select
TRADING_COST_BPS = 5.0         # Trading cost in basis points
```

## Outputs

### Performance Metrics

The script computes and displays:

- **CAGR** (Compound Annual Growth Rate)
- **Annualized Volatility**
- **Sharpe Ratio**
- **Max Drawdown**
- **Calmar Ratio**
- **Average Turnover**
- **Hit Rate** (% of positive days)
- **Worst 1-Day Loss**
- **Worst 1-Month Return**

### Benchmarks

The strategy is compared against:

1. **DIA ETF** (Buy & Hold) - The SPDR Dow Jones Industrial Average ETF
2. **Equal-Weight DJIA** - Equal-weighted portfolio of all 30 DJIA stocks, rebalanced monthly

### Plots

All plots are saved to the `plots/` directory:

1. **Equity Curve** - Strategy vs benchmarks over time
2. **Drawdown Curve** - Drawdown over time
3. **Rolling 12-Month Returns** - Rolling annual returns
4. **Monthly Returns Heatmap** - Calendar view of monthly performance
5. **Daily Returns Distribution** - Histogram with statistics
6. **Average Weights** - Bar chart of average portfolio weights per stock
7. **Selection Frequency** - How often each stock is selected as "oversold"

## Key Features

- **Modular Architecture**: Clean separation of concerns (data, signals, portfolio, backtest, performance)
- **Realistic Backtesting**: Includes trading costs and proper look-ahead bias prevention
- **Comprehensive Metrics**: Industry-standard performance analytics
- **Professional Visualizations**: Publication-ready plots
- **Benchmark Comparison**: Multiple benchmarks for context

## Data

- Data is fetched from Yahoo Finance using `yfinance`
- First run downloads data and caches it to `data/djia_prices.pkl`
- Subsequent runs use cached data (faster execution)

## Notes

- The strategy assumes signals are computed at end of day t
- Trades are executed at the close of day t+1 (avoids look-ahead bias)
- Trading costs are applied based on turnover (default: 5 bps per unit turnover)
- Missing data is forward-filled where appropriate

## License

This project is for educational and research purposes.

