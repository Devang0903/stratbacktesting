# DJIA Mean-Reversion Strategy - Web Application

## 🚀 Quick Start

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the web application:**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser:**
   - The app will automatically open at `http://localhost:8501`
   - If it doesn't, navigate to that URL manually

## 📖 How to Use

### For Non-Technical Users

1. **Configure Strategy** (Left Sidebar):
   - **Date Range**: Select start and end dates for backtesting
   - **Number of Stocks**: Choose how many worst-performing stocks to select (1-30)
   - **Trading Cost**: Set trading costs in basis points (default: 5 bps)

2. **Run Backtest**:
   - Click the **"🚀 Run Backtest"** button
   - Wait for the backtest to complete (usually 1-2 minutes)

3. **View Results**:
   - **Performance Metrics**: Key statistics at the top
   - **Equity Curve**: Visual comparison with DIA ETF benchmark
   - **Drawdown Chart**: Risk analysis over time
   - **Monthly Returns Heatmap**: Performance by month and year
   - **Daily Trades**: View and filter trades by date

4. **Download Results**:
   - Download trades CSV
   - Download backtest results CSV

## 🎯 Features

- ✅ **Interactive Interface**: No coding required
- ✅ **Real-time Backtesting**: Run strategy with different parameters
- ✅ **Visual Analytics**: Charts and graphs for easy understanding
- ✅ **Performance Comparison**: Strategy vs DIA ETF benchmark
- ✅ **Trade Export**: Download all trades as CSV
- ✅ **Date Filtering**: View trades for specific dates

## 📊 Understanding the Results

### Key Metrics

- **CAGR**: Compound Annual Growth Rate - average yearly return
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Max Drawdown**: Worst peak-to-trough decline
- **Volatility**: Annualized standard deviation of returns
- **Hit Rate**: Percentage of profitable days

### Charts

- **Equity Curve**: Shows how $1 invested would grow over time
- **Drawdown**: Shows periods of losses from peak
- **Monthly Returns Heatmap**: Color-coded monthly performance

## 🔧 Technical Details

- **Framework**: Streamlit
- **Visualization**: Plotly (interactive charts)
- **Data**: Cached locally for fast access
- **Strategy**: Daily mean-reversion on 30 DJIA stocks

## 💡 Tips

- Start with default settings to see baseline performance
- Adjust the number of stocks to see how it affects returns
- Compare different date ranges to see strategy performance in different market conditions
- Use the monthly heatmap to identify best/worst performing periods

## 🐛 Troubleshooting

**App won't start:**
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check that you're in the correct directory

**Backtest fails:**
- Check date range (should be valid trading days)
- Ensure cached data exists (first run may take longer)
- Check internet connection if fetching new data

**Charts not displaying:**
- Refresh the page
- Check browser console for errors

## 📞 Support

For issues or questions, check the main README.md file or review the code documentation.

