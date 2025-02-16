# Portfolio Analysis Tool

## Overview
A comprehensive portfolio analysis tool that generates detailed PDF reports with technical indicators, risk metrics, and market analysis.

## Portfolio Analysis Script Features

### Data Retrieval & Market Analysis
- Historical price data for portfolio securities
- Benchmark index data (S&P 500)
- Risk-free rate data from FRED
- Market indicators tracking:
  - Fed Funds Rate
  - 10-Year Treasury Yield
  - CPI and Core CPI
  - Unemployment Rate
  - GDP Growth
  - ISM Manufacturing
  - Consumer Confidence
  - Retail Sales
  - Housing Starts
  - S&P 500 P/E Ratio
  - VIX Index

### Portfolio Performance Metrics
- Total portfolio value
- Portfolio weights
- Daily returns
- Portfolio beta
- Sharpe ratio
- Sortino ratio
- Risk-adjusted returns
- Portfolio concentration metrics
- Historical performance tracking

### Risk Analysis
- Value at Risk (VaR):
  - Historical VaR
  - Parametric VaR
  - Conditional VaR (Expected Shortfall)
- Monte Carlo simulation:
  - Future value projections
  - Risk scenarios
  - Confidence intervals
- Kelly Criterion calculations
- Stop loss analysis:
  - ATR-based stops
  - Trailing stops
  - Position-size-weighted stops
- Volatility metrics
- Drawdown analysis

### Correlation Analysis
- Correlation matrix
- Correlation heatmaps
- Rolling correlations
- Correlation stability metrics
- High-volatility period correlations
- Cross-asset correlations

### Income Analysis
- Dividend metrics:
  - Dividend yield
  - Dividend growth
  - Annual dividend income
  - Position-level income
  - Portfolio-level yield
- Income projections
- Payout ratio analysis

### Geographic Analysis
- Revenue exposure by region
- Geographic concentration (HHI)
- Country/region allocation
- Geographic diversification metrics

### Sector Analysis
- Sector allocation
- Sector concentration
- Sector performance contribution
- Sector rotation analysis

### Technical Indicators
- Moving Averages:
  - Simple Moving Averages (10, 20, 50, 100, 200 days)
  - Exponential Moving Averages (12, 26, 50, 200 days)
  - Moving average crossovers
  - Trend strength indicators
- MACD (Moving Average Convergence Divergence):
  - MACD line
  - Signal line
  - MACD histogram
  - Trend signals
- RSI (Relative Strength Index)
- Ichimoku Cloud Analysis:
  - Tenkan-sen (Conversion Line)
  - Kijun-sen (Base Line)
  - Senkou Span A & B (Leading Spans)
  - Chikou Span (Lagging Span)
  - Cloud color analysis
  - Trend strength indicators

### Fundamental Analysis
- Free Cash Flow Analysis:
  - FCF metrics
  - FCF margin
  - FCF growth
  - FCF yield
- Valuation Ratios:
  - P/E ratio
  - P/B ratio
  - EV/EBITDA
- Financial Statement Analysis:
  - Balance sheet metrics
  - Income statement analysis
  - Cash flow statement analysis

### Reporting & Visualization
- PDF Report Generation:
  - Portfolio holdings summary
  - Performance metrics
  - Risk analysis
  - Technical analysis
  - Income analysis
  - Geographic exposure
  - Market indicators
- Data Visualization:
  - Correlation heatmaps
  - Monte Carlo simulation paths
  - Sector allocation pie charts
  - Performance charts
  - Technical indicator charts

### Error Handling & Logging
- Comprehensive error handling
- Production-level logging
- Diagnostic information
- Error recovery mechanisms

### Configuration & Setup
- Environment variable management
- API key configuration
- Portfolio configuration
- Customizable parameters
- Flexible date ranges

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/portfolio-analysis-tool.git
   ```
2. Install required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn reportlab fredapi yfinance pandas_datareader
   ```
3. Add your FRED API key to `api_key.env`
4. Configure your portfolio in `plots/portfolio_config.py`

## Usage
Run the main script to generate a comprehensive PDF report:
```bash
python main.py
```


## Configuration
Edit `plots/portfolio_config.py` to specify your portfolio holdings in the format `ticker: number_of_shares`

Example portfolio:
```python
portfolio = {
    'IGV': 21.00,
    'TSM': 5.00,
    'AVGO': 8.35,
    'GOOGL': 12.52,
    'NVDA': 14.27,
```

## Output
The tool generates a detailed PDF report including:

### 1. Portfolio Summary
- Current holdings and valuations
- Performance metrics
- Risk analysis

### 2. Market Indicators
- Interest rates
- Economic indicators
- Market sentiment metrics

### 3. Dividend Analysis
- Yield analysis
- Income projections
- Growth metrics

### 4. Geographic Allocation
- Regional exposure
- Concentration analysis

### 5. Risk Metrics
- Value at Risk (VaR)
- Expected Shortfall
- Beta analysis

### 6. Correlation Analysis
- Asset correlation matrix
- Heatmap visualization

### 7. Monte Carlo Simulation
- Future value projections
- Risk scenarios

### 8. Individual Stock Analysis
- Technical indicators
- Fundamental metrics
- Risk assessment

## Dependencies
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `matplotlib`: Data visualization
- `seaborn`: Statistical visualizations
- `reportlab`: PDF generation
- `fredapi`: Federal Reserve economic data
- `yfinance`: Market data retrieval
- `pandas_datareader`: Additional data sources

## License
MIT License

Copyright (c) 2024 [Yunxiao Song]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact
- Email: yunxiaoson98@gmail.com

## Disclaimer
This tool is for informational purposes only and should not be considered as financial advice. Always conduct your own research and due diligence before making investment decisions.
