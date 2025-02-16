# Portfolio Analysis Tool

## Overview
A comprehensive portfolio analysis tool that generates detailed PDF reports with technical indicators, risk metrics, and market analysis.

## Features
- **Portfolio Performance Metrics**
  - Beta, Sharpe Ratio, Sortino Ratio
  - Risk metrics (VaR, Expected Shortfall)
  - Kelly Criterion
- **Market Analysis**
  - Market indicators tracking
  - Correlation analysis with heatmaps
  - Monte Carlo simulations
- **Income Analysis**
  - Dividend analysis
  - Income projections
  - Geographic allocation analysis
- **Technical Indicators**
  - MACD indicators
  - Moving averages (SMA/EMA)
  - RSI
  - Ichimoku Cloud

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
