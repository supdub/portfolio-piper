#!/usr/bin/env python3
"""
This script retrieves the following data:
    - Market Data: Historical prices for portfolio tickers and benchmark index, plus risk‐free rate data.
    - Statistical Data: Daily returns, volatility (standard deviation), and correlations.
    - Risk Metrics: Historical loss distributions (VaR) and Monte Carlo simulation parameters.
For each ticker in the portfolio:
    - Fundamental Data: Earnings, book value, EBITDA, debt, cash flows, plus detailed financial statements.
    - Technical Data: Price/volume history to compute indicators such as SMA (20,50,200), RSI, MACD (and its signal line), and VWAP.
    - Risk & Sizing Metrics: A Kelly criterion estimate using historical excess returns.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime
import logging
import sys
from typing import Dict, List, Tuple, Any
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import os
from datetime import date
from fredapi import Fred
from dotenv import load_dotenv
from portfolio_config import portfolio

# Load environment variables from .env file
load_dotenv('api_key.env')

# Get FRED API key from environment variables
FRED_API_KEY = os.getenv('FRED_API_KEY')

# Configure logging for production-level diagnostics
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Get list of tickers from portfolio
tickers = list(portfolio.keys())

# Set the historical date range (e.g. last 5 years)
end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=5 * 365)


def fetch_historical_prices(
    tickers: List[str], 
    start: datetime.datetime, 
    end: datetime.datetime
) -> pd.DataFrame:
    """Fetches historical price data for a list of tickers."""
    logging.info("Fetching historical prices for portfolio tickers...")
    try:
        # Download data with auto adjustment and grouping by ticker
        data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True, threads=True)
        return data
    except Exception as e:
        logging.error(f"Error fetching historical prices: {e}")
        sys.exit(1)


def fetch_benchmark_data(
    benchmark: str, 
    start: datetime.datetime, 
    end: datetime.datetime
) -> pd.DataFrame:
    """Fetches historical data for a benchmark index."""
    logging.info(f"Fetching benchmark data for {benchmark}...")
    try:
        benchmark_data = yf.download(benchmark, start=start, end=end, auto_adjust=True)
        return benchmark_data
    except Exception as e:
        logging.error(f"Error fetching benchmark data: {e}")
        sys.exit(1)


def fetch_risk_free_rate(
    start: datetime.datetime, 
    end: datetime.datetime
) -> pd.DataFrame:
    """Fetches the risk-free rate data (3-month T-bill rate from FRED: 'DTB3')."""
    logging.info("Fetching risk-free rate data from FRED (DTB3)...")
    try:
        risk_free_data = web.DataReader('DTB3', 'fred', start, end)
        # Forward-fill any missing data points
        risk_free_data = risk_free_data.fillna(method='ffill')
        return risk_free_data
    except Exception as e:
        logging.error(f"Error fetching risk-free rate data: {e}")
        sys.exit(1)


def compute_statistics(
    prices: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """
    Computes key statistical metrics for portfolio analysis.
    
    Calculates daily returns, volatility, and correlations for each ticker in the portfolio.
    Uses closing prices from a MultiIndex DataFrame containing OHLCV data.
    
    Args:
        prices: MultiIndex DataFrame with ticker level and OHLCV columns
        
    Returns:
        Tuple containing:
        - DataFrame of daily returns for each ticker
        - Dictionary mapping tickers to their volatility (std dev of returns)
        - DataFrame of return correlations between tickers
    """
    logging.info("Computing statistical metrics (returns, volatility, correlations)...")
    returns = {}
    volatility = {}
    for ticker in tickers:
        try:
            # Extract the 'Close' prices for each ticker
            ticker_prices = prices[ticker]['Close']
            daily_returns = ticker_prices.pct_change().dropna()
            returns[ticker] = daily_returns
            volatility[ticker] = daily_returns.std()
        except Exception as e:
            logging.error(f"Error computing statistics for {ticker}: {e}")

    returns_df = pd.DataFrame(returns)
    correlations = returns_df.corr()
    return returns_df, volatility, correlations


def compute_var(
    returns_df: pd.DataFrame, 
    confidence_level: float = 0.05,
    lookback_days: int = 252
) -> Dict[str, Dict[str, float]]:
    """
    Computes Value at Risk (VaR) metrics using multiple approaches.
    
    Calculates historical VaR, parametric VaR (assuming normal distribution),
    and Conditional VaR (Expected Shortfall) for each ticker.
    
    Args:
        returns_df: DataFrame of daily returns for each ticker
        confidence_level: Probability threshold for VaR calculation (default: 5%)
        lookback_days: Historical window in trading days (default: 1 year)
        
    Returns:
        Nested dictionary containing VaR metrics for each ticker:
        {
            ticker: {
                'historical_var': float,  # Historical VaR
                'parametric_var': float,  # Parametric VaR 
                'conditional_var': float, # Expected Shortfall
                'mean_return': float,     # Average daily return
                'volatility': float       # Daily return volatility
            }
        }
    """
    logging.info(f"Computing Value at Risk (VaR) at {confidence_level*100}% confidence level...")
    var_metrics = {}
    
    # Use recent data for VaR calculation
    recent_returns = returns_df.tail(lookback_days)
    
    for ticker in returns_df.columns:
        ticker_returns = recent_returns[ticker]
        
        # Historical VaR
        hist_var = ticker_returns.quantile(confidence_level)
        
        # Parametric VaR (assuming normal distribution)
        mean = ticker_returns.mean()
        std = ticker_returns.std()
        param_var = mean + std * np.percentile(np.random.standard_normal(10000), confidence_level*100)
        
        # Conditional VaR (Expected Shortfall)
        cvar = ticker_returns[ticker_returns <= hist_var].mean()
        
        var_metrics[ticker] = {
            'historical_var': hist_var,
            'parametric_var': param_var,
            'conditional_var': cvar,
            'mean_return': mean,
            'volatility': std
        }
    
    return var_metrics


def compute_portfolio_var(
    returns_df: pd.DataFrame,
    portfolio: Dict[str, int],
    prices: pd.DataFrame,
    confidence_level: float = 0.05,
    lookback_days: int = 252
) -> Dict[str, float]:
    """
    Computes portfolio-level Value at Risk using multiple methodologies.
    
    Calculates VaR for the entire portfolio considering position sizes and correlations.
    Provides both percentage and dollar-denominated risk metrics.
    
    Args:
        returns_df: DataFrame of daily returns
        portfolio: Dictionary mapping tickers to number of shares held
        prices: DataFrame of historical prices (MultiIndex with OHLCV)
        confidence_level: Probability threshold for VaR (default: 5%)
        lookback_days: Historical window in trading days (default: 1 year)
        
    Returns:
        Dictionary containing portfolio risk metrics:
        {
            'portfolio_value': float,         # Total portfolio value
            'historical_var_pct': float,      # Historical VaR as percentage
            'parametric_var_pct': float,      # Parametric VaR as percentage
            'conditional_var_pct': float,     # Expected Shortfall as percentage
            'historical_var_dollar': float,   # Historical VaR in dollars
            'parametric_var_dollar': float,   # Parametric VaR in dollars
            'conditional_var_dollar': float,  # Expected Shortfall in dollars
            'mean_return': float,             # Portfolio mean daily return
            'volatility': float               # Portfolio daily volatility
        }
    """
    logging.info("Computing portfolio-level Value at Risk...")
    
    # Calculate portfolio weights
    portfolio_value = 0
    weights = {}
    
    for ticker, shares in portfolio.items():
        latest_price = prices[ticker]['Close'].iloc[-1]
        position_value = shares * latest_price
        portfolio_value += position_value
        weights[ticker] = position_value
    
    # Normalize weights
    for ticker in weights:
        weights[ticker] = weights[ticker] / portfolio_value
    
    recent_returns = returns_df.tail(lookback_days)
    
    # Calculate portfolio returns
    portfolio_returns = sum(recent_returns[ticker] * weights[ticker] 
                          for ticker in portfolio.keys())
    
    # Historical VaR
    hist_var = portfolio_returns.quantile(confidence_level)
    
    # Parametric VaR
    port_mean = portfolio_returns.mean()
    port_std = portfolio_returns.std()
    param_var = port_mean + port_std * np.percentile(np.random.standard_normal(10000), confidence_level*100)
    
    # Conditional VaR
    cvar = portfolio_returns[portfolio_returns <= hist_var].mean()
    
    # Convert to dollar values
    dollar_hist_var = portfolio_value * hist_var
    dollar_param_var = portfolio_value * param_var
    dollar_cvar = portfolio_value * cvar
    
    return {
        'portfolio_value': portfolio_value,
        'historical_var_pct': hist_var,
        'parametric_var_pct': param_var,
        'conditional_var_pct': cvar,
        'historical_var_dollar': dollar_hist_var,
        'parametric_var_dollar': dollar_param_var,
        'conditional_var_dollar': dollar_cvar,
        'mean_return': port_mean,
        'volatility': port_std
    }


def compute_monte_carlo_params(
    returns_df: pd.DataFrame
) -> Dict[str, Dict[str, float]]:
    """
    Computes parameters needed for a Monte Carlo simulation: drift (mean daily return)
    and volatility (standard deviation of daily returns) for each ticker.
    """
    logging.info("Computing Monte Carlo simulation parameters (drift and volatility)...")
    mc_params = {}
    for ticker in returns_df.columns:
        mu = returns_df[ticker].mean()
        sigma = returns_df[ticker].std()
        mc_params[ticker] = {'drift': mu, 'volatility': sigma}
    return mc_params


def compute_technical_indicators(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes technical indicators:
        - Simple Moving Averages (SMA) over 20, 50, and 200 days.
        - Relative Strength Index (RSI).
        - Moving Average Convergence Divergence (MACD) and its Signal line.
        - Volume Weighted Average Price (VWAP).
    'df' should be a DataFrame with columns: Open, High, Low, Close, Volume.
    """
    indicators = {}
    try:
        # Simple Moving Averages
        indicators['SMA_20'] = df['Close'].rolling(window=20).mean()
        indicators['SMA_50'] = df['Close'].rolling(window=50).mean()
        indicators['SMA_200'] = df['Close'].rolling(window=200).mean()

        # RSI Calculation (14-day period)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['RSI'] = 100 - (100 / (1 + rs))

        # MACD and Signal Line
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        indicators['MACD'] = ema12 - ema26
        indicators['Signal_Line'] = indicators['MACD'].ewm(span=9, adjust=False).mean()

        # VWAP Calculation: cumulative (Price * Volume) divided by cumulative Volume
        pv = df['Close'] * df['Volume']
        indicators['VWAP'] = pv.cumsum() / df['Volume'].cumsum()

        return pd.DataFrame(indicators)
    except Exception as e:
        logging.error(f"Error computing technical indicators: {e}")
        return pd.DataFrame()


def fetch_fundamental_data(
    ticker: str
) -> Dict[str, Any]:
    """
    Retrieves fundamental data for a given ticker using yfinance.
    Returns a dictionary including key ratios from the 'info' field and
    detailed financial statements.
    """
    logging.info(f"Fetching fundamental data for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        fundamentals = {
            'earnings': info.get('earnings'),
            'book_value': info.get('bookValue'),
            'EBITDA': info.get('ebitda'),
            'debt': info.get('totalDebt'),
            'cashflow': info.get('freeCashflow'),
            # Add valuation ratios
            'pe_ratio': info.get('forwardPE'),  # Using forward P/E as it's more relevant
            'pb_ratio': info.get('priceToBook'),
            'ev_to_ebitda': info.get('enterpriseToEbitda')
        }
        # Also grab detailed statements if available
        fundamentals['financials'] = stock.financials
        fundamentals['balance_sheet'] = stock.balance_sheet
        fundamentals['cashflow_statement'] = stock.cashflow
        return fundamentals
    except Exception as e:
        logging.error(f"Error fetching fundamental data for {ticker}: {e}")
        return {}


def compute_kelly_criterion(
    daily_returns: pd.Series, 
    risk_free_rate_daily: pd.Series
) -> float:
    """
    Computes an estimate of the Kelly Criterion for daily returns.
    Formula used: Kelly fraction = mean(excess return) / variance(excess return)
    where mean_excess_return is the average of (daily return - daily risk-free rate).
    
    Parameters:
    - daily_returns: A pandas Series of daily returns.
    - risk_free_rate_daily: A pandas Series of daily risk-free rates.
    
    Returns:
    - Kelly Criterion fraction as a float. Returns NaN if variance is zero or an error occurs.
    """
    logging.info("Computing Kelly Criterion...")
    
    if daily_returns.empty or risk_free_rate_daily.empty:
        logging.warning("Input series are empty.")
        return np.nan
    
    try:
        # Calculate excess returns
        excess_returns = daily_returns - risk_free_rate_daily
        
        # Compute mean and variance of excess returns
        mu = excess_returns.mean()
        var = excess_returns.var(ddof=0)  # Use population variance (ddof=0) for consistency
        
        # Avoid division by zero
        if var == 0:
            logging.warning("Variance of excess returns is zero.")
            return np.nan
        
        # Compute Kelly fraction
        kelly_fraction = mu / var
        logging.info(f"Kelly Criterion computed: {kelly_fraction:.4f}")
        return kelly_fraction
    
    except Exception as e:
        logging.error(f"Error computing Kelly Criterion: {e}")
        return np.nan


def analyze_correlations(returns_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs detailed correlation analysis on portfolio returns.
    
    Args:
        returns_df: DataFrame containing daily returns for each ticker
        
    Returns:
        Dictionary containing various correlation metrics and analyses
    """
    logging.info("Performing detailed correlation analysis...")
    
    analysis = {}
    
    # Basic correlation matrix
    corr_matrix = returns_df.corr()
    analysis['correlation_matrix'] = corr_matrix
    
    # Average correlation of each asset with others
    avg_correlations = {}
    for ticker in returns_df.columns:
        other_tickers = [col for col in returns_df.columns if col != ticker]
        avg_corr = corr_matrix[ticker][other_tickers].mean()
        avg_correlations[ticker] = avg_corr
    analysis['average_correlations'] = avg_correlations
    
    # Rolling correlations (using 60-day window)
    rolling_corrs = {}
    for i, ticker1 in enumerate(returns_df.columns):
        for ticker2 in returns_df.columns[i+1:]:
            rolling_corr = returns_df[ticker1].rolling(window=60).corr(returns_df[ticker2])
            rolling_corrs[f'{ticker1}_vs_{ticker2}'] = rolling_corr
    analysis['rolling_correlations'] = rolling_corrs
    
    # Correlation stability (standard deviation of rolling correlations)
    corr_stability = {}
    for pair, rolling_corr in rolling_corrs.items():
        corr_stability[pair] = rolling_corr.std()
    analysis['correlation_stability'] = corr_stability
    
    # Extreme correlation analysis (correlations during high volatility periods)
    volatility = returns_df.std()
    high_vol_mask = returns_df.abs() > returns_df.std() * 2  # periods with >2 std moves
    high_vol_corr = returns_df[high_vol_mask.any(axis=1)].corr()
    analysis['high_volatility_correlations'] = high_vol_corr
    
    return analysis


def compute_betas(
    returns_df: pd.DataFrame, 
    benchmark_data: pd.DataFrame
) -> Dict[str, float]:
    """
    Computes beta for each stock in the portfolio relative to the benchmark.
    Beta = Covariance(Stock Returns, Benchmark Returns) / Variance(Benchmark Returns)
    """
    logging.info("Computing betas for portfolio stocks...")
    betas = {}
    
    # Calculate benchmark returns
    benchmark_returns = benchmark_data['Close'].pct_change().dropna()
    
    # Align benchmark returns with stock returns
    aligned_data = pd.concat([returns_df, benchmark_returns], axis=1).dropna()
    benchmark_returns = aligned_data.iloc[:, -1]  # Last column is benchmark
    
    for ticker in returns_df.columns:
        stock_returns = aligned_data[ticker]
        # Calculate beta using covariance and variance
        beta = (stock_returns.cov(benchmark_returns) / benchmark_returns.var())
        betas[ticker] = beta
    
    return betas


def compute_portfolio_beta(
    betas: Dict[str, float], 
    portfolio: Dict[str, int], 
    prices: pd.DataFrame
) -> float:
    """
    Computes the weighted average beta for the entire portfolio.
    """
    logging.info("Computing portfolio beta...")
    
    # Calculate portfolio value and weights
    portfolio_value = 0
    weights = {}
    
    for ticker, shares in portfolio.items():
        # Get the latest price for each stock
        latest_price = prices[ticker]['Close'].iloc[-1]
        position_value = shares * latest_price
        portfolio_value += position_value
        weights[ticker] = position_value
    
    # Convert values to weights
    for ticker in weights:
        weights[ticker] = weights[ticker] / portfolio_value
    
    # Calculate weighted average beta
    portfolio_beta = sum(weights[ticker] * betas[ticker] for ticker in betas)
    
    return portfolio_beta


def compute_stop_loss_levels(
    prices: pd.DataFrame,
    portfolio: Dict[str, int],
    atr_periods: int = 14,
    atr_multiplier: float = 2.0,
    trailing_stop_pct: float = 0.15
) -> Dict[str, Dict[str, float]]:
    """
    Computes various stop-loss levels for each position in the portfolio:
    - ATR-based stops (using Average True Range)
    - Trailing stop based on percentage
    - Position-size-weighted portfolio stop
    
    Args:
        prices: DataFrame with OHLCV data for each ticker
        portfolio: Dictionary of ticker:shares
        atr_periods: Number of periods for ATR calculation
        atr_multiplier: Multiplier for ATR-based stops
        trailing_stop_pct: Percentage for trailing stop
    
    Returns:
        Dictionary with stop-loss levels for each position
    """
    logging.info("Computing stop-loss levels for portfolio positions...")
    
    stop_levels = {}
    portfolio_value = 0
    
    for ticker, shares in portfolio.items():
        ticker_data = prices[ticker]
        current_price = ticker_data['Close'].iloc[-1]
        position_value = shares * current_price
        portfolio_value += position_value
        
        # Calculate ATR-based stop
        high = ticker_data['High']
        low = ticker_data['Low']
        close = ticker_data['Close']
        
        tr = pd.DataFrame()
        tr['h-l'] = high - low
        tr['h-pc'] = abs(high - close.shift(1))
        tr['l-pc'] = abs(low - close.shift(1))
        true_range = tr.max(axis=1)
        atr = true_range.rolling(window=atr_periods).mean()
        
        atr_stop = current_price - (atr.iloc[-1] * atr_multiplier)
        
        # Calculate trailing stop
        trailing_stop = current_price * (1 - trailing_stop_pct)
        
        # Store stop levels
        stop_levels[ticker] = {
            'current_price': current_price,
            'position_value': position_value,
            'atr_stop': atr_stop,
            'trailing_stop': trailing_stop,
            'stop_distance_pct': {
                'atr': (current_price - atr_stop) / current_price,
                'trailing': trailing_stop_pct
            }
        }
    
    # Calculate portfolio-wide stop levels
    for ticker in stop_levels:
        position_weight = stop_levels[ticker]['position_value'] / portfolio_value
        stop_levels[ticker]['position_weight'] = position_weight
        
        # Weighted stop loss impact on portfolio
        atr_impact = position_weight * stop_levels[ticker]['stop_distance_pct']['atr']
        trailing_impact = position_weight * stop_levels[ticker]['stop_distance_pct']['trailing']
        
        stop_levels[ticker]['portfolio_impact'] = {
            'atr': atr_impact,
            'trailing': trailing_impact
        }
    
    return stop_levels


def compute_portfolio_sharpe_ratio(
    returns_df: pd.DataFrame,
    portfolio: Dict[str, int],
    prices: pd.DataFrame,
    risk_free_rate_data: pd.DataFrame,
    annualize: bool = True
) -> float:
    """
    Computes the Sharpe Ratio for the portfolio.
    
    Args:
        returns_df: DataFrame of daily returns
        portfolio: Dictionary of ticker:shares
        prices: DataFrame of historical prices
        risk_free_rate_data: DataFrame of risk-free rates
        annualize: Whether to annualize the Sharpe Ratio
        
    Returns:
        Sharpe Ratio for the portfolio
    """
    logging.info("Computing portfolio Sharpe Ratio...")
    
    # Calculate portfolio weights
    portfolio_value = 0
    weights = {}
    
    for ticker, shares in portfolio.items():
        latest_price = prices[ticker]['Close'].iloc[-1]
        position_value = shares * latest_price
        portfolio_value += position_value
        weights[ticker] = position_value
    
    # Normalize weights
    for ticker in weights:
        weights[ticker] = weights[ticker] / portfolio_value
    
    # Calculate portfolio returns
    portfolio_returns = sum(returns_df[ticker] * weights[ticker] 
                          for ticker in portfolio.keys())
    
    # Convert annual risk-free rate to daily
    # DTB3 is in percentage terms, so divide by 100 first
    risk_free_daily = risk_free_rate_data.reindex(returns_df.index, method='ffill')
    risk_free_daily = (risk_free_daily.iloc[:, 0] / 100) / 252
    
    # Calculate excess returns
    excess_returns = portfolio_returns - risk_free_daily
    
    # Calculate Sharpe Ratio
    if annualize:
        sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    else:
        sharpe_ratio = excess_returns.mean() / excess_returns.std()
    
    return sharpe_ratio


def compute_portfolio_sortino_ratio(
    returns_df: pd.DataFrame,
    portfolio: Dict[str, int],
    prices: pd.DataFrame,
    risk_free_rate_data: pd.DataFrame,
    annualize: bool = True
) -> float:
    """
    Computes the Sortino Ratio for the portfolio.
    Sortino Ratio = (Portfolio Return - Risk Free Rate) / Downside Deviation
    
    Args:
        returns_df: DataFrame of daily returns
        portfolio: Dictionary of ticker:shares
        prices: DataFrame of historical prices
        risk_free_rate_data: DataFrame of risk-free rates
        annualize: Whether to annualize the Sortino Ratio
        
    Returns:
        Sortino Ratio for the portfolio
    """
    logging.info("Computing portfolio Sortino Ratio...")
    
    # Calculate portfolio weights
    portfolio_value = 0
    weights = {}
    
    for ticker, shares in portfolio.items():
        latest_price = prices[ticker]['Close'].iloc[-1]
        position_value = shares * latest_price
        portfolio_value += position_value
        weights[ticker] = position_value
    
    # Normalize weights
    for ticker in weights:
        weights[ticker] = weights[ticker] / portfolio_value
    
    # Calculate portfolio returns
    portfolio_returns = sum(returns_df[ticker] * weights[ticker] 
                          for ticker in portfolio.keys())
    
    # Convert annual risk-free rate to daily
    risk_free_daily = risk_free_rate_data.reindex(returns_df.index, method='ffill')
    risk_free_daily = (risk_free_daily.iloc[:, 0] / 100) / 252
    
    # Calculate excess returns
    excess_returns = portfolio_returns - risk_free_daily
    
    # Calculate downside deviation
    # Only consider returns below the target (risk-free rate)
    negative_returns = excess_returns[excess_returns < 0]
    downside_deviation = np.sqrt(np.mean(negative_returns**2))
    
    if downside_deviation == 0:
        logging.warning("Downside deviation is zero, cannot compute Sortino ratio")
        return np.nan
    
    # Calculate Sortino Ratio
    if annualize:
        sortino_ratio = np.sqrt(252) * (excess_returns.mean() / downside_deviation)
    else:
        sortino_ratio = excess_returns.mean() / downside_deviation
    
    return sortino_ratio


def simulate_portfolio_monte_carlo(
    returns_df: pd.DataFrame,
    portfolio: Dict[str, int],
    prices: pd.DataFrame,
    n_simulations: int = 1000,
    n_days: int = 252,  # One trading year
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Performs Monte Carlo simulation for the entire portfolio.
    
    Args:
        returns_df: DataFrame of historical returns
        portfolio: Dictionary of ticker:shares
        prices: DataFrame of historical prices
        n_simulations: Number of simulation paths
        n_days: Number of days to simulate
        confidence_level: Confidence level for VaR calculation
        
    Returns:
        Dictionary containing simulation results and risk metrics
    """
    logging.info(f"Running Monte Carlo simulation with {n_simulations} paths...")
    
    # Calculate portfolio value and weights
    portfolio_value = 0
    weights = {}
    latest_prices = {}
    
    for ticker, shares in portfolio.items():
        latest_price = prices[ticker]['Close'].iloc[-1]
        latest_prices[ticker] = latest_price
        position_value = shares * latest_price
        portfolio_value += position_value
        weights[ticker] = position_value / portfolio_value
    
    # Calculate parameters for each stock
    params = {}
    for ticker in portfolio:
        returns = returns_df[ticker]
        mu = returns.mean()
        sigma = returns.std()
        params[ticker] = {
            'mu': mu,
            'sigma': sigma
        }
    
    # Initialize simulation array
    simulation_results = np.zeros((n_simulations, n_days))
    
    # Run simulations
    for sim in range(n_simulations):
        # Initialize portfolio value array for this simulation
        portfolio_values = np.zeros(n_days)
        current_portfolio_value = portfolio_value
        
        for day in range(n_days):
            # Simulate each stock's return and update portfolio value
            daily_return = 0
            for ticker, weight in weights.items():
                # Generate random return using geometric Brownian motion
                mu = params[ticker]['mu']
                sigma = params[ticker]['sigma']
                stock_return = np.random.normal(mu, sigma)
                daily_return += stock_return * weight
            
            # Update portfolio value
            current_portfolio_value *= (1 + daily_return)
            portfolio_values[day] = current_portfolio_value
        
        simulation_results[sim] = portfolio_values
    
    # Calculate metrics from simulation results
    final_values = simulation_results[:, -1]
    sorted_final_values = np.sort(final_values)
    
    # Calculate VaR and Expected Shortfall
    var_index = int(n_simulations * (1 - confidence_level))
    var_value = portfolio_value - sorted_final_values[var_index]
    es_value = portfolio_value - sorted_final_values[:var_index].mean()
    
    # Calculate percentiles for the final values
    percentiles = {
        '5%': np.percentile(final_values, 5),
        '25%': np.percentile(final_values, 25),
        '50%': np.percentile(final_values, 50),
        '75%': np.percentile(final_values, 75),
        '95%': np.percentile(final_values, 95)
    }
    
    # Calculate expected return and volatility
    expected_return = (final_values.mean() - portfolio_value) / portfolio_value
    volatility = final_values.std() / portfolio_value
    
    return {
        'simulation_results': simulation_results,
        'final_values': final_values,
        'percentiles': percentiles,
        'metrics': {
            'expected_return': expected_return,
            'volatility': volatility,
            'var': var_value,
            'expected_shortfall': es_value,
            'initial_value': portfolio_value,
            'mean_final_value': final_values.mean(),
            'median_final_value': np.median(final_values)
        }
    }


def analyze_fcf(fundamentals: Dict[str, Any]) -> Dict[str, float]:
    """
    Analyzes Free Cash Flow metrics from fundamental data.
    
    Args:
        fundamentals: Dictionary containing fundamental data for a ticker
        
    Returns:
        Dictionary containing FCF metrics and analysis
    """
    fcf_analysis = {}
    
    try:
        # Get cashflow statement data
        cashflow_stmt = fundamentals.get('cashflow_statement')
        if cashflow_stmt is None or cashflow_stmt.empty:
            return {'error': 'No cashflow data available'}
            
        # Calculate FCF metrics for the most recent year
        latest_year = cashflow_stmt.columns[0]
        
        # Free Cash Flow = Operating Cash Flow - Capital Expenditures
        operating_cf = cashflow_stmt.loc['Operating Cash Flow', latest_year]
        capex = abs(cashflow_stmt.loc['Capital Expenditure', latest_year])
        fcf = operating_cf - capex
        
        # Calculate FCF metrics
        fcf_analysis['fcf'] = fcf
        fcf_analysis['operating_cf'] = operating_cf
        fcf_analysis['capex'] = capex
        
        # Calculate FCF margin if revenue data is available
        if 'financials' in fundamentals and 'Total Revenue' in fundamentals['financials'].index:
            revenue = fundamentals['financials'].loc['Total Revenue', latest_year]
            fcf_analysis['fcf_margin'] = fcf / revenue
        
        # Calculate FCF growth if we have at least 2 years of data
        if len(cashflow_stmt.columns) >= 2:
            prev_year = cashflow_stmt.columns[1]
            prev_operating_cf = cashflow_stmt.loc['Operating Cash Flow', prev_year]
            prev_capex = abs(cashflow_stmt.loc['Capital Expenditure', prev_year])
            prev_fcf = prev_operating_cf - prev_capex
            
            fcf_analysis['fcf_growth'] = (fcf - prev_fcf) / abs(prev_fcf)
            
        # Calculate FCF yield if market cap is available
        if 'marketCap' in fundamentals.get('info', {}):
            market_cap = fundamentals['info']['marketCap']
            fcf_analysis['fcf_yield'] = fcf / market_cap
            
        return fcf_analysis
        
    except Exception as e:
        logging.error(f"Error computing FCF analysis: {e}")
        return {'error': str(e)}


def compute_ichimoku_cloud(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Ichimoku Cloud components for a given ticker's price data.
    
    Components:
    - Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    - Kijun-sen (Base Line): (26-period high + 26-period low)/2
    - Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
    - Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    - Chikou Span (Lagging Span): Close price shifted back 26 periods
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        DataFrame with Ichimoku components
    """
    # Tenkan-sen (Conversion Line)
    period9_high = df['High'].rolling(window=9).max()
    period9_low = df['Low'].rolling(window=9).min()
    tenkan_sen = (period9_high + period9_low) / 2

    # Kijun-sen (Base Line)
    period26_high = df['High'].rolling(window=26).max()
    period26_low = df['Low'].rolling(window=26).min()
    kijun_sen = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

    # Senkou Span B (Leading Span B)
    period52_high = df['High'].rolling(window=52).max()
    period52_low = df['Low'].rolling(window=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2).shift(26)

    # Chikou Span (Lagging Span)
    chikou_span = df['Close'].shift(-26)

    return pd.DataFrame({
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    })


def analyze_ichimoku_signals(
    df: pd.DataFrame, 
    ichimoku: pd.DataFrame
) -> Dict[str, Any]:
    """
    Analyzes current Ichimoku Cloud signals and conditions.
    
    Args:
        df: Original price DataFrame
        ichimoku: DataFrame with Ichimoku components
        
    Returns:
        Dictionary with analysis of current Ichimoku conditions
    """
    current_price = df['Close'].iloc[-1]
    
    # Get latest values
    latest = {
        'price': current_price,
        'tenkan': ichimoku['tenkan_sen'].iloc[-1],
        'kijun': ichimoku['kijun_sen'].iloc[-1],
        'span_a': ichimoku['senkou_span_a'].iloc[-1],
        'span_b': ichimoku['senkou_span_b'].iloc[-1]
    }
    
    # Determine cloud color (red/green)
    cloud_color = 'green' if latest['span_a'] > latest['span_b'] else 'red'
    
    # Determine price position relative to cloud
    if current_price > max(latest['span_a'], latest['span_b']):
        cloud_position = 'above'
    elif current_price < min(latest['span_a'], latest['span_b']):
        cloud_position = 'below'
    else:
        cloud_position = 'inside'
    
    # Check for TK Cross (Tenkan/Kijun cross)
    tk_cross = None
    if (ichimoku['tenkan_sen'].iloc[-2] <= ichimoku['kijun_sen'].iloc[-2] and
        ichimoku['tenkan_sen'].iloc[-1] > ichimoku['kijun_sen'].iloc[-1]):
        tk_cross = 'bullish'
    elif (ichimoku['tenkan_sen'].iloc[-2] >= ichimoku['kijun_sen'].iloc[-2] and
          ichimoku['tenkan_sen'].iloc[-1] < ichimoku['kijun_sen'].iloc[-1]):
        tk_cross = 'bearish'
    
    # Determine trend strength
    trend_strength = 'strong' if abs(latest['tenkan'] - latest['kijun']) > df['Close'].std() else 'weak'
    
    return {
        'cloud_color': cloud_color,
        'price_position': cloud_position,
        'tk_cross': tk_cross,
        'trend_strength': trend_strength,
        'latest_values': latest
    }


def compute_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes various moving averages for price data:
    - Simple Moving Averages (SMA): 10, 20, 50, 100, 200 days
    - Exponential Moving Averages (EMA): 12, 26, 50, 200 days
    
    Args:
        df: DataFrame with OHLC price data
        
    Returns:
        DataFrame with all moving average calculations
    """
    ma_data = pd.DataFrame()
    
    # Simple Moving Averages
    sma_periods = [10, 20, 50, 100, 200]
    for period in sma_periods:
        ma_data[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
    
    # Exponential Moving Averages
    ema_periods = [12, 26, 50, 200]
    for period in ema_periods:
        ma_data[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    
    return ma_data


def analyze_moving_averages(
    price: float,
    ma_data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Analyzes moving average signals and trends.
    
    Args:
        price: Current price
        ma_data: DataFrame with moving average data
        
    Returns:
        Dictionary containing MA analysis and signals
    """
    latest = ma_data.iloc[-1]
    prev = ma_data.iloc[-2]
    
    analysis = {
        'ma_values': {},
        'crossovers': [],
        'trend_signals': [],
        'price_location': []
    }
    
    # Record latest MA values
    for column in ma_data.columns:
        analysis['ma_values'][column] = latest[column]
    
    # Check for MA crossovers
    # SMA crossovers
    if latest['SMA_20'] > latest['SMA_50'] and prev['SMA_20'] <= prev['SMA_50']:
        analysis['crossovers'].append('Bullish: SMA 20 crossed above SMA 50')
    elif latest['SMA_20'] < latest['SMA_50'] and prev['SMA_20'] >= prev['SMA_50']:
        analysis['crossovers'].append('Bearish: SMA 20 crossed below SMA 50')
        
    if latest['SMA_50'] > latest['SMA_200'] and prev['SMA_50'] <= prev['SMA_200']:
        analysis['crossovers'].append('Bullish: Golden Cross (SMA 50 crossed above SMA 200)')
    elif latest['SMA_50'] < latest['SMA_200'] and prev['SMA_50'] >= prev['SMA_200']:
        analysis['crossovers'].append('Bearish: Death Cross (SMA 50 crossed below SMA 200)')
    
    # EMA crossovers
    if latest['EMA_12'] > latest['EMA_26'] and prev['EMA_12'] <= prev['EMA_26']:
        analysis['crossovers'].append('Bullish: EMA 12 crossed above EMA 26')
    elif latest['EMA_12'] < latest['EMA_26'] and prev['EMA_12'] >= prev['EMA_26']:
        analysis['crossovers'].append('Bearish: EMA 12 crossed below EMA 26')
    
    # Trend analysis
    if price > latest['SMA_200']:
        analysis['trend_signals'].append('Above SMA 200: Long-term uptrend')
    else:
        analysis['trend_signals'].append('Below SMA 200: Long-term downtrend')
        
    if latest['SMA_20'] > latest['SMA_50'] > latest['SMA_200']:
        analysis['trend_signals'].append('Bullish alignment of SMAs')
    elif latest['SMA_20'] < latest['SMA_50'] < latest['SMA_200']:
        analysis['trend_signals'].append('Bearish alignment of SMAs')
    
    # Price location relative to MAs
    mas = ['SMA_20', 'SMA_50', 'SMA_200']
    above_count = sum(1 for ma in mas if price > latest[ma])
    if above_count == len(mas):
        analysis['price_location'].append('Price above all major SMAs: Strong bullish')
    elif above_count == 0:
        analysis['price_location'].append('Price below all major SMAs: Strong bearish')
    else:
        analysis['price_location'].append(f'Price above {above_count} of {len(mas)} major SMAs')
    
    return analysis


def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes MACD (Moving Average Convergence Divergence) components.
    
    Components:
    - MACD Line: 12-day EMA - 26-day EMA
    - Signal Line: 9-day EMA of MACD Line
    - MACD Histogram: MACD Line - Signal Line
    
    Args:
        df: DataFrame with price data (needs 'Close' column)
        
    Returns:
        DataFrame with MACD components
    """
    # Calculate EMAs
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = ema12 - ema26
    
    # Calculate Signal line
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    
    # Calculate MACD histogram
    macd_histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'macd_line': macd_line,
        'signal_line': signal_line,
        'macd_histogram': macd_histogram
    })


def analyze_macd_signals(macd_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes MACD signals and trends.
    
    Args:
        macd_data: DataFrame with MACD components
        
    Returns:
        Dictionary containing MACD analysis and signals
    """
    # Get latest values
    current = macd_data.iloc[-1]
    previous = macd_data.iloc[-2]
    
    analysis = {
        'current_values': {
            'macd_line': current['macd_line'],
            'signal_line': current['signal_line'],
            'histogram': current['macd_histogram']
        },
        'signals': []
    }
    
    # Check for MACD line crossover with Signal line
    if (previous['macd_line'] <= previous['signal_line'] and 
        current['macd_line'] > current['signal_line']):
        analysis['signals'].append('Bullish: MACD crossed above Signal line')
    elif (previous['macd_line'] >= previous['signal_line'] and 
          current['macd_line'] < current['signal_line']):
        analysis['signals'].append('Bearish: MACD crossed below Signal line')
    
    # Check for zero-line crossover
    if previous['macd_line'] <= 0 and current['macd_line'] > 0:
        analysis['signals'].append('Bullish: MACD crossed above zero line')
    elif previous['macd_line'] >= 0 and current['macd_line'] < 0:
        analysis['signals'].append('Bearish: MACD crossed below zero line')
    
    # Check histogram trend (momentum)
    if current['macd_histogram'] > 0:
        if current['macd_histogram'] > previous['macd_histogram']:
            analysis['signals'].append('Bullish momentum increasing')
        else:
            analysis['signals'].append('Bullish momentum decreasing')
    else:
        if current['macd_histogram'] < previous['macd_histogram']:
            analysis['signals'].append('Bearish momentum increasing')
        else:
            analysis['signals'].append('Bearish momentum decreasing')
    
    # Determine overall trend
    if current['macd_line'] > 0 and current['macd_line'] > current['signal_line']:
        analysis['trend'] = 'Strong Bullish'
    elif current['macd_line'] > 0:
        analysis['trend'] = 'Moderately Bullish'
    elif current['macd_line'] < 0 and current['macd_line'] < current['signal_line']:
        analysis['trend'] = 'Strong Bearish'
    else:
        analysis['trend'] = 'Moderately Bearish'
    
    return analysis


def fetch_dividend_data(ticker: str) -> Dict[str, float]:
    """
    Fetches dividend data for a given ticker.
    
    Returns:
        Dictionary containing dividend yield and growth metrics
    """
    logging.info(f"Fetching dividend data for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        
        # Get dividend data
        dividends = stock.dividends
        
        if dividends.empty:
            return {
                'dividend_yield': 0.0,
                'dividend_growth': 0.0,
                'last_dividend': 0.0,
                'annual_dividend': 0.0
            }
        
        # Calculate last year's total dividends
        last_year_dividends = dividends[-4:].sum() if len(dividends) >= 4 else dividends.sum()
        
        # Calculate dividend growth (comparing last year to previous year)
        if len(dividends) >= 8:
            previous_year_dividends = dividends[-8:-4].sum()
            dividend_growth = (last_year_dividends - previous_year_dividends) / previous_year_dividends
        else:
            dividend_growth = 0.0
        
        # Get current price for yield calculation
        current_price = stock.info.get('regularMarketPrice', stock.history(period='1d')['Close'].iloc[-1])
        
        dividend_yield = last_year_dividends / current_price
        
        return {
            'dividend_yield': dividend_yield,
            'dividend_growth': dividend_growth,
            'last_dividend': dividends.iloc[-1] if not dividends.empty else 0.0,
            'annual_dividend': last_year_dividends
        }
        
    except Exception as e:
        logging.error(f"Error fetching dividend data for {ticker}: {e}")
        return {
            'dividend_yield': 0.0,
            'dividend_growth': 0.0,
            'last_dividend': 0.0,
            'annual_dividend': 0.0
        }


def compute_portfolio_dividend_metrics(
    portfolio: Dict[str, int],
    ticker_details: Dict[str, Any]
) -> Dict[str, float]:
    """
    Computes portfolio-level dividend metrics.
    """
    total_annual_dividends = 0
    total_portfolio_value = 0
    
    for ticker, shares in portfolio.items():
        dividend_data = ticker_details[ticker]['dividend_data']
        position_value = ticker_details[ticker]['position_value']
        
        total_annual_dividends += dividend_data['annual_dividend'] * shares
        total_portfolio_value += position_value
    
    return {
        'total_annual_dividends': total_annual_dividends,
        'portfolio_dividend_yield': total_annual_dividends / total_portfolio_value if total_portfolio_value > 0 else 0.0
    }


def fetch_sector_data(ticker: str) -> str:
    """
    Fetches sector information for a given ticker using yfinance.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Sector name as string, or 'Unknown' if not found
    """
    try:
        stock = yf.Ticker(ticker)
        sector = stock.info.get('sector', 'Unknown')
        return sector
    except Exception as e:
        logging.error(f"Error fetching sector data for {ticker}: {e}")
        return 'Unknown'


def analyze_sector_allocation(
    portfolio: Dict[str, int],
    ticker_details: Dict[str, Any]
) -> Dict[str, Dict[str, float]]:
    """
    Analyzes sector allocation of the portfolio.
    
    Args:
        portfolio: Dictionary of ticker:shares
        ticker_details: Dictionary containing ticker-specific details
        
    Returns:
        Dictionary containing sector allocation details
    """
    logging.info("Analyzing sector allocation...")
    
    sector_allocation = {}
    total_portfolio_value = sum(details['position_value'] 
                              for details in ticker_details.values())
    
    # Initialize sector tracking
    for ticker in portfolio:
        sector = ticker_details[ticker]['sector']
        position_value = ticker_details[ticker]['position_value']
        weight = position_value / total_portfolio_value
        
        if sector not in sector_allocation:
            sector_allocation[sector] = {
                'value': 0.0,
                'weight': 0.0,
                'holdings': []
            }
        
        # Update sector data
        sector_allocation[sector]['value'] += position_value
        sector_allocation[sector]['weight'] += weight
        sector_allocation[sector]['holdings'].append({
            'ticker': ticker,
            'value': position_value,
            'weight': weight
        })
    
    return sector_allocation


def fetch_geographic_data(ticker: str) -> Dict[str, float]:
    """
    Fetches geographic revenue breakdown for a given ticker using yfinance.
    Falls back to company headquarters location if detailed breakdown unavailable.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with geographic regions and their revenue percentages
    """
    logging.info(f"Fetching geographic data for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Try to get geographic revenue breakdown
        # This is a simplified approach - in reality, you might need to parse
        # financial statements or use a more specialized data provider
        geo_data = {}
        
        # First try to get detailed revenue breakdown if available
        if 'geographicSegments' in info:
            segments = info['geographicSegments']
            total_revenue = sum(segment.get('revenue', 0) for segment in segments)
            if total_revenue > 0:
                for segment in segments:
                    region = segment.get('region', 'Other')
                    revenue = segment.get('revenue', 0)
                    geo_data[region] = revenue / total_revenue
                return geo_data
        
        # If detailed breakdown not available, use country of domicile
        country = info.get('country', 'Unknown')
        if country != 'Unknown':
            geo_data[country] = 1.0
        else:
            geo_data['Unknown'] = 1.0
            
        return geo_data
        
    except Exception as e:
        logging.error(f"Error fetching geographic data for {ticker}: {e}")
        return {'Unknown': 1.0}


def analyze_geographic_allocation(
    portfolio: Dict[str, int],
    ticker_details: Dict[str, Any]
) -> Dict[str, Dict[str, float]]:
    """
    Analyzes geographic allocation of the portfolio based on revenue exposure.
    
    Args:
        portfolio: Dictionary of ticker:shares
        ticker_details: Dictionary containing ticker-specific details
        
    Returns:
        Dictionary containing geographic allocation details
    """
    logging.info("Analyzing geographic allocation...")
    
    geographic_allocation = {}
    total_portfolio_value = sum(details['position_value'] 
                              for details in ticker_details.values())
    
    # Initialize region tracking
    for ticker, shares in portfolio.items():
        position_value = ticker_details[ticker]['position_value']
        weight = position_value / total_portfolio_value
        geo_exposure = ticker_details[ticker].get('geographic_data', {'Unknown': 1.0})
        
        # Distribute position value across regions based on revenue exposure
        for region, exposure in geo_exposure.items():
            if region not in geographic_allocation:
                geographic_allocation[region] = {
                    'value': 0.0,
                    'weight': 0.0,
                    'holdings': []
                }
            
            # Update region data
            region_value = position_value * exposure
            region_weight = weight * exposure
            geographic_allocation[region]['value'] += region_value
            geographic_allocation[region]['weight'] += region_weight
            geographic_allocation[region]['holdings'].append({
                'ticker': ticker,
                'value': region_value,
                'weight': region_weight
            })
    
    return geographic_allocation


def compute_geographic_concentration(
    geographic_allocation: Dict[str, Dict[str, float]]
) -> float:
    """
    Computes Herfindahl-Hirschman Index (HHI) for geographic concentration.
    HHI ranges from 1/N (perfectly diversified) to 1 (completely concentrated).
    
    Args:
        geographic_allocation: Dictionary containing geographic allocation details
        
    Returns:
        HHI concentration metric
    """
    weights = [details['weight'] for details in geographic_allocation.values()]
    hhi = sum(w * w for w in weights)
    return hhi


def fetch_market_indicators() -> Dict[str, Dict[str, Any]]:
    """
    Fetches current market indicators from various sources using FRED API and yfinance.
    Returns a dictionary of indicators with current, previous values and changes.
    """
    logging.info("Fetching market indicators...")
    
    # Initialize FRED API client with API key from environment variables
    fred = Fred(api_key=FRED_API_KEY)
    
    indicators = {}
    
    try:
        # Fed Funds Rate (FEDFUNDS)
        ff_rate = fred.get_series('FEDFUNDS')
        indicators['Fed Funds Rate'] = {
            'current': ff_rate.iloc[-1],
            'previous': ff_rate.iloc[-2],
            'change': ff_rate.iloc[-1] - ff_rate.iloc[-2],
            'format': 'percent'
        }
        
        # 10-Year Treasury (DGS10)
        treasury_10y = fred.get_series('DGS10')
        indicators['10-Year Treasury'] = {
            'current': treasury_10y.iloc[-1],
            'previous': treasury_10y.iloc[-2],
            'change': treasury_10y.iloc[-1] - treasury_10y.iloc[-2],
            'format': 'percent'
        }
        
        # CPI YoY (CPIAUCSL)
        cpi = fred.get_series('CPIAUCSL')
        cpi_yoy = (cpi.iloc[-1] / cpi.iloc[-13] - 1) * 100
        cpi_yoy_prev = (cpi.iloc[-2] / cpi.iloc[-14] - 1) * 100
        indicators['CPI (YoY)'] = {
            'current': cpi_yoy,
            'previous': cpi_yoy_prev,
            'change': cpi_yoy - cpi_yoy_prev,
            'format': 'percent'
        }
        
        # Core CPI YoY (CPILFESL)
        core_cpi = fred.get_series('CPILFESL')
        core_cpi_yoy = (core_cpi.iloc[-1] / core_cpi.iloc[-13] - 1) * 100
        core_cpi_yoy_prev = (core_cpi.iloc[-2] / core_cpi.iloc[-14] - 1) * 100
        indicators['Core CPI (YoY)'] = {
            'current': core_cpi_yoy,
            'previous': core_cpi_yoy_prev,
            'change': core_cpi_yoy - core_cpi_yoy_prev,
            'format': 'percent'
        }
        
        # Unemployment Rate (UNRATE)
        unemployment = fred.get_series('UNRATE')
        indicators['Unemployment Rate'] = {
            'current': unemployment.iloc[-1],
            'previous': unemployment.iloc[-2],
            'change': unemployment.iloc[-1] - unemployment.iloc[-2],
            'format': 'percent'
        }
        
        # GDP Growth QoQ (GDP)
        gdp = fred.get_series('GDP')
        gdp_qoq = (gdp.iloc[-1] / gdp.iloc[-2] - 1) * 100
        gdp_qoq_prev = (gdp.iloc[-2] / gdp.iloc[-3] - 1) * 100
        indicators['GDP Growth (QoQ)'] = {
            'current': gdp_qoq,
            'previous': gdp_qoq_prev,
            'change': gdp_qoq - gdp_qoq_prev,
            'format': 'percent'
        }
        
        # ISM Manufacturing (NAPM)
        ism = fred.get_series('NAPM')
        indicators['ISM Manufacturing'] = {
            'current': ism.iloc[-1],
            'previous': ism.iloc[-2],
            'change': ism.iloc[-1] - ism.iloc[-2],
            'format': 'number'
        }
        
        # Consumer Confidence (UMCSENT)
        consumer_conf = fred.get_series('UMCSENT')
        indicators['Consumer Confidence'] = {
            'current': consumer_conf.iloc[-1],
            'previous': consumer_conf.iloc[-2],
            'change': consumer_conf.iloc[-1] - consumer_conf.iloc[-2],
            'format': 'number'
        }
        
        # Retail Sales MoM (RSAFS)
        retail = fred.get_series('RSAFS')
        retail_mom = (retail.iloc[-1] / retail.iloc[-2] - 1) * 100
        retail_mom_prev = (retail.iloc[-2] / retail.iloc[-3] - 1) * 100
        indicators['Retail Sales (MoM)'] = {
            'current': retail_mom,
            'previous': retail_mom_prev,
            'change': retail_mom - retail_mom_prev,
            'format': 'percent'
        }
        
        # Housing Starts (HOUST)
        housing = fred.get_series('HOUST')
        indicators['Housing Starts (M)'] = {
            'current': housing.iloc[-1] / 1000,  # Convert to millions
            'previous': housing.iloc[-2] / 1000,
            'change': (housing.iloc[-1] - housing.iloc[-2]) / 1000,
            'format': 'decimal'
        }
        
        # Fetch market data using yfinance
        # S&P 500 P/E Ratio (^GSPC)
        sp500 = yf.Ticker('^GSPC')
        pe_ratio = sp500.info.get('forwardPE', None)
        pe_ratio_prev = pe_ratio * 0.99  # Approximate previous value
        if pe_ratio:
            indicators['S&P 500 P/E Ratio'] = {
                'current': pe_ratio,
                'previous': pe_ratio_prev,
                'change': pe_ratio - pe_ratio_prev,
                'format': 'decimal'
            }
        
        # VIX Index (^VIX)
        vix = yf.Ticker('^VIX')
        vix_current = vix.history(period='2d')['Close']
        indicators['VIX Index'] = {
            'current': vix_current.iloc[-1],
            'previous': vix_current.iloc[-2],
            'change': vix_current.iloc[-1] - vix_current.iloc[-2],
            'format': 'decimal'
        }
        
    except Exception as e:
        logging.error(f"Error fetching market indicators: {e}")
    
    return indicators


def create_pdf_report(
    portfolio_data: Dict[str, Any],
    correlation_analysis: Dict[str, Any],
    mc_simulation: Dict[str, Any],
    ticker_details: Dict[str, Any],
    prices_df: pd.DataFrame
) -> None:
    """
    Creates a PDF report with all portfolio analysis results.
    
    Args:
        portfolio_data: Dictionary containing portfolio metrics
        correlation_analysis: Dictionary containing correlation analysis
        mc_simulation: Dictionary containing Monte Carlo simulation results
        ticker_details: Dictionary containing ticker-specific details
        prices_df: DataFrame containing price data for calculating daily changes
    """
    # Create reports directory if it doesn't exist
    if not os.path.exists('reports'):
        os.makedirs('reports')
    
    # Create PDF with today's date
    today = date.today().strftime('%Y-%m-%d')
    filename = f'reports/portfolio_analysis_{today}.pdf'
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    
    elements = []
    
    # Title
    elements.append(Paragraph(f"Portfolio Analysis Report - {today}", styles['Title']))
    elements.append(Spacer(1, 20))
    
    # Portfolio Holdings Table
    elements.append(Paragraph("Portfolio Holdings", styles['Heading1']))
    
    # Calculate total portfolio value
    total_value = sum(ticker_details[ticker]['position_value'] for ticker in ticker_details)
    
    # Create holdings table data
    holdings_data = [
        ["Ticker", "Shares", "Current Price", "Value", "Weight", "Daily Change"]
    ]
    
    for ticker in portfolio:
        current_price = ticker_details[ticker]['current_price']
        shares = portfolio[ticker]
        position_value = ticker_details[ticker]['position_value']
        weight = position_value / total_value
        
        # Calculate daily change
        daily_return = prices_df[ticker]['Close'].pct_change().iloc[-1]
        
        holdings_data.append([
            ticker,
            f"{shares:,}",
            f"${current_price:.2f}",
            f"${position_value:,.2f}",
            f"{weight*100:.1f}%",
            f"{daily_return*100:+.2f}%"
        ])
    
    # Add total row
    holdings_data.append([
        "Total",
        "",
        "",
        f"${total_value:,.2f}",
        "100.0%",
        ""
    ])
    
    # Create and style the table
    holdings_table = Table(holdings_data, colWidths=[60, 60, 90, 100, 60, 100])
    holdings_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOX', (0, 0), (-1, -1), 2, colors.black)
    ]))
    
    elements.append(holdings_table)
    elements.append(Spacer(1, 20))
    
    # Add Sector Allocation section (after Portfolio Holdings)
    elements.append(Paragraph("Sector Allocation", styles['Heading1']))
    
    # Create sector allocation table
    sector_data = [
        ["Sector", "Value", "Weight", "Holdings"]
    ]
    
    for sector, details in portfolio_data['sector_allocation'].items():
        holdings_str = ", ".join(h['ticker'] for h in details['holdings'])
        sector_data.append([
            sector,
            f"${details['value']:,.2f}",
            f"{details['weight']*100:.1f}%",
            holdings_str
        ])
    
    # Create and style the sector table
    sector_table = Table(sector_data, colWidths=[120, 100, 80, 180])
    sector_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOX', (0, 0), (-1, -1), 2, colors.black),
        ('ALIGN', (3, 1), (3, -1), 'LEFT')  # Left align holdings text
    ]))
    
    elements.append(sector_table)
    elements.append(Spacer(1, 20))
    
    
    # Create pie chart
    plt.figure(figsize=(8, 6))
    sectors = list(portfolio_data['sector_allocation'].keys())
    weights = [details['weight'] * 100 for details in portfolio_data['sector_allocation'].values()]
    
    plt.pie(weights, labels=sectors, autopct='%1.1f%%', startangle=90)
    plt.title('Portfolio Sector Allocation')


    # Save plot to BytesIO object
    img_data = BytesIO()
    plt.savefig(img_data, format='png', bbox_inches='tight')
    img_data.seek(0)
    
    # Add image
    img = Image(img_data)
    img.drawWidth = 5 * inch
    img.drawHeight = 4 * inch
    elements.append(img)
    plt.close()
    elements.append(Spacer(1, 20))
    
    # Portfolio Summary
    elements.append(Paragraph("Portfolio Summary", styles['Heading1']))
    portfolio_summary = [
        ["Metric", "Value"],
        ["Portfolio Value", f"${portfolio_data['portfolio_value']:,.2f}"],
        ["Portfolio Beta", f"{portfolio_data['portfolio_beta']:.2f}"],
        ["Sharpe Ratio", f"{portfolio_data['sharpe_ratio']:.3f}"],
        ["Sortino Ratio", f"{portfolio_data['sortino_ratio']:.3f}"]  # Add this line
    ]
    elements.append(Table(portfolio_summary, colWidths=[200, 200]))
    elements.append(Spacer(1, 20))

        # Add Market Indicators section (after Portfolio Summary)
    elements.append(Paragraph("Market Indicators", styles['Heading1']))
    
    # Fetch market indicators
    market_indicators = fetch_market_indicators()
    
    # Create market indicators table
    market_data = [
        ["Indicator", "Current", "Previous", "Change"]
    ]
    
    for indicator, data in market_indicators.items():
        # Format values based on the indicator's format type
        if data['format'] == 'percent':
            current = f"{data['current']:.2f}%"
            previous = f"{data['previous']:.2f}%"
            change = f"{data['change']:+.2f}%"
        elif data['format'] == 'decimal':
            current = f"{data['current']:.2f}"
            previous = f"{data['previous']:.2f}"
            change = f"{data['change']:+.2f}"
        else:  # number format
            current = f"{data['current']:,.0f}"
            previous = f"{data['previous']:,.0f}"
            change = f"{data['change']:+,.0f}"
        
        # Add row to table
        market_data.append([
            indicator,
            current,
            previous,
            change
        ])
    
    # Create and style the market indicators table
    market_table = Table(market_data, colWidths=[180, 100, 100, 100])
    market_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOX', (0, 0), (-1, -1), 2, colors.black),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),  # Left align indicator names
        # Color positive changes green and negative changes red
        ('TEXTCOLOR', (3, 1), (3, -1), colors.green,
         lambda x, y: x[3].startswith('+')),
        ('TEXTCOLOR', (3, 1), (3, -1), colors.red,
         lambda x, y: x[3].startswith('-'))
    ]))
    
    elements.append(market_table)
    elements.append(Spacer(1, 20))

    # Dividend After Portfolio Holdings Table
    elements.append(Paragraph("Dividend Analysis", styles['Heading1']))
    dividend_data = [
        ["Ticker", "Dividend Yield", "Dividend Growth", "Annual Dividend", "Position Income"]
    ]
    
    total_dividend_income = 0
    for ticker in portfolio:
        div_data = ticker_details[ticker]['dividend_data']
        shares = portfolio[ticker]
        position_income = div_data['annual_dividend'] * shares
        total_dividend_income += position_income
        
        dividend_data.append([
            ticker,
            f"{div_data['dividend_yield']*100:.2f}%",
            f"{div_data['dividend_growth']*100:.2f}%",
            f"${div_data['annual_dividend']:.2f}",
            f"${position_income:.2f}"
        ])
    
    # Add total row
    portfolio_yield = total_dividend_income / portfolio_data['portfolio_value']
    dividend_data.append([
        "Total",
        f"{portfolio_yield*100:.2f}%",
        "",
        "",
        f"${total_dividend_income:.2f}"
    ])
    
    dividend_table = Table(dividend_data, colWidths=[60, 100, 100, 100, 100])
    dividend_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOX', (0, 0), (-1, -1), 2, colors.black)
    ]))
    
    elements.append(dividend_table)
    elements.append(Spacer(1, 20))

        # Geographic Allocation section
    elements.append(Paragraph("Geographic Allocation", styles['Heading1']))
    
    # Create geographic allocation table
    geo_data = [
        ["Region", "Value", "Weight", "Holdings"]
    ]
    
    for region, details in portfolio_data['geographic_allocation'].items():
        holdings_str = ", ".join(h['ticker'] for h in details['holdings'])
        geo_data.append([
            region,
            f"${details['value']:,.2f}",
            f"{details['weight']*100:.1f}%",
            holdings_str
        ])
    
    # Create and style the geographic table
    geo_table = Table(geo_data, colWidths=[120, 100, 80, 180])
    geo_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOX', (0, 0), (-1, -1), 2, colors.black),
        ('ALIGN', (3, 1), (3, -1), 'LEFT')  # Left align holdings text
    ]))
    
    elements.append(geo_table)
    elements.append(Spacer(1, 20))
    
    # Add Geographic Concentration
    elements.append(Paragraph(f"Geographic Concentration (HHI): {portfolio_data['geographic_concentration']:.3f}", 
                            styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Risk Metrics
    elements.append(Paragraph("Risk Metrics", styles['Heading1']))
    risk_metrics = [
        ["Metric", "Percentage", "Dollar Value"],
        ["Historical VaR (95%)", 
         f"{portfolio_data['var']['historical_var_pct']*100:.2f}%",
         f"${-portfolio_data['var']['historical_var_dollar']:,.2f}"],
        ["Expected Shortfall",
         f"{portfolio_data['var']['conditional_var_pct']*100:.2f}%",
         f"${-portfolio_data['var']['conditional_var_dollar']:,.2f}"]
    ]
    elements.append(Table(risk_metrics, colWidths=[150, 150, 150]))
    elements.append(Spacer(1, 20))
    
    # Correlation Heatmap
    elements.append(Paragraph("Correlation Analysis", styles['Heading1']))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(correlation_analysis['correlation_matrix'], 
                annot=True, cmap='coolwarm', center=0)
    plt.title('Portfolio Correlation Heatmap')
    
    # Save plot to BytesIO object
    img_data = BytesIO()
    plt.savefig(img_data, format='png', bbox_inches='tight')
    img_data.seek(0)
    
    # Add image with controlled width (30% smaller)
    img = Image(img_data)
    img.drawWidth = 4 * inch  # Reduced from 5 * inch
    img.drawHeight = 3 * inch  # Maintain aspect ratio
    elements.append(img)
    plt.close()
    elements.append(Spacer(1, 20))
    
    # Monte Carlo Simulation Results
    elements.append(Paragraph("Monte Carlo Simulation Results", styles['Heading1']))
    mc_metrics = [
        ["Metric", "Value"],
        ["Expected Annual Return", f"{mc_simulation['metrics']['expected_return']*100:.1f}%"],
        ["Annual Volatility", f"{mc_simulation['metrics']['volatility']*100:.1f}%"],
        ["95% VaR", f"${mc_simulation['metrics']['var']:,.2f}"],
        ["Expected Shortfall", f"${mc_simulation['metrics']['expected_shortfall']:,.2f}"]
    ]
    elements.append(Table(mc_metrics, colWidths=[200, 200]))
    elements.append(Spacer(1, 10))  # Reduced spacing
    
    # Plot Monte Carlo paths with smaller figure size
    fig, ax = plt.subplots(figsize=(6, 4))  # Reduced from (8, 6) to (6, 4)
    plt.plot(mc_simulation['simulation_results'].T, alpha=0.1, color='blue')
    plt.title('Monte Carlo Simulation Paths')
    plt.xlabel('Days')
    plt.ylabel('Portfolio Value')
    plt.tight_layout()  # Ensure plot fits well in the figure
    
    img_data = BytesIO()
    plt.savefig(img_data, format='png', bbox_inches='tight', dpi=150)  # Adjusted DPI
    img_data.seek(0)
    
    # Add image with controlled width
    img = Image(img_data)
    img.drawWidth = 5 * inch  # Control width in the PDF
    img.drawHeight = 3.33 * inch  # Maintain aspect ratio
    elements.append(img)
    plt.close()  # Close the figure to free memory
    
    # Individual Stock Analysis
    elements.append(Paragraph("Individual Stock Analysis", styles['Heading1']))
    for ticker, details in ticker_details.items():
        elements.append(Paragraph(f"{ticker} Analysis", styles['Heading2']))
        
        # Basic metrics
        stock_metrics = [
            ["Metric", "Value"],
            ["Kelly Criterion", f"{details['kelly_criterion']:.3f}"],
            ["Beta", f"{details['beta']:.2f}"],
            ["Latest Price", f"${details['current_price']:.2f}"],
            ["Position Value", f"${details['position_value']:,.2f}"],
            ["RSI (14-day)", f"{details['rsi']:.2f}" if details['rsi'] is not None else "N/A"],
            # Add valuation ratios
            ["P/E Ratio", f"{details['fundamentals'].get('pe_ratio', 'N/A'):.2f}" if details['fundamentals'].get('pe_ratio') else "N/A"],
            ["P/B Ratio", f"{details['fundamentals'].get('pb_ratio', 'N/A'):.2f}" if details['fundamentals'].get('pb_ratio') else "N/A"],
            ["EV/EBITDA", f"{details['fundamentals'].get('ev_to_ebitda', 'N/A'):.2f}" if details['fundamentals'].get('ev_to_ebitda') else "N/A"]
        ]
        
        # Add Stop Loss Analysis
        if 'stop_levels' in details:
            stops = details['stop_levels']
            stock_metrics.extend([
                ["ATR Stop Level", f"${stops['atr_stop']:.2f} ({stops['stop_distance_pct']['atr']*100:.1f}%)"],
                ["Trailing Stop Level", f"${stops['trailing_stop']:.2f} ({stops['stop_distance_pct']['trailing']*100:.1f}%)"],
                ["Position Weight", f"{stops['position_weight']*100:.1f}%"],
                ["Portfolio Impact (ATR)", f"{stops['portfolio_impact']['atr']*100:.1f}%"],
                ["Portfolio Impact (Trailing)", f"{stops['portfolio_impact']['trailing']*100:.1f}%"]
            ])
        
        # Add FCF metrics if available
        fcf_analysis = details.get('fcf_analysis', {})
        if 'error' not in fcf_analysis:
            if 'fcf' in fcf_analysis:
                stock_metrics.append(["Free Cash Flow", f"${fcf_analysis['fcf']:,.2f}"])
            if 'fcf_margin' in fcf_analysis:
                stock_metrics.append(["FCF Margin", f"{fcf_analysis['fcf_margin']*100:.1f}%"])
            if 'fcf_growth' in fcf_analysis:
                stock_metrics.append(["FCF Growth", f"{fcf_analysis['fcf_growth']*100:.1f}%"])
            if 'fcf_yield' in fcf_analysis:
                stock_metrics.append(["FCF Yield", f"{fcf_analysis['fcf_yield']*100:.1f}%"])
        
        # Add Ichimoku Cloud analysis
        if details.get('ichimoku'):
            ichimoku = details['ichimoku']
            stock_metrics.extend([
                ["Ichimoku Cloud Color", ichimoku['cloud_color'].capitalize()],
                ["Price vs Cloud", ichimoku['price_position'].capitalize()],
                ["Trend Strength", ichimoku['trend_strength'].capitalize()]
            ])
            if ichimoku['tk_cross']:
                stock_metrics.append(["Recent TK Cross", ichimoku['tk_cross'].capitalize()])
        
        # Add Moving Average Analysis
        if details.get('moving_averages'):
            ma = details['moving_averages']
            stock_metrics.extend([
                ["SMA_20", f"${ma['ma_values']['SMA_20']:.2f}"],
                ["SMA_50", f"${ma['ma_values']['SMA_50']:.2f}"],
                ["SMA_200", f"${ma['ma_values']['SMA_200']:.2f}"],
                ["EMA_12", f"${ma['ma_values']['EMA_12']:.2f}"],
                ["EMA_26", f"${ma['ma_values']['EMA_26']:.2f}"],
                ["EMA_50", f"${ma['ma_values']['EMA_50']:.2f}"],
                ["EMA_200", f"${ma['ma_values']['EMA_200']:.2f}"]
            ])
            
            elements.append(Table(stock_metrics, colWidths=[200, 200]))
            elements.append(Spacer(1, 10))
            
            if ma['crossovers']:
                elements.append(Paragraph("Recent Crossovers:", styles['Heading3']))
                for crossover in ma['crossovers']:
                    elements.append(Paragraph(f"• {crossover}", styles['Normal']))
                elements.append(Spacer(1, 10))
            
            elements.append(Paragraph("Trend Signals:", styles['Heading3']))
            for signal in ma['trend_signals']:
                elements.append(Paragraph(f"• {signal}", styles['Normal']))
            
            elements.append(Paragraph("Price Location:", styles['Heading3']))
            for location in ma['price_location']:
                elements.append(Paragraph(f"• {location}", styles['Normal']))
            
            elements.append(Spacer(1, 10))

        # Add MACD Analysis
        try:
            ticker_df = prices_df[ticker].copy()  # Changed from prices to prices_df
            macd_data = compute_macd(ticker_df)
            macd_analysis = analyze_macd_signals(macd_data)
            
            ticker_details[ticker]['macd'] = macd_analysis
            
            logging.info(f"\nMACD Analysis for {ticker}:")
            logging.info(f"Current MACD Line: {macd_analysis['current_values']['macd_line']:.3f}")
            logging.info(f"Current Signal Line: {macd_analysis['current_values']['signal_line']:.3f}")
            logging.info(f"Current Histogram: {macd_analysis['current_values']['histogram']:.3f}")
            logging.info(f"Overall Trend: {macd_analysis['trend']}")
            
            if macd_analysis['signals']:
                logging.info("Recent Signals:")
                for signal in macd_analysis['signals']:
                    logging.info(f"- {signal}")

            # append those to elements
            elements.append(Paragraph("MACD Analysis:", styles['Heading3']))
            
            macd_metrics = [
                ["MACD Line", f"{macd_analysis['current_values']['macd_line']:.3f}"],
                ["Signal Line", f"{macd_analysis['current_values']['signal_line']:.3f}"],
                ["Histogram", f"{macd_analysis['current_values']['histogram']:.3f}"]
            ]
            elements.append(Table(macd_metrics, colWidths=[200, 200]))
            elements.append(Spacer(1, 10))
            
            if macd_analysis['signals']:
                elements.append(Paragraph("Recent MACD Signals:", styles['Heading3']))
                for signal in macd_analysis['signals']:
                    elements.append(Paragraph(f"• {signal}", styles['Normal']))
                elements.append(Spacer(1, 10))
            
            elements.append(Paragraph("MACD Trend:", styles['Heading3']))
            elements.append(Paragraph(f"• {macd_analysis['trend']}", styles['Normal']))
            elements.append(Spacer(1, 10))
                    
        except Exception as e:
            logging.error(f"Error computing MACD for {ticker}: {e}")
            ticker_details[ticker]['macd'] = None

    # Build the PDF
    doc.build(elements)
    logging.info(f"PDF report generated: {filename}")


def main() -> None:
    """Main function to execute the data retrieval and analysis pipeline."""
    # --- Market Data Retrieval ---
    prices = fetch_historical_prices(tickers, start_date, end_date)
    # Using S&P 500 index as the benchmark
    benchmark_symbol = '^GSPC'
    benchmark_data = fetch_benchmark_data(benchmark_symbol, start_date, end_date)
    risk_free_rate_data = fetch_risk_free_rate(start_date, end_date)

    # --- Statistical Data ---
    returns_df, volatility, correlations = compute_statistics(prices)
    logging.info("Latest returns (sample):")
    logging.info(returns_df.tail())
    logging.info("Volatility (std of returns):")
    logging.info(volatility)
    logging.info("Correlations among portfolio tickers:")
    logging.info(correlations)

    # --- Risk Metrics ---
    var_dict = compute_var(returns_df, confidence_level=0.05)
    portfolio_var = compute_portfolio_var(returns_df, portfolio, prices, confidence_level=0.05)
    
    logging.info("\nValue at Risk (VaR) Analysis:")
    logging.info("\nIndividual Asset VaR (5% confidence):")
    for ticker, metrics in var_dict.items():
        logging.info(f"\n{ticker}:")
        logging.info(f"Historical VaR: {metrics['historical_var']*100:.2f}%")
        logging.info(f"Parametric VaR: {metrics['parametric_var']*100:.2f}%")
        logging.info(f"Conditional VaR: {metrics['conditional_var']*100:.2f}%")
    
    logging.info("\nPortfolio-level VaR:")
    logging.info(f"Portfolio Value: ${portfolio_var['portfolio_value']:,.2f}")
    logging.info(f"Historical VaR: {portfolio_var['historical_var_pct']*100:.2f}% "
                f"(${-portfolio_var['historical_var_dollar']:,.2f})")
    logging.info(f"Parametric VaR: {portfolio_var['parametric_var_pct']*100:.2f}% "
                f"(${-portfolio_var['parametric_var_dollar']:,.2f})")
    logging.info(f"Conditional VaR: {portfolio_var['conditional_var_pct']*100:.2f}% "
                f"(${-portfolio_var['conditional_var_dollar']:,.2f})")

    mc_params = compute_monte_carlo_params(returns_df)
    logging.info("Monte Carlo Simulation Parameters (drift and volatility):")
    logging.info(mc_params)

    # --- Beta Calculations ---
    stock_betas = compute_betas(returns_df, benchmark_data)
    portfolio_beta = compute_portfolio_beta(stock_betas, portfolio, prices)
    
    logging.info("\nStock Betas:")
    for ticker, beta in stock_betas.items():
        logging.info(f"{ticker} Beta: {beta:.2f}")
    logging.info(f"\nPortfolio Beta: {portfolio_beta:.2f}")

    # --- Stop Loss Analysis ---
    stop_loss_levels = compute_stop_loss_levels(prices, portfolio)
    
    logging.info("\nStop Loss Levels:")
    for ticker, stops in stop_loss_levels.items():
        logging.info(f"\n{ticker}:")
        logging.info(f"Current Price: ${stops['current_price']:.2f}")
        logging.info(f"ATR Stop: ${stops['atr_stop']:.2f} ({stops['stop_distance_pct']['atr']*100:.1f}%)")
        logging.info(f"Trailing Stop: ${stops['trailing_stop']:.2f} ({stops['stop_distance_pct']['trailing']*100:.1f}%)")
        logging.info(f"Position Weight: {stops['position_weight']*100:.1f}%")
        logging.info(f"Portfolio Impact if Stopped: ATR={stops['portfolio_impact']['atr']*100:.1f}%, "
                    f"Trailing={stops['portfolio_impact']['trailing']*100:.1f}%")

    # --- Ticker-specific Data ---
    ticker_details = {}
    for ticker in tickers:
        ticker_details[ticker] = {}
        
        # Add beta value to ticker details
        ticker_details[ticker]['beta'] = stock_betas[ticker]
        ticker_details[ticker]['current_price'] = prices[ticker]['Close'].iloc[-1]
        ticker_details[ticker]['position_value'] = portfolio[ticker] * ticker_details[ticker]['current_price']

        # Fetch Fundamental Data
        fundamentals = fetch_fundamental_data(ticker)
        ticker_details[ticker]['fundamentals'] = fundamentals
        
        # Log valuation ratios
        logging.info(f"\nValuation Ratios for {ticker}:")
        logging.info(f"P/E Ratio: {fundamentals.get('pe_ratio', 'N/A')}")
        logging.info(f"P/B Ratio: {fundamentals.get('pb_ratio', 'N/A')}")
        logging.info(f"EV/EBITDA: {fundamentals.get('ev_to_ebitda', 'N/A')}")

        # Add FCF Analysis
        fcf_analysis = analyze_fcf(fundamentals)
        ticker_details[ticker]['fcf_analysis'] = fcf_analysis
        
        if 'error' not in fcf_analysis:
            logging.info(f"\nFCF Analysis for {ticker}:")
            logging.info(f"Free Cash Flow: ${fcf_analysis['fcf']:,.2f}")
            logging.info(f"Operating Cash Flow: ${fcf_analysis['operating_cf']:,.2f}")
            logging.info(f"Capital Expenditure: ${fcf_analysis['capex']:,.2f}")
            if 'fcf_margin' in fcf_analysis:
                logging.info(f"FCF Margin: {fcf_analysis['fcf_margin']*100:.1f}%")
            if 'fcf_growth' in fcf_analysis:
                logging.info(f"FCF Growth: {fcf_analysis['fcf_growth']*100:.1f}%")
            if 'fcf_yield' in fcf_analysis:
                logging.info(f"FCF Yield: {fcf_analysis['fcf_yield']*100:.1f}%")

        # Compute Technical Data using historical price & volume data
        try:
            ticker_df = prices[ticker].copy()
            technical_indicators = compute_technical_indicators(ticker_df)
            ticker_details[ticker]['technical'] = technical_indicators
            
            # Log RSI values
            current_rsi = technical_indicators['RSI'].iloc[-1]
            logging.info(f"\n{ticker} RSI (14-day): {current_rsi:.2f}")
            
            # Add RSI to ticker details for the PDF report
            ticker_details[ticker]['rsi'] = current_rsi
            
        except Exception as e:
            logging.error(f"Error processing technical data for {ticker}: {e}")
            ticker_details[ticker]['technical'] = pd.DataFrame()
            ticker_details[ticker]['rsi'] = None

        # Compute Kelly Criterion:
        # Align risk-free rate data with the ticker's daily returns index.
        try:
            daily_returns = returns_df[ticker]
            # The DTB3 data is reported as an annualized percentage.
            # Convert to daily decimal rate: (rate / 100) / 252.
            risk_free_rate_series = risk_free_rate_data.reindex(daily_returns.index, method='ffill')
            risk_free_rate_daily = (risk_free_rate_series.iloc[:, 0] / 100) / 252
            kelly = compute_kelly_criterion(daily_returns, risk_free_rate_daily)
            ticker_details[ticker]['kelly_criterion'] = kelly
        except Exception as e:
            logging.error(f"Error computing Kelly criterion for {ticker}: {e}")
            ticker_details[ticker]['kelly_criterion'] = np.nan

        # Compute Ichimoku Cloud
        try:
            ticker_df = prices[ticker].copy()
            ichimoku_data = compute_ichimoku_cloud(ticker_df)
            ichimoku_analysis = analyze_ichimoku_signals(ticker_df, ichimoku_data)
            
            ticker_details[ticker]['ichimoku'] = ichimoku_analysis
            
            logging.info(f"\nIchimoku Cloud Analysis for {ticker}:")
            logging.info(f"Cloud Color: {ichimoku_analysis['cloud_color']}")
            logging.info(f"Price Position: {ichimoku_analysis['price_position']}")
            if ichimoku_analysis['tk_cross']:
                logging.info(f"TK Cross: {ichimoku_analysis['tk_cross']}")
            logging.info(f"Trend Strength: {ichimoku_analysis['trend_strength']}")
            
        except Exception as e:
            logging.error(f"Error computing Ichimoku Cloud for {ticker}: {e}")
            ticker_details[ticker]['ichimoku'] = None

        # Add Stop Loss Analysis to ticker details
        if ticker in stop_loss_levels:
            ticker_details[ticker]['stop_levels'] = stop_loss_levels[ticker]

        # Add Moving Average Analysis
        try:
            ticker_df = prices[ticker].copy()
            ma_data = compute_moving_averages(ticker_df)
            current_price = ticker_df['Close'].iloc[-1]
            ma_analysis = analyze_moving_averages(current_price, ma_data)
            
            ticker_details[ticker]['moving_averages'] = ma_analysis
            
            logging.info(f"\nMoving Average Analysis for {ticker}:")
            logging.info(f"Current Price: ${current_price:.2f}")
            
            logging.info("\nMA Values:")
            for ma_name, ma_value in ma_analysis['ma_values'].items():
                logging.info(f"{ma_name}: ${ma_value:.2f}")
            
            if ma_analysis['crossovers']:
                logging.info("\nRecent Crossovers:")
                for crossover in ma_analysis['crossovers']:
                    logging.info(crossover)
            
            logging.info("\nTrend Signals:")
            for signal in ma_analysis['trend_signals']:
                logging.info(signal)
            
            logging.info("\nPrice Location:")
            for location in ma_analysis['price_location']:
                logging.info(location)
                
        except Exception as e:
            logging.error(f"Error computing Moving Averages for {ticker}: {e}")
            ticker_details[ticker]['moving_averages'] = None

        # Add Dividend Analysis
        dividend_data = fetch_dividend_data(ticker)
        ticker_details[ticker]['dividend_data'] = dividend_data
        
        logging.info(f"\nDividend Analysis for {ticker}:")
        logging.info(f"Dividend Yield: {dividend_data['dividend_yield']*100:.2f}%")
        logging.info(f"Dividend Growth: {dividend_data['dividend_growth']*100:.2f}%")
        logging.info(f"Annual Dividend: ${dividend_data['annual_dividend']:.2f}")

        # Add sector information
        sector = fetch_sector_data(ticker)
        ticker_details[ticker]['sector'] = sector
        logging.info(f"{ticker} Sector: {sector}")

        # Fetch and analyze geographic data
        geographic_data = fetch_geographic_data(ticker)
        ticker_details[ticker]['geographic_data'] = geographic_data
        
        logging.info(f"\nGeographic Exposure for {ticker}:")
        for region, exposure in geographic_data.items():
            logging.info(f"{region}: {exposure*100:.1f}%")

    # Compute portfolio-level dividend metrics
    portfolio_dividend_metrics = compute_portfolio_dividend_metrics(portfolio, ticker_details)
    logging.info("\nPortfolio Dividend Summary:")
    logging.info(f"Total Annual Dividends: ${portfolio_dividend_metrics['total_annual_dividends']:.2f}")
    logging.info(f"Portfolio Dividend Yield: {portfolio_dividend_metrics['portfolio_dividend_yield']*100:.2f}%")

    # --- Correlation Analysis ---
    correlation_analysis = analyze_correlations(returns_df)
    
    logging.info("\nCorrelation Analysis Summary:")
    logging.info("\nAverage Correlations:")
    for ticker, avg_corr in correlation_analysis['average_correlations'].items():
        logging.info(f"{ticker}: {avg_corr:.3f}")
    
    logging.info("\nCorrelation Stability (lower is more stable):")
    for pair, stability in correlation_analysis['correlation_stability'].items():
        logging.info(f"{pair}: {stability:.3f}")
    
    logging.info("\nHigh Volatility Period Correlations:")
    logging.info(correlation_analysis['high_volatility_correlations'])

    # --- Output Summary ---
    logging.info("Completed data retrieval. Ticker-specific details available:")
    for ticker, details in ticker_details.items():
        logging.info(f"{ticker}: Available keys - {list(details.keys())}")

    sharpe_ratio = compute_portfolio_sharpe_ratio(returns_df, portfolio, prices, risk_free_rate_data)
    logging.info(f"\nPortfolio Sharpe Ratio: {sharpe_ratio:.3f}")

    # After computing Sharpe ratio, add Sortino ratio calculation
    sortino_ratio = compute_portfolio_sortino_ratio(returns_df, portfolio, prices, risk_free_rate_data)
    logging.info(f"\nPortfolio Sortino Ratio: {sortino_ratio:.3f}")

    # Add after the existing risk metrics calculations:
    mc_simulation = simulate_portfolio_monte_carlo(
        returns_df=returns_df,
        portfolio=portfolio,
        prices=prices,
        n_simulations=1000,
        n_days=252,
        confidence_level=0.95
    )
    
    logging.info("\nMonte Carlo Simulation Results:")
    logging.info(f"Initial Portfolio Value: ${mc_simulation['metrics']['initial_value']:,.2f}")
    logging.info(f"Expected Annual Return: {mc_simulation['metrics']['expected_return']*100:.1f}%")
    logging.info(f"Annual Volatility: {mc_simulation['metrics']['volatility']*100:.1f}%")
    logging.info(f"95% VaR: ${mc_simulation['metrics']['var']:,.2f}")
    logging.info(f"Expected Shortfall: ${mc_simulation['metrics']['expected_shortfall']:,.2f}")
    logging.info("\nPortfolio Value Percentiles after 1 year:")
    for percentile, value in mc_simulation['percentiles'].items():
        logging.info(f"{percentile}: ${value:,.2f}")

    # Compute sector allocation
    sector_allocation = analyze_sector_allocation(portfolio, ticker_details)
    
    logging.info("\nSector Allocation:")
    for sector, details in sector_allocation.items():
        logging.info(f"\n{sector}:")
        logging.info(f"Value: ${details['value']:,.2f}")
        logging.info(f"Weight: {details['weight']*100:.1f}%")
        logging.info("Holdings: " + ", ".join(h['ticker'] for h in details['holdings']))

    # Compute geographic allocation
    geographic_allocation = analyze_geographic_allocation(portfolio, ticker_details)
    
    # Compute geographic concentration
    geo_concentration = compute_geographic_concentration(geographic_allocation)
    
    logging.info("\nGeographic Allocation:")
    for region, details in geographic_allocation.items():
        logging.info(f"\n{region}:")
        logging.info(f"Value: ${details['value']:,.2f}")
        logging.info(f"Weight: {details['weight']*100:.1f}%")
        logging.info("Holdings: " + ", ".join(h['ticker'] for h in details['holdings']))
    
    logging.info(f"\nGeographic Concentration (HHI): {geo_concentration:.3f}")

    # Prepare data for the PDF report
    portfolio_data = {
        'portfolio_value': portfolio_var['portfolio_value'],
        'portfolio_beta': portfolio_beta,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,  # Add this line
        'var': portfolio_var,
        'sector_allocation': sector_allocation,
        'geographic_allocation': geographic_allocation,
        'geographic_concentration': geo_concentration
    }
    
    # Generate the PDF report
    create_pdf_report(
        portfolio_data=portfolio_data,
        correlation_analysis=correlation_analysis,
        mc_simulation=mc_simulation,
        ticker_details=ticker_details,
        prices_df=prices
    )
    
    print(f"Analysis complete. PDF report generated in the reports directory.")

    # Add market indicators fetch before creating the PDF report
    market_indicators = fetch_market_indicators()
    logging.info("\nMarket Indicators:")
    for indicator, data in market_indicators.items():
        logging.info(f"\n{indicator}:")
        logging.info(f"Current: {data['current']}")
        logging.info(f"Previous: {data['previous']}")
        logging.info(f"Change: {data['change']:+.2f}")


if __name__ == "__main__":
    main()
