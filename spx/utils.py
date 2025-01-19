import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sci_opt
import seaborn as sns

"""
Returns: 

1. sliced_price_df: dataframe containing prices for selected range of dates
"""
def get_range(start_date, end_date, price_df):

    # Get prices in date range
    sliced_price_df = price_df[start_date:end_date]

    # Get rid of duplicate dates
    sliced_price_df = sliced_price_df[~sliced_price_df.index.duplicated()]

    # Fill all missing prices with the previous known price
    sliced_price_df = sliced_price_df.bfill().ffill()

    return sliced_price_df

"""
Returns:

1. daily_returns: dataframe containing daily returns
2. daily_log_returns: dataframe containing daily log returns
3. annualised_returns: dataframe containing returns, scaled to one year
4. annualised_log_returns: dataframe containing log returns, scaled to one year
5. top_n_stocks: series containing the tickers of the top n stocks with the highest returns
6. covariance_matrix: dataframe? containing the covariances between the top n stocks plus SPX
"""
def get_metrics(price_df, n=50, show=True):
    
    # Calculate expected daily returns
    daily_returns = price_df.pct_change() 
    daily_log_returns = np.log(1.0 + daily_returns)

    # Calculate expected annualised returns (pd.Series sorted in descending order)
    annualised_log_returns = (daily_log_returns.mean(axis=0) * 252)
    annualised_returns = np.exp(annualised_log_returns) - 1.0

    # Get top n stocks based on annual returns
    top_n_stocks = annualised_returns.drop(columns="SPX").sort_values(ascending=False).index[:n]

    # Calculate covariance matrix, diagonal terms represent variance
    covariance_matrix = daily_log_returns[top_n_stocks.append(pd.Index(["SPX"]))].cov()

    # Plot heatmap of covriance matrix of top n stocks with SPX
    if show:
        sns.heatmap(covariance_matrix, vmin=-0.002, vmax=0.002, cmap="RdBu", annot=False, fmt="0.0f")
        plt.show()

    return daily_returns, daily_log_returns, annualised_returns, annualised_log_returns, top_n_stocks, covariance_matrix

"""
Returns

1. 
"""

