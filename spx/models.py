import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sci_opt

class NaiveSharpe(object):

    def __init__(self, annualised_log_returns, top_n_stocks, covariance_matrix):
        # Optimiser input
        self.annualised_log_returns = annualised_log_returns
        self.top_n_stocks = top_n_stocks
        self.covariance_matrix = covariance_matrix

        # Parameters and constraints for optimisation
        self.num_stocks = len(top_n_stocks)
        self.unhedged_weights = self.num_stocks * [1 / self.num_stocks]  # initial portfolio weights guess
        self.bounds = tuple((0, 1) for _ in range(self.num_stocks))
        self.constraints = ({'type': 'eq', 'fun': self.check_sum})

        # Weights after hedging
        self.long_weights: pd.Series = None
        self.short_weights: pd.Series = None

    # These are general functions that take in the argument weights
    def get_metrics(self, weights) -> np.array:
        # Convert to a Numpy Array.
        weights = np.array(weights)

        # Calculate the returns, remember to annualize them (252)
        ret = np.sum(self.annualised_log_returns[self.top_n_stocks] * weights) * 252

        # Calculate the volatility, remember to annualize them (252), also remove .SPX from covariance matrix
        vol = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix.iloc[:-1, :-1] * 252, weights)))

        # Calculate the Sharpe Ratio.
        sr = ret / vol

        return np.array([ret, vol, sr])

    def grab_negative_sharpe(self, weights) -> np.array:
        return -1*self.get_metrics(weights)[2]

    def grab_volatility(self, weights) -> np.array:
        return self.get_metrics(weights)[1]

    @staticmethod
    def check_sum(weights) -> float:
        return np.sum(weights) - 1
    
    def optimise(self, show=True):

        def grab_negative_sharpe(weights):
            return self.grab_negative_sharpe(weights)

        optimized_sharpe = sci_opt.minimize(
            grab_negative_sharpe,           # minimize this.
            self.unhedged_weights,          # start with these values.
            method='SLSQP',
            bounds=self.bounds,             # don't exceed these bounds.
            constraints=self.constraints    # make sure you don't exceed the 100% constraint.
        )

        # Print the results.
        print('='*80)
        print('OPTIMIZED SHARPE RATIO:')
        print('-'*80)
        print(optimized_sharpe)
        print('-'*80)

        # Get and save portolio weights
        portfolio_weights = optimized_sharpe.x
        portfolio_stocks = self.top_n_stocks
        self.unhedged_weights = portfolio_weights

        if show:
            # Plot portfolio weights
            plt.figure(dpi=600)
            plt.bar(portfolio_stocks, portfolio_weights)
            plt.tick_params(axis='x', labelrotation=90)
            plt.show()

    def hedge(self):

        # Get covariance of stocks and index (SPX)
        long_cov = self.covariance_matrix.iloc[:-1, -1] # all rows except last, last column
        short_cov = np.zeros((1, 1))

        # Calculate variance of index
        index_var = self.covariance_matrix.iloc[-1, -1]

        # Get beta values for hedging
        # long_betas = 1.0*(long_cov/index_var) # alternative hedging method
        long_betas = 0.2 + 0.8*(long_cov/index_var) 
        short_betas = np.zeros(1)

        # Get weights for long portfolio
        long_weights = self.unhedged_weights
        short_weights = np.array([])

        # Calculate total hedge on long portfolio
        long_hedge = (long_weights * long_betas).sum()
        short_hedge = 0.0

        # Append long hedge to short portfolio
        long_weights = np.append(long_weights, short_hedge)
        short_weights = np.append(short_weights, long_hedge)

        # Normalise portfolio weights
        sum_of_weights = long_weights.sum() + short_weights.sum()
        long_weights = long_weights/sum_of_weights
        short_weights = short_weights/sum_of_weights

        # Convert long and short weights to series
        self.long_weights = pd.Series(index=self.top_n_stocks.append(pd.Index(["SPX"])), data=long_weights)
        self.short_weights = pd.Series(index=pd.Index(["SPX"]), data=short_weights)

    def backtest(self, future_price_df):

        # Get price histories of stocks in our portfolio
        future_long_price_df = future_price_df[self.top_n_stocks.append(pd.Index(["SPX"]))]
        future_short_price_df = future_price_df[pd.Index(["SPX"])]

        # Compute future daily returns, shifted up one day
        future_long_daily_returns = future_long_price_df.pct_change()
        long_returns_tomorrow = future_long_daily_returns.shift(-1)

        future_short_daily_returns = -1.0*future_short_price_df.pct_change()
        short_returns_tomorrow = future_short_daily_returns.shift(-1)

        # Iterate through the date range to calculate returns
        dates = []
        cumulative_long_values = []
        cumulative_short_values = []
        cumulative_portfolio_values = []

        long_weights = self.long_weights.copy()
        short_weights = self.short_weights.copy()

        for idx in range(len(long_returns_tomorrow.index)):

            # Get the date
            date = long_returns_tomorrow.index[idx]

            # Calculate portfolio value today
            cumulative_long_value = long_weights.sum()
            cumulative_short_value = short_weights.sum()

            # Calculate total portfolio value
            cumulative_portfolio_value = cumulative_long_value + cumulative_short_value

            # Set long portfolio value to portfolio value tomorrow
            long_weights = long_weights * (1.0 + long_returns_tomorrow.iloc[idx, :])
            short_weights = short_weights * (1.0 + short_returns_tomorrow.iloc[idx, :])

            # Append daily returns that are not NaNs
            if cumulative_portfolio_value != np.nan:
                dates.append(date)
                cumulative_long_values.append(cumulative_long_value)
                cumulative_short_values.append(cumulative_short_value)
                cumulative_portfolio_values.append(cumulative_portfolio_value)

        # Get the index price history for plotting
        # Rescale for plotting
        scaled_future_spx_df = future_price_df[["SPX"]].to_numpy().reshape(future_price_df.shape[0])
        scaled_future_spx_df = scaled_future_spx_df/scaled_future_spx_df[0]

        # Plot total returns with time
        plt.figure(dpi=500)
        plt.plot(dates, np.asarray(cumulative_long_values/cumulative_long_values[0]), label="Long Value", c="green")
        plt.plot(dates, np.asarray(cumulative_short_values/cumulative_short_values[0]), label="Long Hedge", c="red")
        plt.plot(dates, np.asarray(cumulative_portfolio_values), label="Portfolio Value", c="blue")
        plt.plot(dates, scaled_future_spx_df, label="Index", c="gray")
        plt.tick_params(axis='x', labelrotation=90)
        plt.title("Portfolio value (fraction of initial capital)")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid()
        plt.show()

        # Calculate and print metrics
        index_returns = np.diff(scaled_future_spx_df)/scaled_future_spx_df[:-1]
        portfolio_returns = np.diff(np.asarray(cumulative_portfolio_values))/np.asarray(cumulative_portfolio_values)[:-1]
        cov_matrix = np.cov(index_returns, portfolio_returns)

        index_sharpe = (np.cumprod(1.0+index_returns)[-1]-1.0)/np.sqrt(cov_matrix[0, 0]*252)
        portfolio_sharpe = (np.cumprod(1.0+portfolio_returns)[-1]-1.0)/np.sqrt(cov_matrix[1, 1]*252)

        print("Index Sharpe: ", index_sharpe)
        print("Portfolio Sharpe: ", portfolio_sharpe)
        print("Correlation: ", cov_matrix[0, 1]/np.sqrt(cov_matrix[0, 0]*cov_matrix[1, 1]))