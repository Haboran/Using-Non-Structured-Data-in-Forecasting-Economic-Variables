# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 14:35:16 2025

@author: oskar
"""

#%%
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
import numpy as np
# from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# from functools import reduce
from sklearn.linear_model import LinearRegression
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from pandas.tseries.offsets import QuarterEnd
# from pyMIDAS.regression import MIDASRegression
from sklearn.linear_model import LassoCV
from scipy.optimize import minimize
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import t
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

#%% 
'''PCA to get the two most important sentiments'''
def get_top_2_sentiments(country_code, gdp_series, quarterly_sentiment, sentiment_cols, window_split=(0.4, 0.4, 0.2)):
    # Align sentiment and GDP
    df = quarterly_sentiment[['quarter'] + [f"{country_code}_{topic}" for topic in sentiment_cols]].copy()
    df = df.set_index('quarter')
    gdp_series = gdp_series.copy()
    gdp_series.index = gdp_series.index.to_period('Q')
    df['gdp'] = gdp_series

    df = df.dropna()
    
    # Only use training + adjustment data for selection
    T = len(df)
    split_idx = int((window_split[0] + window_split[1]) * T)
    df_train = df.iloc[:split_idx]

    X = df_train.drop(columns='gdp')
    y = df_train['gdp']

    model = LinearRegression().fit(X, y)
    coefs = abs(model.coef_)
    top2_idx = coefs.argsort()[-2:][::-1]
    top2_features = X.columns[top2_idx].tolist()
    
    return [col.split("_")[1] for col in top2_features]  # return topic names only



#%%       


# Create a MIDAS forecasting function to be integrated with existing models (ARIMA, ARIMAX)
def construct_midas_features(sentiment_series, gdp_dates, lags=90):
    """
    Constructs MIDAS features from high-frequency sentiment data to match GDP quarterly dates.

    Parameters:
    - sentiment_series: daily sentiment time series (pd.Series with datetime index)
    - gdp_dates: pd.DatetimeIndex of GDP observations
    - lags: number of daily lags to use

    Returns:
    - X: np.ndarray of MIDAS regressors (lags as features)
    - valid_dates: matching GDP dates for each row in X
    """
    midas_features = []
    valid_dates = []

    for gdp_date in gdp_dates:
        end_date = pd.Timestamp(gdp_date)
        start_date = end_date - pd.Timedelta(days=lags)
        if start_date < sentiment_series.index[0]:
            continue
        window = sentiment_series.loc[start_date:end_date]
        if len(window) >= lags:
            window = window.tail(lags)  # take only the last N rows if longer
            midas_features.append(window.values[::-1])
            valid_dates.append(end_date)

    return pd.DataFrame(midas_features, index=valid_dates)

def construct_midas_features_pca(sentiment_df_daily, country_code, sentiment_cols, gdp_dates, lags=90):
    """
    Constructs MIDAS features using the first principal component of the 6 sentiment topics.
    """
    daily_df = sentiment_df_daily.set_index("date")
    topic_cols = [f"{country_code}_{col}" for col in sentiment_cols if f"{country_code}_{col}" in daily_df.columns]

    X_pca_input = daily_df[topic_cols].dropna()
    X_pca_input = (X_pca_input - X_pca_input.mean()) / X_pca_input.std()  # standardize

    # Create a principal component time series
    pca = PCA(n_components=1)
    pc1_series = pd.Series(index=X_pca_input.index, data=pca.fit_transform(X_pca_input).flatten())

    # Use original construct_midas_features logic on PC1
    return construct_midas_features(pc1_series, gdp_dates, lags=lags)

def exponential_almon_weights(theta, K):
    exponent = theta[0] * np.arange(1, K + 1) + theta[1] * (np.arange(1, K + 1) ** 2)
    exponent = np.clip(exponent, -50, 50)  # prevent overflow
    weights = np.exp(exponent)
    return weights / np.sum(weights)


def build_lagged_sentiment_matrix(sentiment_df_daily, country_code, sentiment_cols, gdp_dates, lags=90):
    """
    Build a lagged feature matrix for all sentiment variables using a wide format (one row per GDP date).
    """
    sentiment_df_daily = sentiment_df_daily.set_index("date")
    X_all = []
    valid_dates = []

    for gdp_date in gdp_dates:
        end_date = pd.Timestamp(gdp_date)
        start_date = end_date - pd.Timedelta(days=lags)
        window = sentiment_df_daily.loc[start_date:end_date]
        if len(window) >= lags:
            row = []
            for topic in sentiment_cols:
                colname = f"{country_code}_{topic}"
                if colname not in window.columns:
                    continue
                values = window[colname].tail(lags).values[::-1]
                row.extend(values)
            if len(row) == lags * len(sentiment_cols):
                X_all.append(row)
                valid_dates.append(end_date)
    
    return pd.DataFrame(X_all, index=valid_dates)


def diebold_mariano_test(actual, pred1, pred2, h=1, alternative='two-sided'):
    """
    Perform Diebold-Mariano test for equal predictive accuracy.

    Parameters:
    - actual: array-like of true values
    - pred1, pred2: predictions from model 1 and 2
    - h: forecast horizon (default 1)
    - alternative: 'two-sided', 'less', or 'greater'

    Returns:
    - DM statistic and p-value
    """
    e1 = np.array(actual) - np.array(pred1)
    e2 = np.array(actual) - np.array(pred2)

    d = (e1 ** 2) - (e2 ** 2)  # squared error loss differential

    d = d[~np.isnan(d)]  # drop NaNs from unequal horizon alignment
    T = len(d)
    d_bar = np.mean(d)
    gamma = np.sum([(1 - i / h) * np.cov(d[:-i], d[i:])[0, 1] for i in range(1, h)], initial=0)
    var_d = (np.var(d, ddof=1) + 2 * gamma) / T
    dm_stat = d_bar / np.sqrt(var_d)

    # p-value
    if alternative == 'two-sided':
        p = 2 * (1 - t.cdf(np.abs(dm_stat), df=T - 1))
    elif alternative == 'greater':
        p = 1 - t.cdf(dm_stat, df=T - 1)
    else:  # 'less'
        p = t.cdf(dm_stat, df=T - 1)

    return dm_stat, p

#%%
# Define the rolling RMSE plot function again
def rolling_rmse_plot(model_outputs, dates, title="Rolling RMSE Comparison by Model"):
    """
    Plot RMSE over time (rolling evaluation) for multiple models.
    
    Parameters:
    - model_outputs: dict of model name -> (actuals, predictions, date_series)
    - dates: evaluation dates (e.g., from date_eval or shared GDP index)
    - title: plot title
    """
    plt.figure(figsize=(12, 6))
    
    for model_name, (actuals, preds, model_dates) in model_outputs.items():
        if len(actuals) != len(preds):
            continue
        errors = np.array(actuals) - np.array(preds)
        trimmed_index = pd.Index(model_dates[-len(errors):])
        rmse_rolling = pd.Series(errors ** 2, index=trimmed_index).rolling(window=4, min_periods=1).mean() ** 0.5

        rmse_rolling.plot(label=model_name)
    
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return

#%% Uncomment to enable: Actual vs. Forecast Scatterplots
def scatter_actual_vs_pred(model_outputs, title_prefix='Actual vs Predicted'):
    """
    For each model, draw Actual vs. Predicted scatter with a 45° line and date-coloring.
    
    Parameters
    ----------
    model_outputs : dict
        model_name -> (actuals_list, preds_list, dates_list)
    title_prefix : str
        Prefix for each plot title.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    for model_name, (actuals, preds, dates) in model_outputs.items():
        actuals = np.array(actuals)
        preds   = np.array(preds)
        dts     = pd.to_datetime(dates)

        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(actuals, preds, c=dts, cmap='viridis', alpha=0.7)
        # 45° line
        mn = min(actuals.min(), preds.min())
        mx = max(actuals.max(), preds.max())
        ax.plot([mn, mx], [mn, mx], 'k--', linewidth=1)
        ax.set_xlabel('Actual log-diff GDP')
        ax.set_ylabel('Predicted log-diff GDP')
        ax.set_title(f'{title_prefix} — {model_name}')
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Date')
        plt.tight_layout()
        plt.show()


#%% Uncomment to enable: Rolling‐Window RMSE Heatmap
def rolling_rmse_heatmap(model_outputs, window=4, title='Rolling RMSE Heatmap'):
    """
    Build a heatmap of rolling RMSE (window-quarter) for each model over time.
    
    Parameters
    ----------
    model_outputs : dict
        model_name -> (actuals_list, preds_list, dates_list)
    window : int
        Rolling window in number of observations (e.g. 4 quarters).
    title : str
        Figure title.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # compute rolling-RMSE series for each model
    rmse_dict = {}
    all_dates = pd.DatetimeIndex([])
    for name, (acts, preds, dates) in model_outputs.items():
        acts = np.array(acts)
        preds = np.array(preds)
        idx = pd.to_datetime(dates[-len(acts):])
        errs = acts - preds
        rmse = pd.Series(errs**2, index=idx).rolling(window=window, min_periods=1).mean()**0.5
        rmse_dict[name] = rmse
        all_dates = all_dates.union(rmse.index)

    # build DataFrame: rows=models, cols=sorted dates
    all_dates = all_dates.sort_values()
    df = pd.DataFrame({m: s.reindex(all_dates) for m, s in rmse_dict.items()})
    df = df.T  # now index=models, columns=dates

    # plot heatmap
    fig, ax = plt.subplots(figsize=(12, len(df)*0.5 + 1))
    c = ax.imshow(df.values, aspect='auto', interpolation='none')
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index)
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels([d.strftime('%Y-%m') for d in df.columns], rotation=90, fontsize=8)
    ax.set_title(title)
    fig.colorbar(c, ax=ax, label='RMSE')
    plt.tight_layout()
    plt.show()