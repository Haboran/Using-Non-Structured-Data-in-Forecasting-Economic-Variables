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
import matplotlib.dates as mdates

#%% 
def get_top_2_sentiments(country_code, target_series, sentiment_df, sentiment_cols,
                         window_split=(0.4, 0.4, 0.2)):
    """
    Picks the two sentiment topics whose levels best explain target_series
    over the first (window_split[0]+window_split[1]) fraction of the sample.

    Automatically handles either:
      - Quarterly sentiment_df (with a 'quarter' Period column), or
      - Monthly sentiment_df   (with a 'month'  Period column).

    If sentiment_df is quarterly but target_series is monthly, target_series
    is resampled to quarters by taking the last value in each quarter.
    """
    df = sentiment_df.copy()

    # 1) Detect periodicity
    if 'quarter' in df.columns:
        period_col, rule = 'quarter', 'Q'
    elif 'month' in df.columns:
        period_col, rule = 'month',   'M'
    else:
        raise ValueError("sentiment_df needs either a 'quarter' or 'month' column")

    # 2) Subset to your exogs and set index
    exog_cols = [f"{country_code}_{topic}" for topic in sentiment_cols]
    df = df[[period_col] + exog_cols].set_index(period_col)

    # 3) Prepare target_series
    ts = target_series.copy()
    # If sentiment_df is quarterly, aggregate monthly target to quarters
    if rule == 'Q':
        ts = ts.resample('Q').last()
    # Now convert to the same PeriodIndex
    ts.index = ts.index.to_period(rule)

    # 4) Merge
    df['y'] = ts
    df = df.dropna()

    # 5) Split off the training+validation window
    T = len(df)
    split_point = int((window_split[0] + window_split[1]) * T)
    train = df.iloc[:split_point]

    # 6) Fit OLS and pick top-2 by absolute coefficient
    X = train.drop(columns='y')
    y = train['y']
    model = LinearRegression().fit(X, y)
    coefs = abs(model.coef_)
    top2_idx = coefs.argsort()[-2:][::-1]
    top2_feats = X.columns[top2_idx]

    # 7) Return just the topic name (strip off "CC_")
    return [feat.split("_", 1)[1] for feat in top2_feats]



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
def rolling_rmse_heatmap(model_outputs, window=4,
                         start='2017-01-01', end='2024-12-31',
                         title='Rolling RMSE Heatmap'):

    # --- compute rolling-RMSE series (unchanged) ---
    rmse_dict = {}
    for name, (acts, preds, dates) in model_outputs.items():
        acts = np.array(acts)
        preds = np.array(preds)
        idx = pd.to_datetime(dates[-len(acts):])
        errs = acts - preds
        rmse = (pd.Series(errs**2, index=idx)
                   .rolling(window=window, min_periods=1)
                   .mean()**0.5)
        rmse_dict[name] = rmse

    # --- define a common full index 2017–2024 at input frequency ---
    # infer the original freq (quarterly or monthly)
    sample = next(iter(rmse_dict.values()))
    freq  = pd.infer_freq(sample.index) or 'M'
    full_dates = pd.date_range(start, end, freq=freq)

    # --- reindex each series onto the common grid and build DataFrame ---
    df = pd.DataFrame({m: s.reindex(full_dates) for m, s in rmse_dict.items()})
    df = df.T  # rows=models, cols=dates

    # --- mask NaNs so they appear white ---
    data = np.ma.masked_invalid(df.values)
    cmap = plt.get_cmap('viridis', 256)
    cmap.set_bad(color='white')  # masked→white

    # --- plot with a true date axis ---
    fig, ax = plt.subplots(figsize=(12, len(df)*0.5 + 1))
    # convert dates to matplotlib floats
    mdates_vals = mdates.date2num(full_dates.to_pydatetime())
    extent = [mdates_vals[0], mdates_vals[-1], 0, len(df)]
    im = ax.imshow(
        data,
        aspect='auto',
        interpolation='nearest',
        cmap=cmap,
        extent=extent,
        origin='lower',
    )

    # y-axis labels
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index)

    # x-axis: only years
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator())      # every Jan.1st
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim([mdates_vals[0], mdates.date2num(pd.Timestamp(end))])

    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='RMSE')
    plt.tight_layout()
    plt.show()
    
    
    #%%
def heatmap_on_ax(ax, model_outputs, window=4,
                  start='2017-01-01', end='2024-12-31', title=''):
    import numpy as np
    import pandas as pd
    import matplotlib.dates as mdates

    # 1) compute rolling RMSE series for each model
    rmse_dict = {}
    for name, (acts, preds, dates) in model_outputs.items():
        a = np.array(acts)
        p = np.array(preds)

        # ← ALIGN TO SAME LENGTH ←
        n = min(len(a), len(p))
        if n == 0:
            continue
        a = a[-n:]
        p = p[-n:]
        idx = pd.to_datetime(dates[-n:])

        errs = a - p
        rmse = (
            pd.Series(errs**2, index=idx)
              .rolling(window=window, min_periods=1)
              .mean()**0.5
        )
        rmse_dict[name] = rmse

    # 2) align to a common date grid
    sample     = next(iter(rmse_dict.values()))
    freq       = pd.infer_freq(sample.index) or 'M'
    full_dates = pd.date_range(start, end, freq=freq)
    df = pd.DataFrame({m: s.reindex(full_dates) for m, s in rmse_dict.items()}).T

    # 3) reverse row order if desired
    df = df.iloc[::-1]

    # 4) build masked array and plot
    data = np.ma.masked_invalid(df.values)
    cmap = plt.get_cmap('viridis', 256)
    cmap.set_bad(color='white')

    mvals  = mdates.date2num(full_dates.to_pydatetime())
    extent = [mvals[0], mvals[-1], 0, len(df)]
    im = ax.imshow(
        data, aspect='auto', interpolation='nearest',
        cmap=cmap, extent=extent, origin='lower'
    )
    # 5) labels & format
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(extent[0], extent[1])
    return im


def plot_country_rolling_rmse(country,
                              gdp_epu, gdp_figas, 
                              inf_epu, inf_figas, 
                              window=4, start='2017-01-01', end='2024-12-31'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex='col')
    # Top row: EPU
    heatmap_on_ax(axes[0,0], gdp_epu, window, start, end, title=f"EPU GDP")
    heatmap_on_ax(axes[0,1], inf_epu, window, start, end, title=f"EPU Inflation")
    # Middle: FIGAS
    heatmap_on_ax(axes[1,0], gdp_figas, window, start, end, title=f"FIGAS GDP")
    heatmap_on_ax(axes[1,1], inf_figas, window, start, end, title=f"FIGAS Inflation")
    # Bottom: Ashwin
    # heatmap_on_ax(axes[2,0], gdp_ashwin, window, start, end, title=f"Ashwin GDP")
    # heatmap_on_ax(axes[2,1], inf_ashwin, window, start, end, title=f"Ashwin Inflation")

    fig.suptitle(f"Rolling RMSE Heatmaps for {country}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

