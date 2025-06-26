# -*- coding: utf-8 -*-
"""
Created on Mon May  5 08:48:30 2025

@author: oskar
"""

#%%
from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')
get_ipython().run_line_magic('clear', '/')

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
from forecast_with_sentiment_models_quarterly_daily import forecast_with_sentiment_models_qd
from forecast_with_sentiment_models_quarterly_monthly import forecast_with_sentiment_models_qm
from Functions import get_top_2_sentiments

#%%
# Utility function for transforming time series data
def transform_series(series, method='log_diff'):
    """
    Transforms a pandas Series based on the specified method.

    Parameters:
    - series (pd.Series): Time series to transform.
    - method (str): Transformation type ('log', 'diff', or 'log_diff').

    Returns:
    - pd.Series: Transformed time series.
    """
    if method == 'log':
        return np.log(series)
    elif method == 'diff':
        return series.diff()
    elif method == 'log_diff':
        return np.log(series).diff()
    else:
        raise ValueError("Choose from 'log', 'diff', or 'log_diff'")



#%% Load GDP data from CSV
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found.*")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

#%%
df = pd.read_csv(r"C:\Users\oskar\Desktop\Uni\4. Mastersemester\Master Thesis\estat_namq_10_gdp_filtered_en (4).csv")
df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'])  # Convert to datetime

#%% Load sentiment data
sentiment_df = pd.read_csv(
    r"C:\Users\oskar\Desktop\Uni\4. Mastersemester\Master Thesis\Data\Barbaglia, L., Consoli, S., & Manzan, S. (2024)\eu_sentiments.csv", 
    parse_dates=['date']
)
#%%
# Define sentiment topics
sentiment_cols = ['economy', 'financial sector', 'inflation', 'manufacturing', 'monetary policy', 'unemployment']

# --- Daily to wide-format pivot ---
# sentiment_df contains ['date','country',<sentiment_cols>]
# Pivot so each country-topic becomes its own column
sentiment_pivot = (
    sentiment_df
    .pivot(index='date', columns='country', values=sentiment_cols)
)
# Flatten MultiIndex columns: ('economy','France')-> 'France_economy'
sentiment_pivot.columns = [f"{country}_{topic}" for topic, country in sentiment_pivot.columns]
# Reset index to bring 'date' back as column
sentiment_pivot = sentiment_pivot.reset_index()

# Ensure 'date' is datetime
sentiment_pivot['date'] = pd.to_datetime(sentiment_pivot['date'])

# --- Create quarterly aggregated DataFrame ---
# 1. Assign quarter periods
sentiment_pivot['quarter'] = sentiment_pivot['date'].dt.to_period('Q')

# 2. Group by quarter and compute mean for each country-topic
#    We exclude 'date' since .mean() on non-numeric will ignore it, but we drop afterwards
quarterly_sentiment = (
    sentiment_pivot
    .groupby('quarter')
    .mean()
    .reset_index()
)

# 3. Map quarter period back to exact quarter-end timestamp
quarterly_sentiment['date'] = (
    quarterly_sentiment['quarter']
    .dt.to_timestamp()      # start of quarter
    + QuarterEnd(0)          # shift to quarter-end
)

#%% Filter GDP data to relevant entries
# First: focus on GDP at market prices
gdp_df = df[df['na_item'] == 'Gross domestic product at market prices']

# Further restrict to seasonally and calendar adjusted data
gdp_df = gdp_df[gdp_df['s_adj'] == 'Seasonally and calendar adjusted data']

# Remove duplicates and average values if multiple entries exist
gdp_df = gdp_df.groupby(['TIME_PERIOD', 'geo'])['OBS_VALUE'].mean().reset_index()

# Ensure datetime format for merging later
gdp_df['TIME_PERIOD'] = pd.to_datetime(gdp_df['TIME_PERIOD'])

# Pivot to wide format: one column per country
gdp_data = gdp_df.pivot(index='TIME_PERIOD', columns='geo', values='OBS_VALUE')

#%%
# First: focus on GDP at market prices
wage_df = df[df['na_item'] == 'Wages and salaries']

# Further restrict to seasonally and calendar adjusted data
wage_df = wage_df[wage_df['s_adj'] == 'Unadjusted data (i.e. neither seasonally adjusted nor calendar adjusted data)']

# Remove duplicates and average values if multiple entries exist
wage_df = wage_df.groupby(['TIME_PERIOD', 'geo'])['OBS_VALUE'].mean().reset_index()

# Ensure datetime format for merging later
wage_df['TIME_PERIOD'] = pd.to_datetime(wage_df['TIME_PERIOD'])

# Pivot to wide format: one column per country
wage_data = wage_df.pivot(index='TIME_PERIOD', columns='geo', values='OBS_VALUE')



#%% Drop unwanted Euro area aggregate columns if present
euro_areas_to_exclude = [
    'Euro area - 19 countries (2015-2022)',
    'Euro area - 19 countries  (2015-2022)',
    'Euro area – 20 countries (from 2023)',  # UTF-8 dash
    'Euro area â€“ 20 countries (from 2023)',  # Incorrect encoding
]

gdp_data = gdp_data.drop(columns=[col for col in euro_areas_to_exclude if col in gdp_data.columns], errors='ignore')
wage_data = wage_data.drop(columns=[col for col in euro_areas_to_exclude if col in wage_data.columns], errors='ignore')
# external_balance_data = gdp_data.drop(columns=[col for col in euro_areas_to_exclude if col in external_balance_data.columns], errors='ignore')

# Limit data to post-1995 only
gdp_data = gdp_data[gdp_data.index >= pd.to_datetime("1995-04-01")]
gdp_data = gdp_data[gdp_data.index <= pd.to_datetime("2024-10-01")]
wage_data = wage_data[wage_data.index >= pd.to_datetime("1995-04-01")]
wage_data = wage_data[wage_data.index <= pd.to_datetime("2024-10-01")]
gdp_data = gdp_data[gdp_data.index >= pd.to_datetime("1995-04-01")]
gdp_data = gdp_data[gdp_data.index <= pd.to_datetime("2024-10-01")]


#%% Load Europe Policy Uncertainty (EPU) data
epu_path = r"C:/Users/oskar/Desktop/Uni/4. Mastersemester/Master Thesis/Data/Baker, S. R., Bloom, N., & Davis, S. J. (2016)/Europe_Policy_Uncertainty_Data.xlsx"
# Read full sheet (wide format)
epu_raw = pd.read_excel(epu_path)
# Construct a datetime index from Year and Month columns
# First drop any footers or metadata rows where Year/Month aren't numeric
epu_raw = epu_raw[pd.to_numeric(epu_raw['Year'], errors='coerce').notnull()]
# Now build date
epu_raw['date'] = pd.to_datetime({
    'year':  epu_raw['Year'].astype(int),
    'month': epu_raw['Month'].astype(int),
    'day':   1
})
# Map wide columns to ISO codes
col_map = {
    'European_News_Index': 'EU',
    'Germany_News_Index':  'DE',
    'Italy_News_Index':    'IT',
    'UK_News_Index':       'UK',
    'France_News_Index':   'FR',
    'Spain_News_Index':    'ES'
}
# Build monthly DataFrame indexed by date with ISO columns
epu_monthly = (
    epu_raw
    .rename(columns=col_map)
    .set_index('date')[list(col_map.values())]
)
# Restrict to Jan 1997–Dec 2024
epu_monthly = epu_monthly.loc['1997-01-01':'2024-12-31']
# Aggregate to quarterly mean and align to quarter-end
epu_quarterly = (
    epu_monthly
    .resample('Q')           
    .mean()
    .reset_index()
)
epu_quarterly['quarter'] = epu_quarterly['date'].dt.to_period('Q')
# THIS yields quarter-start if you only do to_timestamp()
epu_quarterly['date'] = (
    epu_quarterly['quarter']
      .dt
      .to_timestamp()        
      + QuarterEnd(0)
)

#%% Plot GDP time series for each country

gdp_data.plot(figsize=(12, 6), title='GDP Over Time by Country')
plt.xlabel('Date')
plt.ylabel('GDP')
plt.grid(True)
plt.show()


log_gdp_data = transform_series(gdp_data, method='log_diff')
log_gdp_data.plot(figsize=(12, 6), title='log GDP Over Time by Country')
plt.xlabel('Date')
plt.ylabel('GDP')
plt.grid(True)
plt.show()


#%% Plot wage time series for each country

wage_data.plot(figsize=(12, 6), title='Wage Over Time by Country')
plt.xlabel('Date')
plt.ylabel('GDP')
plt.grid(True)
plt.show()


log_wage_data = transform_series(wage_data, method='log_diff')

for country in wage_data.columns:
    try:
        result = seasonal_decompose(wage_data[country].dropna(), model='additive', period=4, extrapolate_trend='freq')
        wage_data[country] = result.trend + result.resid

        
        result2 = seasonal_decompose(log_wage_data[country].dropna(), model='additive', period=4, extrapolate_trend='freq')
        log_wage_data[country] = result2.trend + result2.resid

    except:
        print(f"Skipping decomposition for {country} due to insufficient data or missing values.")
        
wage_data.plot(figsize=(12, 6), title='Wage Over Time by Country')
plt.xlabel('Date')
plt.ylabel('GDP')
plt.grid(True)
plt.show()


# log_wage_data = transform_series(wage_data, method='log_diff')
log_wage_data.plot(figsize=(12, 6), title='log Wage Over Time by Country')
plt.xlabel('Date')
plt.ylabel('GDP')
plt.grid(True)
plt.show()
#%% Plot the Sentiments

# Reload sentiment data to make sure we're working from scratch
sentiment_df = pd.read_csv(r"C:\Users\oskar\Desktop\Uni\4. Mastersemester\Master Thesis\Data\Barbaglia, L., Consoli, S., & Manzan, S. (2024)\eu_sentiments.csv", parse_dates=['date'])

# Set up
country_codes = {'Germany': 'DE', 'France': 'FR', 'Spain': 'ES', 'Italy': 'IT', 'United Kingdom': 'UK'}
focus_sets = {
    "Germany": ['Germany'],
    "France": ['France'],
    "Italy": ['Italy'],
    "Germany & France": ['Germany', 'France']
}
topics = ['economy', 'financial sector', 'inflation', 'manufacturing', 'monetary policy', 'unemployment']

# Add quarter
sentiment_df['quarter'] = sentiment_df['date'].dt.to_period('Q')

# Compute global y-axis limits
all_vals = []
for topic in topics:
    for code in country_codes.values():
        all_vals += sentiment_df[sentiment_df['country'] == code][topic].dropna().tolist()
ymin, ymax = min(all_vals), max(all_vals)

# ---------- All Countries per Topic ----------
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 10), sharex=True, sharey=True)
axs = axs.flatten()

for i, topic in enumerate(topics):
    ax = axs[i]
    for country, code in country_codes.items():
        df_country = sentiment_df[sentiment_df['country'] == code]
        df_topic = df_country[['quarter', topic]].dropna()

        grouped = df_topic.groupby('quarter')[topic]
        q_mean = grouped.mean()
        q_min = grouped.min()
        q_max = grouped.max()

        q_index = q_mean.index.to_timestamp()
        ax.plot(q_index, q_mean.values.astype(float), label=country)
        ax.fill_between(q_index,
                        q_min.reindex(q_mean.index).values.astype(float),
                        q_max.reindex(q_mean.index).values.astype(float),
                        alpha=0.15)

    ax.set_title(topic.capitalize())
    ax.set_ylim(ymin, ymax)
    ax.grid(True)
    if i >= 3:
        ax.set_xlabel("Date")
    if i % 3 == 0:
        ax.set_ylabel("Sentiment Index")

fig.suptitle("Quarterly Sentiment by Topic — All Countries", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
plt.show()

# ---------- Focused Country Plots ----------
for label, countries in focus_sets.items():
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 10), sharex=True, sharey=True)
    axs = axs.flatten()

    for i, topic in enumerate(topics):
        ax = axs[i]
        for country in countries:
            code = country_codes[country]
            df_country = sentiment_df[sentiment_df['country'] == code]
            df_topic = df_country[['quarter', topic]].dropna()

            grouped = df_topic.groupby('quarter')[topic]
            q_mean = grouped.mean()
            q_min = grouped.min()
            q_max = grouped.max()

            q_index = q_mean.index.to_timestamp()
            ax.plot(q_index, q_mean.values.astype(float), label=country)
            ax.fill_between(q_index,
                            q_min.reindex(q_mean.index).values.astype(float),
                            q_max.reindex(q_mean.index).values.astype(float),
                            alpha=0.15)

        ax.set_title(topic.capitalize())
        ax.set_ylim(ymin, ymax)
        ax.grid(True)
        if i >= 3:
            ax.set_xlabel("Date")
        if i % 3 == 0:
            ax.set_ylabel("Sentiment Index")

    fig.suptitle(f"Quarterly Sentiment by Topic — {label}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.legend(loc='upper right')
    plt.show()

#%% Evaluation Methods

# # ADF test function
# def adf_test(series, name=''):
#     result = adfuller(series.dropna())
#     print(f'\nADF Test for {name}')
#     print(f'ADF Statistic: {result[0]}')
#     print(f'p-value: {result[1]}')
#     print('=> Stationary' if result[1] < 0.05 else '=> Non-stationary')

# # ARMA order selection
# def select_arma_order(series, max_p=8, max_q=8):
#     best_aic = np.inf
#     best_order = None
#     series = series.dropna()
#     for p in range(max_p + 1):
#         for q in range(max_q + 1):
#             try:
#                 with warnings.catch_warnings():
#                     warnings.filterwarnings("ignore")
#                     model = ARIMA(series, order=(p, 0, q)).fit()
#                     if model.aic < best_aic:
#                         best_aic = model.aic
#                         best_order = (p, q)
#             except:
#                 continue
#     print(f'Best ARMA order: {best_order} with AIC: {best_aic}')
#     return best_order
   
# # # Run on each country
# # for country in gdp_data.columns:
# #     print(f'\n--- {country} ---')
# #     ts = transform_series(gdp_data[country], method='log_diff')
# #     adf_test(ts, name=country)
# #     select_arma_order(ts)


# # Apply it to all columns
# gdp_data_tf = gdp_data.apply(transform_series, method='log_diff')
# # Plot GDP time series for each country
# gdp_data_tf.plot(figsize=(12, 6), title='GDP Over Time by Country')
# plt.xlabel('Date')
# plt.ylabel('GDP')
# plt.grid(True)
# plt.show()
# '''To evaluate and compare the predictive performance of different models across countries, 
# we loop through each country in the GDP dataset and apply a unified forecasting framework. 
# For each country, the GDP series is first log-differenced to ensure stationarity. 
# A two-letter country code is mapped, and for Italy, the time series is truncated to align with the available sentiment data. 
# The auto_arima function is used to automatically determine the optimal ARIMA order for univariate benchmarks. 
# Next, the top two most predictive sentiment topics are selected based on a linear regression fit to the in-sample GDP data. 
# All six sentiment topics are passed for use in the Lasso regression. The core forecasting is performed by the forecast_with_sentiment_models function, 
# which implements and evaluates five models: ARIMA, ARIMAX, U-MIDAS (unrestricted MIDAS with daily lags), 
# r-MIDAS (restricted MIDAS using exponential Almon lag structure), and Lasso (shrinkage regression over all sentiment lags). 
# The function returns RMSE values for each model, and optionally plots the forecast paths against actual GDP growth. 
# This setup allows consistent, comparative, and extensible forecasting across different model classes and countries.'''

# # Calling the functions

# arimax_results = {}
# gdp_cutoff = pd.to_datetime("2022-06-30")
# wage_cutoff = pd.to_datetime("2022-06-30")

# # Trim GDP and wage data
# gdp_data = gdp_data[gdp_data.index < gdp_cutoff]
# wage_data = wage_data[wage_data.index < wage_cutoff]

# for country in gdp_data.columns:
#     print(f"\n>>> {country} <<<")
    
#     # Transform GDP to log-diff
#     series = transform_series(gdp_data[country], method='log_diff')

#     # Map country to code
#     country_map = {'Germany': 'DE', 'France': 'FR', 'Spain': 'ES', 'Italy': 'IT'}
#     country_code = country_map.get(country)
#     if not country_code:
#         continue

#     # --- make local copies of your sentiment dataframes ---
#     qs = quarterly_sentiment.copy()
#     sp = sentiment_pivot.copy()

#     # Adjust start dates ONLY for Italy due to sentiment coverage
#     if country == "Italy":
#         print(f"{country}: Adjusting sample start due to sentiment coverage")
#         series = series[series.index >= pd.to_datetime("1997-01-01")]
#         qs = qs[qs['quarter'] >= pd.Period("1996Q3")]
#         sp = sp[sp['date']   >= pd.to_datetime("1996-09-05")]

#     # Estimate ARIMA order
#     try:
#         model = auto_arima(series.dropna(), max_p=8, max_d=2, max_q=8,
#                            seasonal=False, stepwise=True,
#                            error_action='ignore', suppress_warnings=True)
#         arima_order = model.order
#     except:
#         arima_order = (1, 0, 1)

#     # Select top 2 sentiment variables (for ARIMAX / MIDAS)
#     top2 = get_top_2_sentiments(country_code, series, qs, sentiment_cols)
#     print(f"{country}: Best sentiment topics = {top2}")

#     # All sentiment topics (for Lasso)
#     sentiment_cols = ['economy', 'financial sector', 'inflation',
#                       'manufacturing', 'monetary policy', 'unemployment']

#     # Run all models, passing the *local* qs & sp
#     rmse_results = forecast_with_sentiment_models_qd(
#         series=series,
#         sentiment_df_quarterly=qs,
#         sentiment_df_daily=sp,
#         country_code=country_code,
#         sentiment_vars=top2,
#         sentiment_cols=sentiment_cols,
#         order=arima_order,
#         forecast_horizon=1,
#         plot=True
#     )

#     print(f"{country} RMSEs: {rmse_results}")


#%%  Evaluate with EPU from 1997-01 to 2024-10
# 1) Trim GDP to 1997-01-01…2024-10-01
gdp_data = gdp_data.loc["1997-01-01":"2024-07-01"]

# 2) Trim EPU series loaded earlier
epu_monthly = epu_monthly.loc["1997-01-01":"2024-10-01"]        # ISO codes as columns: 'DE','FR','ES','IT',…
epu_quarterly = (
    epu_quarterly[
      (epu_quarterly['date'] >= pd.Timestamp("1997-01-01")) &
      (epu_quarterly['date'] <= pd.Timestamp("2024-10-01"))
    ]
    .copy()
)


# 3) Set up loop
country_map = {'Germany':'DE','France':'FR','Spain':'ES','Italy':'IT'}
for country, code in country_map.items():
    if country not in gdp_data:
        continue

    # 4) Slice GDP series by country-specific EPU start
    start = "2001-01-01" if code=="ES" else "1997-01-01"
    series = gdp_data[country].loc[start:"2024-10-01"].dropna()
    series = transform_series(series, method="log_diff")

    # 5) Prepare EPU DataFrames for this country:
    #    - monthly: columns ['date', 'DE_EPU']
    #    - quarterly: columns ['date', 'DE_EPU']
    ms = (
        epu_monthly[[code]]
        .rename(columns={code: f"{code}_EPU"})
        .reset_index()
        .rename(columns={'date':'date'})
    )
    qs = (
        epu_quarterly[['date', code]]
        .rename(columns={code: f"{code}_EPU"})
    )

    # 6) Forecast (one series only: EPU)
    rmse = forecast_with_sentiment_models_qm(
        series=series,
        sentiment_df_quarterly=qs,
        sentiment_df_monthly=ms,
        country_code=code,
        sentiment_vars=["EPU"],    # only one high-freq indicator
        sentiment_cols=["EPU"],    # likewise for Lasso/RF
        order=(1,0,1),             # or your auto_arima order
        forecast_horizon=1,
        plot=True
    )

    print(f"{country} RMSEs: {rmse}")

#%%
'''
Now change the end of the time series to evaluate the models and their estimates pre covid 
for comparison in times with high uncertainty
'''

    
gdp_cutoff = pd.to_datetime("2020-01-01")
wage_cutoff = pd.to_datetime("2020-01-01")
sentiment_cutoff = pd.to_datetime("2020-01-01")

# Trim GDP and wage data
gdp_data = gdp_data[gdp_data.index < gdp_cutoff]
wage_data = wage_data[wage_data.index < wage_cutoff]

# Trim sentiment data
quarterly_sentiment = quarterly_sentiment[quarterly_sentiment['quarter'] < pd.Period("2020Q1")]
sentiment_pivot = sentiment_pivot[sentiment_pivot['date'] < sentiment_cutoff]


arimax_results = {}

for country in gdp_data.columns:
    print(f"\n>>> {country} <<<")
    
    # Transform GDP to log-diff
    series = transform_series(gdp_data[country], method='log_diff')
    # series = transform_series(wage_data[country], method='log_diff')

    # Map country to code
    country_map = {'Germany': 'DE', 'France': 'FR', 'Spain': 'ES', 'Italy': 'IT',  'United Kingdom': 'UK'}
    country_code = country_map.get(country)
    if not country_code:
        continue

    # Adjust start dates ONLY for Italy due to sentiment data availability
    if country == "Italy":
        print(f"{country}: Adjusting sample start due to sentiment coverage")
        series = series[series.index >= pd.to_datetime("1997-01-01")]
        quarterly_sentiment = quarterly_sentiment[quarterly_sentiment['quarter'] >= pd.Period("1996Q3")]
        sentiment_pivot = sentiment_pivot[sentiment_pivot['date'] >= pd.to_datetime("1996-09-05")]

    # Estimate ARIMA order
    try:
        model = auto_arima(series.dropna(), max_p=8, max_d=2, max_q=8, seasonal=False, stepwise=True,
                           error_action='ignore', suppress_warnings=True)
        arima_order = model.order
    except:
        arima_order = (1, 0, 1)

    # Select top 2 sentiment variables (for ARIMAX / MIDAS)
    top2 = get_top_2_sentiments(country_code, series, quarterly_sentiment, sentiment_cols)
    print(f"{country}: Best sentiment topics = {top2}")

    # All sentiment topics (for Lasso)
    sentiment_cols = ['economy', 'financial sector', 'inflation', 'manufacturing', 'monetary policy', 'unemployment']

    # Run all models
    rmse_results = forecast_with_sentiment_models_qd(
        series=series,
        sentiment_df_quarterly=quarterly_sentiment,
        sentiment_df_daily=sentiment_pivot,
        country_code=country_code,
        sentiment_vars=top2,
        sentiment_cols=sentiment_cols,
        order=arima_order,
        plot=True
    )

    print(f"{country} RMSEs: {rmse_results}")
