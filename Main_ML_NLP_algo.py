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
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# from functools import reduce
from sklearn.linear_model import LinearRegression
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from pandas.tseries.offsets import MonthEnd, QuarterEnd
# from pyMIDAS.regression import MIDASRegression
from sklearn.linear_model import LassoCV
from scipy.optimize import minimize
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import t
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import matplotlib.dates as mdates
import sys
import os

#%%

# Path to your helper functions directory
helper_path = r"C:\Users\oskar\Desktop\Uni\4. Mastersemester\Master Thesis\Coding\Code"

# Add to system path if not already there
if helper_path not in sys.path:
    sys.path.insert(0, helper_path)
from forecast_with_sentiment_models_quarterly_daily import forecast_with_sentiment_models_qd
from forecast_with_sentiment_models_quarterly_monthly import forecast_with_sentiment_models_qm
from forecast_with_sentiment_models_monthly_daily import forecast_with_sentiment_models_md
from forecast_with_sentiment_models_monthly_monthly import forecast_with_sentiment_models_mm
from matplotlib.colors import LinearSegmentedColormap
from Functions import get_top_2_sentiments
from Functions import plot_country_rolling_rmse
from Functions import heatmap_on_ax

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



#%% 
#  Forecast horizon:
h = 12

lag_qu_day = 90
lag_month_day = 360
lag_quarter_month = 3


'''     Target macro variables     '''


'''     GDP - quarterly     '''
#Load GDP data from CSV
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found.*")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

#%%
df = pd.read_csv(r"C:\Users\oskar\Desktop\Uni\4. Mastersemester\Master Thesis\estat_namq_10_gdp_filtered_en (4).csv")
df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'])  # Convert to datetime


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


#%% 

'''    Inflation - monthly '''

# 1. Point to your file
file_path = r"C:\Users\oskar\Desktop\Uni\4. Mastersemester\Master Thesis\Data\prc_hicp_manr__custom_16484139_spreadsheet.xlsx"

# 2. Read the “Data” sheet, using row 9 (0-based index 8) as the header row, and first column as the country label
df_inf = pd.read_excel(
    file_path,
    sheet_name='Data',   # or sheet_name=2
    header=8,            # header is on the 9th row
    index_col=0          # first column has the country names
)

# 3. Drop every column whose header is NaN (those were the blank interleaved columns)
df_inf = df_inf.loc[:, df_inf.columns.notna()]

# 4. Transpose so that dates run down the rows and countries across the columns
df_inf = df_inf.T

# 5. Parse the index as monthly; coerce “Unnamed” → NaT
df_inf.index = pd.to_datetime(df_inf.index, format='%Y-%m', errors='coerce')

# 6. Drop all rows whose index is NaT (these were the blank “Unnamed” rows)
df_inf = df_inf.loc[df_inf.index.notna()]

# 7. Shift the now-clean index to month-ends
df_inf.index = df_inf.index.to_period('M').to_timestamp('M')

# Keep only the first occurrence of each column name
df_inf = df_inf.loc[:, ~df_inf.columns.duplicated()]

# 8. Select only the four countries you need
df_inf = df_inf.loc[:, ["Germany", "Spain", "France", "Italy"]]

# 9. (Optional) Inspect
df_inf = transform_series(df_inf, method='diff')

# for col in df_inf.columns:
#     # Example: use your Germany inflation series
#     y = df_inf[col].dropna()

#     # 1) Run ADF with no automatic lag selection if you want to fix your lag length
#     #    (you can also use autolag='AIC' or 'BIC' if you prefer)
#     result = adfuller(y, maxlag=12, regression='c', autolag=None)

#     adf_stat   = result[0]
#     n_lags     = result[2]
#     n_obs      = result[3]
#     crit_vals  = result[4]    # dict: { '1%': val1, '5%': val5, '10%': val10 }

#     print(f"ADF statistic: {adf_stat:.3f}")
#     print(f"Number of lags used: {n_lags}")
#     print(f"Number of observations: {n_obs}")
#     print("Critical values:")
#     for level, cv in crit_vals.items():
#         print(f"   {level} : {cv:.3f}")

# Assuming df_inf is already defined and contains the four series
fig, ax = plt.subplots()
for col in df_inf.columns:
    ax.plot(df_inf.index, df_inf[col], label=col)
ax.legend()
ax.set_title('EU Inflation (Monthly) for Selected Countries')
ax.set_xlabel('Date')
ax.set_ylabel('Inflation Rate')
plt.tight_layout()
plt.show()
#%%


'''     Sentiments      '''



sentiment_df = pd.read_csv(
    r"C:\Users\oskar\Desktop\Uni\4. Mastersemester\Master Thesis\Data\Barbaglia, L., Consoli, S., & Manzan, S. (2024)\eu_sentiments.csv", 
    parse_dates=['date']
)

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


# 1. Assign month periods
sentiment_pivot['month'] = sentiment_pivot['date'].dt.to_period('M')

# 2. Group by month and compute mean for each country-topic
monthly_sentiment = (
    sentiment_pivot
      .groupby('month')
      .mean()            # numeric cols only, so 'date' dropped automatically
      .reset_index()
)

# 3. Map period back to exact month-end timestamp
monthly_sentiment['date'] = (
    monthly_sentiment['month']
      .dt.to_timestamp()  # gives first-of-month
      + MonthEnd(0)        # shift to month-end
)

# 4. (Optional) Drop the 'month' column if you only need 'date'
monthly_sentiment = monthly_sentiment.drop(columns='month')

#%%

# 0) Define the exact Ashwin sentiments in the file:
ashwin_sentiment_cols = [
    'loughran_sum_fr', 'stability_sum_fr', 'afinn_sum_fr', 'vader_sum_fr', 'econlex_sum_fr',
    'loughran_sum_ge', 'stability_sum_ge', 'afinn_sum_ge', 'vader_sum_ge', 'econlex_sum_ge',
    'loughran_sum_it', 'stability_sum_it', 'afinn_sum_it', 'vader_sum_it', 'econlex_sum_it',
    'loughran_sum_sp', 'stability_sum_sp', 'afinn_sum_sp', 'vader_sum_sp', 'econlex_sum_sp'
]
# ——————————————————————————————————————————

# 1. Peek at header to confirm what's present
header_df    = pd.read_csv(
    r"C:\Users\oskar\Desktop\Uni\4. Mastersemester\Master Thesis\Data\Ashwin, J., Kalamara, E., & Saiz, L. (2024)\euro_daily_export.csv",
    nrows=0
)
existing_cols = set(header_df.columns)
ashwin_cols_to_load = [c for c in ashwin_sentiment_cols if c in existing_cols]

# 2. Load just date + those sentiment series
ashwin_daily_country_sentiment_df = pd.read_csv(
    r"C:\Users\oskar\Desktop\Uni\4. Mastersemester\Master Thesis\Data\Ashwin, J., Kalamara, E., & Saiz, L. (2024)\euro_daily_export.csv",
    usecols=['date'] + ashwin_cols_to_load
)

# 3. Ensure 'date' is datetime
ashwin_daily_country_sentiment_df['date'] = pd.to_datetime(
    ashwin_daily_country_sentiment_df['date'],
    dayfirst=True,    # if needed
    errors='coerce'
)

# 4. Create quarter & month helpers
ashwin_daily_country_sentiment_df['quarter'] = (
    ashwin_daily_country_sentiment_df['date']
    .dt.to_period('Q')
)
ashwin_daily_country_sentiment_df['month'] = (
    ashwin_daily_country_sentiment_df['date']
    .dt.to_period('M')
)

# 5. Build rename map to go from "<prefix>_sum_<suffix>" → "<CC>_<prefix>"
suffix_to_code = {'fr':'FR','ge':'DE','it':'IT','sp':'ES'}
sum_prefixes   = ['loughran_sum','stability_sum','afinn_sum','vader_sum','econlex_sum']

rename_map = {}
for pref in sum_prefixes:
    topic = pref.replace('_sum','')    # drop the "_sum"
    for suf, code in suffix_to_code.items():
        old = f"{pref}_{suf}"
        new = f"{code}_{topic}"
        if old in ashwin_daily_country_sentiment_df.columns:
            rename_map[old] = new

# 6. Apply renaming
ashwin_daily_country_sentiment_df.rename(columns=rename_map, inplace=True)

# 7. Define the new sentiment column names
sentiment_cols_renamed = list(rename_map.values())
sentiment_cols_ashwin = ['loughran', 'stability', 'afinn', 'vader', 'econlex']
# 8. Aggregate to quarterly
ashwin_quarterly_country_sentiment_df = (
    ashwin_daily_country_sentiment_df
    .groupby('quarter')[sentiment_cols_renamed]
    .mean()
    .reset_index()
)
ashwin_quarterly_country_sentiment_df['date'] = (
    ashwin_quarterly_country_sentiment_df['quarter']
    .dt.to_timestamp() + QuarterEnd(0)
)

# 9. Aggregate to monthly
ashwin_monthly_country_sentiment_df = (
    ashwin_daily_country_sentiment_df
      .groupby('month')[sentiment_cols_renamed]
      .mean()
      .reset_index()      # preserves 'month'
)
ashwin_monthly_country_sentiment_df['date'] = (
    ashwin_monthly_country_sentiment_df['month']
      .dt.to_timestamp()  # first of month
    + MonthEnd(0)         # end of month
)

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
country_codes = {'Germany': 'DE', 'France': 'FR', 'Spain': 'ES', 'Italy': 'IT'}
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

#%%
'''To evaluate and compare the predictive performance of different models across countries, 
we loop through each country in the GDP dataset and apply a unified forecasting framework. 
For each country, the GDP series is first log-differenced to ensure stationarity. 
A two-letter country code is mapped, and for Italy, the time series is truncated to align with the available sentiment data. 
The auto_arima function is used to automatically determine the optimal ARIMA order for univariate benchmarks. 
Next, the top two most predictive sentiment topics are selected based on a linear regression fit to the in-sample GDP data. 
All six sentiment topics are passed for use in the Lasso regression. The core forecasting is performed by the forecast_with_sentiment_models function, 
which implements and evaluates five models: ARIMA, ARIMAX, U-MIDAS (unrestricted MIDAS with daily lags), 
r-MIDAS (restricted MIDAS using exponential Almon lag structure), and Lasso (shrinkage regression over all sentiment lags). 
The function returns RMSE values for each model, and optionally plots the forecast paths against actual GDP growth. 
This setup allows consistent, comparative, and extensible forecasting across different model classes and countries.'''

''' GDP Figas'''
# Calling the functions

arimax_results = {}
all_summary_rmse_gdp_figas = {}
all_raw_outputs_gdp_figas   = {}
all_combo_outputs_gdp_figas = {}


gdp_cutoff = pd.to_datetime("2022-06-30")
wage_cutoff = pd.to_datetime("2022-06-30")

# Trim GDP and wage data
gdp_data_tmp = gdp_data[gdp_data.index < gdp_cutoff]
wage_data = wage_data[wage_data.index < wage_cutoff]

for country in gdp_data.columns:
    print(f"\n>>> {country} <<<")
    
    # Transform GDP to log-diff
    series = transform_series(gdp_data_tmp[country], method='log_diff')
    # Map country to code
    country_map = {'Germany': 'DE', 'France': 'FR', 'Spain': 'ES', 'Italy': 'IT'}
    country_code = country_map.get(country)
    if not country_code:
        continue

    # --- make local copies of your sentiment dataframes ---
    qs = quarterly_sentiment.copy()
    sp = sentiment_pivot.copy()

    # Adjust start dates ONLY for Italy due to sentiment coverage
    if country == "Italy":
        print(f"{country}: Adjusting sample start due to sentiment coverage")
        series = series[series.index >= pd.to_datetime("1997-01-01")]
        qs = qs[qs['quarter'] >= pd.Period("1996Q3")]
        sp = sp[sp['date']   >= pd.to_datetime("1996-09-05")]

    # Estimate ARIMA order
    try:
        model = auto_arima(series.dropna(), max_p=8, max_d=2, max_q=8,
                            seasonal=False, stepwise=True,
                            error_action='ignore', suppress_warnings=True)
        arima_order = model.order
    except:
        arima_order = (h,0,h)

    # Select top 2 sentiment variables (for ARIMAX / MIDAS)
    top2 = get_top_2_sentiments(country_code, series, qs, sentiment_cols)
    print(f"{country}: Best sentiment topics = {top2}")

    # All sentiment topics (for Lasso)
    sentiment_cols = ['economy', 'financial sector', 'inflation',
                      'manufacturing', 'monetary policy', 'unemployment']

    # Run all models, passing the *local* qs & sp
    results = forecast_with_sentiment_models_qd(
        series=series,
        sentiment_df_quarterly=qs,
        sentiment_df_daily=sp,
        country_code=country_code,
        sentiment_vars=top2,
        sentiment_cols=sentiment_cols,
        order=arima_order,
        forecast_horizon=h,
        lags=lag_qu_day,
        plot=True
    )

    print(f"{country} RMSEs: {results['summary_rmse']}")
    all_summary_rmse_gdp_figas[country] = results['summary_rmse']
    all_raw_outputs_gdp_figas[country]   = results['raw_outputs']
    all_combo_outputs_gdp_figas[country] = results['combo_outputs']

#%%
''' Inf Figas'''
   
cutoff = pd.to_datetime("2022-06-30")

inf_data = df_inf[df_inf.index < cutoff]
msent    = monthly_sentiment[monthly_sentiment['date'] < cutoff]
dsent    = sentiment_pivot[sentiment_pivot['date'] < cutoff]


all_summary_rmse_inf_figas = {}
all_raw_outputs_inf_figas   = {}
all_combo_outputs_inf_figas = {}
# --- 3) Loop over each country and call the forecast function ---
results_monthly = {}
country_map = {'Germany':'DE','France':'FR','Spain':'ES','Italy':'IT'}

for country in inf_data.columns:
    print(f"\n>>> {country} <<<")
    series = inf_data[country].dropna()
    # Map to code
    code = country_map.get(country)
    if code is None:
        continue

    # For Italy: align start date if necessary
    ms = msent.copy()
    ds = dsent.copy()
    if country == "Italy":
        print(" Italy: trimming to match sentiment coverage")
        series = series[series.index >= pd.to_datetime("1997-01-01")]
        ms     = ms[ms['date'] >= pd.to_datetime("1996-09-30")]
        ds     = ds[ds['date'] >= pd.to_datetime("1996-09-05")]

    # 1) Fit auto_arima to pick (p,d,q) on the inflation series
    try:
        am = auto_arima(series, seasonal=False, stepwise=True,
                        error_action='ignore', suppress_warnings=True,
                        max_p=24, max_d=2, max_q=24)
        order = am.order
    except:
        order = (h,0,h)

    # 2) Pick your top-2 sentiment topics however you like; here we just hardcode two:
    top2 = get_top_2_sentiments(code, series, ms, sentiment_cols)
    print(f"{country}: Best sentiment topics = {top2}")
    
    # 3) Call the monthly/daily function
    results = forecast_with_sentiment_models_md(
        series=series,
        sentiment_df_monthly=ms,
        sentiment_df_daily=ds,
        country_code=code,
        sentiment_vars=top2,
        sentiment_cols=['economy','financial sector','inflation',
                        'manufacturing','monetary policy','unemployment'],
        order=order,
        forecast_horizon=h,
        lags=lag_month_day,
        plot=True
    )

    print(f"{country} RMSEs: {results['summary_rmse']}")
    all_summary_rmse_inf_figas[country] = results['summary_rmse']
    all_raw_outputs_inf_figas[country]   = results['raw_outputs']
    all_combo_outputs_inf_figas[country] = results['combo_outputs']
    
    
#%%  Evaluate with EPU from 1997-01 to 2024-10
''' GDP EPU'''
# 1) Trim GDP to 1997-01-01…2024-10-01
gdp_data_tmp = gdp_data.loc["1997-01-01":"2024-07-01"]

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

all_summary_rmse_gdp_epu = {}
all_raw_outputs_gdp_epu   = {}
all_combo_outputs_gdp_epu = {}


country_map = {'Germany':'DE','France':'FR','Spain':'ES','Italy':'IT'}
for country, code in country_map.items():
    print(f"\n>>> {country} <<<")
    if country not in gdp_data:
        continue

    # 4) Slice GDP series by country-specific EPU start
    start = "2001-01-01" if code=="ES" else "1997-01-01"
    series = gdp_data_tmp[country].loc[start:"2024-10-01"].dropna()
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
    # 1) Fit auto_arima to pick (p,d,q) on the inflation series
    try:
        am = auto_arima(series, seasonal=False, stepwise=True,
                        error_action='ignore', suppress_warnings=True,
                        max_p=8, max_d=2, max_q=8)
        order = am.order
    except:
        order = (h,0,h)
    # 6) Forecast (one series only: EPU)
    results = forecast_with_sentiment_models_qm(
        series=series,
        sentiment_df_quarterly=qs,
        sentiment_df_monthly=ms,
        country_code=code,
        sentiment_vars=["EPU"],    # only one high-freq indicator
        sentiment_cols=["EPU"],    # likewise for Lasso/RF
        order=order,             # or your auto_arima order
        forecast_horizon=h,
        lags=lag_quarter_month,
        plot=True
    )

    print(f"{country} RMSEs: {results['summary_rmse']}")
    all_summary_rmse_gdp_epu[country] = results['summary_rmse']
    all_raw_outputs_gdp_epu[country]   = results['raw_outputs']
    all_combo_outputs_gdp_epu[country] = results['combo_outputs']
    
    
#%%
'''    Inf EPU'''
# 1) Cutoff
cutoff = pd.to_datetime("2024-12-01")
inf_data = df_inf[df_inf.index < cutoff]
# or if its index is a PeriodIndex (freq='M'), convert:
epu_monthly['date'] = epu_monthly.index + MonthEnd(0)

# 2) Prepare result containers
all_summary_rmse_inf_epu = {}
all_raw_outputs_inf_epu  = {}
all_combo_outputs_inf_epu  = {}

country_map = {'Germany':'DE','France':'FR','Spain':'ES','Italy':'IT'}
for country, code in country_map.items():
    if country not in gdp_data:
        continue

    print(f"\n>>> {country} <<<")
    # 1) Definiere deine beiden Grenzen
    start = "2001-01-01" if code=="ES" else "1997-01-01"
    sent_start = pd.to_datetime(start)
    sent_end   = pd.to_datetime("2024-12-01")
    
    # 2) Baue die Ziel‐Serie y so, dass sie nur in diesem Intervall lebt
    y = inf_data[country].dropna().copy()
    # auf Monatsende verschieben
    y.index = y.index.to_period('M').to_timestamp() + MonthEnd(0)
    # nun lower + upper cut
    y = y.loc[(y.index >= sent_start) & (y.index <= sent_end)]
    
    # 3) Baue dein EPU‐DF so, dass es nur in diesem Intervall lebt
    if 'date' in epu_monthly.columns:
        ms = epu_monthly[['date', code]].copy()
    else:
        ms = epu_monthly.reset_index().rename(columns={'index':'date'})[['date', code]].copy()
    ms['date'] = pd.to_datetime(ms['date']) + MonthEnd(0)
    ms = ms.rename(columns={code: f"{code}_EPU"})
    ms = ms.loc[(ms['date'] >= sent_start) & (ms['date'] <= sent_end)]
    
    # 4) Intersection (eigentlich überflüssig, weil du schon beschnitten hast)
    common = y.index.intersection(ms['date'])
    y      = y.loc[common]
    ms     = ms[ms['date'].isin(common)].copy()

    try:
        am = auto_arima(y, seasonal=False, stepwise=True,
                        error_action='ignore', suppress_warnings=True,
                        max_p=24, max_d=2, max_q=24)
        order = am.order
    except:
        order = (h,0,h)

    # d) Call the monthly–monthly function
    results = forecast_with_sentiment_models_mm(
        series=y,
        sentiment_df_monthly=ms,
        country_code=code,
        sentiment_vars=["EPU"],      # for ARIMAX / DL
        sentiment_cols=["EPU"],      # for LASSO / RF
        order=order,
        forecast_horizon=h,
        lags=1,
        plot=True
    )

    # e) Store
    print(f"{country} RMSEs: {results['summary_rmse']}")
    all_summary_rmse_inf_epu[country] = results['summary_rmse']
    all_raw_outputs_inf_epu[country]  = results['raw_outputs']
    all_combo_outputs_inf_epu[country] = results['combo_outputs']
    
    

#%%
''' GDP Ashwin'''
# 4) Filter to your common sentiment sample 2002-01-02 → 2020-10-01
sent_start = pd.to_datetime("2002-01-02")
sent_end   = pd.to_datetime("2020-01-01")

ashwin_daily_sent_filt = ashwin_daily_country_sentiment_df[
    (ashwin_daily_country_sentiment_df['date'] >= sent_start) &
    (ashwin_daily_country_sentiment_df['date'] <= sent_end)
].copy()

ashwin_quarterly_sent_filt = ashwin_quarterly_country_sentiment_df[
    (ashwin_quarterly_country_sentiment_df['date'] >= sent_start) &
    (ashwin_quarterly_country_sentiment_df['date'] <= sent_end)
].copy()

# 5) Trim your GDP data
gdp_cutoff = pd.to_datetime("2020-01-01")
gdp_data_tmp   = gdp_data[gdp_data.index < gdp_cutoff]

# 6) Now your loop, selecting top-2 by quarterly correlation of the raw *_sum_<cc> names
# --- GDP Ashwin: loop over each sentiment individually ---
all_summary_rmse_gdp_ashwin_indiv = {}
all_raw_outputs_gdp_ashwin_indiv   = {}
all_combo_outputs_gdp_ashwin_indiv = {}

sentiments = sentiment_cols_ashwin.copy()         # ['loughran','stability','afinn','vader','econlex']
country_map = {'Germany':'DE','France':'FR','Spain':'ES','Italy':'IT'}

for country in gdp_data.columns:
    print(f"\n>>> {country} — Ashwin (individual sentiments) <<<")
    # 1) log-diff transform & trim to your sentiment window
    series = (
        transform_series(gdp_data_tmp[country], method='log_diff')
        .dropna()
        .loc[sent_start:sent_end]
    )

    code = country_map.get(country)
    if not code:
        continue

    # 2) local copies of Ashwin sentiment
    qs = ashwin_quarterly_sent_filt.copy()
    ds = ashwin_daily_sent_filt.copy()

    # 3) Italy’s later start
    if country == "Italy":
        series = series.loc["1997-01-01":]
        qs     = qs[qs['quarter'] >= pd.Period("1996Q3")]
        ds     = ds[ds['date']   >= pd.to_datetime("1996-09-05")]

    # 4) pick ARIMA order on the truncated series
    try:
        m = auto_arima(series, seasonal=False, stepwise=True,
                       max_p=8, max_d=2, max_q=8,
                       error_action='ignore', suppress_warnings=True)
        arima_order = m.order
    except:
        arima_order = (h,0,h)

    # 5) loop through each sentiment one by one
    for sent in sentiments:
        print(f"  → Sentiment = {sent}")
        results = forecast_with_sentiment_models_qd(
            series=series,
            sentiment_df_quarterly=qs,
            sentiment_df_daily=ds,
            country_code=code,
            sentiment_vars=[sent],      # only this one
            sentiment_cols=[sent],      # for Lasso/RF use the same single
            order=arima_order,
            forecast_horizon=h,
            lags=lag_qu_day,
            plot=True
        )

        # 6) store under nested dicts
        all_summary_rmse_gdp_ashwin_indiv.setdefault(country, {})[sent] = results['summary_rmse']
        all_raw_outputs_gdp_ashwin_indiv.setdefault(country, {})[sent]   = results['raw_outputs']
        all_combo_outputs_gdp_ashwin_indiv.setdefault(country, {})[sent] = results['combo_outputs']
    
    
#%%

''' Inf Ashwin'''
# 4) Filter to your common sentiment sample 2002-01-02 → 2020-10-01
sent_start = pd.to_datetime("2002-01-02")
sent_end   = pd.to_datetime("2020-01-01")

cutoff = pd.to_datetime("2020-01-01")

inf_data = df_inf[df_inf.index < cutoff]


# ensure the index is datetime or period type
ashwin_monthly_country_sentiment_df.index = pd.to_datetime(
    ashwin_monthly_country_sentiment_df.index
)

ashwin_monthly = ashwin_monthly_country_sentiment_df[
    (ashwin_monthly_country_sentiment_df['date'] >= sent_start) &
    (ashwin_monthly_country_sentiment_df['date'] <= sent_end)
].copy()


ashwin_daily = ashwin_daily_country_sentiment_df[
    (ashwin_daily_country_sentiment_df['date'] >= sent_start) &
    (ashwin_daily_country_sentiment_df['date'] <= sent_end)
].copy()

# --- Inf Ashwin: loop over each sentiment individually ---
all_summary_rmse_inf_ashwin_indiv = {}
all_raw_outputs_inf_ashwin_indiv  = {}
all_combo_outputs_inf_ashwin_indiv = {}

for country in inf_data.columns:
    print(f"\n>>> {country} — Ashwin (individual sentiments) <<<")
    # 1) Build & trim your inflation series
    series = (
        inf_data[country]
        .dropna()
        .loc[sent_start:sent_end]
    )

    code = country_map.get(country)
    if code is None:
        continue

    # 2) Local copies of Ashwin sentiment
    ms = ashwin_monthly.copy()
    ds = ashwin_daily.copy()

    # 4) Auto-ARIMA on the (trimmed) inflation series
    try:
        am    = auto_arima(series, seasonal=False, stepwise=True,
                           error_action='ignore', suppress_warnings=True,
                           max_p=24, max_d=2, max_q=24)
        order = am.order
    except:
        order = (h,0,h)

    # 5) Loop through each Ashwin sentiment one by one
    for sent in sentiment_cols_ashwin:
        print(f"  → Sentiment = {sent}")
        results = forecast_with_sentiment_models_md(
            series             = series,
            sentiment_df_monthly = ms,
            sentiment_df_daily   = ds,
            country_code       = code,
            sentiment_vars     = [sent],      # only this one
            sentiment_cols     = [sent],      # same for Lasso/RF
            order              = order,
            forecast_horizon   = h,
            lags               = lag_month_day,
            plot               = True
        )

        # 6) Store under nested dicts: country → sentiment
        all_summary_rmse_inf_ashwin_indiv.setdefault(country, {})[sent] = results['summary_rmse']
        all_raw_outputs_inf_ashwin_indiv.setdefault(country, {})[sent]  = results['raw_outputs']
        all_combo_outputs_inf_ashwin_indiv.setdefault(country, {})[sent] = results['combo_outputs']


    
#%%
''' Pre covid    GDP FIGAS'''

# --- GDP Figas: mixed-frequency models on pre-COVID data ---
all_summary_rmse_gdp_figas_pre_covid = {}
all_raw_outputs_gdp_figas_pre_covid   = {}
all_combo_outputs_gdp_figas_pre_covid = {}

# define pre-COVID cutoff (up to end of 2019)
gdp_pre_covid_cutoff = pd.to_datetime("2020-01-01")
gdp_data_pre_covid   = gdp_data[gdp_data.index < gdp_pre_covid_cutoff]

for country in gdp_data.columns:
    print(f"\n>>> {country} <<<")
    
    # Transform GDP to log-diff
    series = transform_series(gdp_data_pre_covid[country], method='log_diff')
    # Map country to code
    country_map = {'Germany': 'DE', 'France': 'FR', 'Spain': 'ES', 'Italy': 'IT'}
    country_code = country_map.get(country)
    if not country_code:
        continue

    # --- make local copies of your sentiment dataframes ---
    qs = quarterly_sentiment.copy()
    sp = sentiment_pivot.copy()

    # Adjust start dates ONLY for Italy due to sentiment coverage
    if country == "Italy":
        print(f"{country}: Adjusting sample start due to sentiment coverage")
        series = series[series.index >= pd.to_datetime("1997-01-01")]
        qs = qs[qs['quarter'] >= pd.Period("1996Q3")]
        sp = sp[sp['date']   >= pd.to_datetime("1996-09-05")]

    # Estimate ARIMA order
    try:
        model = auto_arima(series.dropna(), max_p=8, max_d=2, max_q=8,
                            seasonal=False, stepwise=True,
                            error_action='ignore', suppress_warnings=True)
        arima_order = model.order
    except:
        arima_order = (h,0,h)

    # Select top 2 sentiment variables (for ARIMAX / MIDAS)
    top2 = get_top_2_sentiments(country_code, series, qs, sentiment_cols)
    print(f"{country}: Best sentiment topics = {top2}")

    # All sentiment topics (for Lasso)
    sentiment_cols = ['economy', 'financial sector', 'inflation',
                      'manufacturing', 'monetary policy', 'unemployment']

    # Run all models, passing the *local* qs & sp
    results_pre = forecast_with_sentiment_models_qd(
        series=series,
        sentiment_df_quarterly=qs,
        sentiment_df_daily=sp,
        country_code=country_code,
        sentiment_vars=top2,
        sentiment_cols=sentiment_cols,
        order=arima_order,
        forecast_horizon=h,
        lags=lag_qu_day,
        plot=True
    )

    # 8) store under new dict names
    print(f"{country} RMSEs: {results_pre['summary_rmse']}")
    all_summary_rmse_gdp_figas_pre_covid[country] = results_pre['summary_rmse']
    all_raw_outputs_gdp_figas_pre_covid[country]   = results_pre['raw_outputs']
    all_combo_outputs_gdp_figas_pre_covid[country] = results_pre['combo_outputs']
    
    
    
#%%    
'''  Inf Figas'''

# --- Inf Figas: monthly/daily models on pre-COVID data ---
all_summary_rmse_inf_figas_pre_covid = {}
all_raw_outputs_inf_figas_pre_covid   = {}
all_combo_outputs_inf_figas_pre_covid = {}

# define cutoff for pre-COVID (up through 2019)
inf_pre_covid_cutoff = pd.to_datetime("2020-01-01")

# truncate your target & sentiment dfs
inf_data_pre_covid = df_inf[df_inf.index < inf_pre_covid_cutoff]
msent_pre_covid   = monthly_sentiment[monthly_sentiment['date'] < inf_pre_covid_cutoff]
dsent_pre_covid   = sentiment_pivot[sentiment_pivot['date'] < inf_pre_covid_cutoff]

country_map = {'Germany':'DE','France':'FR','Spain':'ES','Italy':'IT'}

for country in inf_data_pre_covid.columns:
    print(f"\n>>> {country} <<<")
    series = inf_data_pre_covid[country].dropna()
    # Map to code
    code = country_map.get(country)
    if code is None:
        continue

    # For Italy: align start date if necessary
    ms = msent.copy()
    ds = dsent.copy()
    if country == "Italy":
        print(" Italy: trimming to match sentiment coverage")
        series = series[series.index >= pd.to_datetime("1997-01-01")]
        ms     = msent_pre_covid[msent_pre_covid['date'] >= pd.to_datetime("1996-09-30")]
        ds     = dsent_pre_covid[dsent_pre_covid['date'] >= pd.to_datetime("1996-09-05")]

    # 1) Fit auto_arima to pick (p,d,q) on the inflation series
    try:
        am = auto_arima(series, seasonal=False, stepwise=True,
                        error_action='ignore', suppress_warnings=True,
                        max_p=24, max_d=2, max_q=24)
        order = am.order
    except:
        order = (h,0,h)

    # 2) Pick your top-2 sentiment topics however you like; here we just hardcode two:
    top2 = get_top_2_sentiments(code, series, ms, sentiment_cols)
    print(f"{country}: Best sentiment topics = {top2}")
    
    # 3) Call the monthly/daily function
    results_pre = forecast_with_sentiment_models_md(
        series=series,
        sentiment_df_monthly=ms,
        sentiment_df_daily=ds,
        country_code=code,
        sentiment_vars=top2,
        sentiment_cols=['economy','financial sector','inflation',
                        'manufacturing','monetary policy','unemployment'],
        order=order,
        forecast_horizon=h,
        lags=lag_month_day,
        plot=True
    )

    # 8) store under new dicts
    print(f"{country} RMSEs: {results_pre['summary_rmse']}")
    all_summary_rmse_inf_figas_pre_covid[country] = results_pre['summary_rmse']
    all_raw_outputs_inf_figas_pre_covid[country]   = results_pre['raw_outputs']
    all_combo_outputs_inf_figas_pre_covid[country] = results_pre['combo_outputs']


#%%
'''GDP EPU'''

# --- GDP EPU: quarterly–monthly models on pre-COVID data ---
all_summary_rmse_gdp_epu_pre_covid = {}
all_raw_outputs_gdp_epu_pre_covid   = {}
all_combo_outputs_gdp_epu_pre_covid = {}

# define pre-COVID cutoff (up through December 2019)
gdp_epu_pre_covid_cutoff = pd.to_datetime("2020-01-01")

# truncate your GDP and EPU series
gdp_data_pre_covid       = gdp_data[gdp_data.index < gdp_epu_pre_covid_cutoff]
epu_monthly_pre_covid    = epu_monthly[epu_monthly.index < gdp_epu_pre_covid_cutoff]
epu_quarterly_pre_covid  = epu_quarterly[epu_quarterly['date'] < gdp_epu_pre_covid_cutoff].copy()

country_map = {'Germany':'DE','France':'FR','Spain':'ES','Italy':'IT'}

for country, code in country_map.items():
    if country not in gdp_data_pre_covid:
        continue

    print(f"\n>>> {country} EPU (pre-COVID) <<<")

    # 1) slice & transform GDP
    start = "2001-01-01" if code == "ES" else "1997-01-01"
    series = gdp_data_pre_covid[country].loc[start:].dropna()
    series = transform_series(series, method="log_diff")

    # 2) build country-specific EPU dfs
    ms = (
        epu_monthly_pre_covid[[code]]
        .rename(columns={code: f"{code}_EPU"})
        .reset_index()
        .rename(columns={"index":"date"})
    )
    qs = (
        epu_quarterly_pre_covid[["date", code]]
        .rename(columns={code: f"{code}_EPU"})
    )
    try:
        am = auto_arima(series, seasonal=False, stepwise=True,
                        error_action='ignore', suppress_warnings=True,
                        max_p=8, max_d=2, max_q=8)
        order = am.order
    except:
        order = (h,0,h)
    # 3) run mixed-frequency quarterly–monthly MIDAS
    results_pre = forecast_with_sentiment_models_qm(
        series=series,
        sentiment_df_quarterly=qs,
        sentiment_df_monthly=ms,
        country_code=code,
        sentiment_vars=["EPU"],
        sentiment_cols=["EPU"],
        order=order,           # or your auto_arima order
        forecast_horizon=h,
        lags=lag_quarter_month,
        plot=True
    )

    # 4) store under new dicts
    print(f"{country} RMSEs: {results_pre['summary_rmse']}")
    all_summary_rmse_gdp_epu_pre_covid[country] = results_pre["summary_rmse"]
    all_raw_outputs_gdp_epu_pre_covid[country]   = results_pre["raw_outputs"]
    all_combo_outputs_gdp_epu_pre_covid[country] = results_pre["combo_outputs"]


#%%
'''Inf EPU'''


# --- Inf EPU: quarterly–monthly models on pre-COVID data ---
all_summary_rmse_inf_epu_pre_covid = {}
all_raw_outputs_inf_epu_pre_covid   = {}
all_combo_outputs_inf_epu_pre_covid = {}

# define pre-COVID cutoff (up through December 2019)
inf_epu_pre_covid_cutoff = pd.to_datetime("2020-01-01")

# truncate your inflation and EPU series
inf_data_pre_covid      = df_inf[df_inf.index < inf_epu_pre_covid_cutoff]
epu_monthly_pre_covid   = epu_monthly[epu_monthly.index < inf_epu_pre_covid_cutoff]
epu_quarterly_pre_covid = epu_quarterly[epu_quarterly['date'] < inf_epu_pre_covid_cutoff].copy()

country_map = {'Germany':'DE','France':'FR','Spain':'ES','Italy':'IT'}

for country, code in country_map.items():
    if country not in gdp_data:
        continue

    print(f"\n>>> {country} <<<")
    # 1) Definiere deine beiden Grenzen
    start = "2001-01-01" if code=="ES" else "1997-01-01"
    sent_start = pd.to_datetime(start)
    sent_end   = pd.to_datetime("2020-01-01")
    
    # 2) Baue die Ziel‐Serie y so, dass sie nur in diesem Intervall lebt
    y = inf_data[country].dropna().copy()
    # auf Monatsende verschieben
    y.index = y.index.to_period('M').to_timestamp() + MonthEnd(0)
    # nun lower + upper cut
    y = y.loc[(y.index >= sent_start) & (y.index <= sent_end)]
    
    # 3) Baue dein EPU‐DF so, dass es nur in diesem Intervall lebt
    if 'date' in epu_monthly.columns:
        ms = epu_monthly[['date', code]].copy()
    else:
        ms = epu_monthly.reset_index().rename(columns={'index':'date'})[['date', code]].copy()
    ms['date'] = pd.to_datetime(ms['date']) + MonthEnd(0)
    ms = ms.rename(columns={code: f"{code}_EPU"})
    ms = ms.loc[(ms['date'] >= sent_start) & (ms['date'] <= sent_end)]
    
    # 4) Intersection (eigentlich überflüssig, weil du schon beschnitten hast)
    common = y.index.intersection(ms['date'])
    y      = y.loc[common]
    ms     = ms[ms['date'].isin(common)].copy()

    try:
        am = auto_arima(y, seasonal=False, stepwise=True,
                        error_action='ignore', suppress_warnings=True,
                        max_p=24, max_d=2, max_q=24)
        order = am.order
    except:
        order = (h,0,h)

    # d) Call the monthly–monthly function
    results_pre = forecast_with_sentiment_models_mm(
        series=y,
        sentiment_df_monthly=ms,
        country_code=code,
        sentiment_vars=["EPU"],      # for ARIMAX / DL
        sentiment_cols=["EPU"],      # for LASSO / RF
        order=order,
        forecast_horizon=h,
        lags=1,
        plot=True
    )

    # 5) store under new dicts
    print(f"{country} RMSEs: {results_pre['summary_rmse']}")
    all_summary_rmse_inf_epu_pre_covid[country] = results_pre["summary_rmse"]
    all_raw_outputs_inf_epu_pre_covid[country]   = results_pre["raw_outputs"]
    all_combo_outputs_inf_epu_pre_covid[country] = results_pre["combo_outputs"]

#%%
# 1) Per-country RMSE min/max for GDP and Inflation
def collect_country_rmse_vals(datasets, country, window=4):
    all_vals = []
    for dataset in datasets:
        outputs = dataset[country]
        if isinstance(outputs, dict) and 'vader' in outputs:  # nested
            for key in outputs:
                for _, (acts, preds, _) in outputs[key].items():
                    a, p = np.array(acts), np.array(preds)
                    n = min(len(a), len(p))
                    if n == 0:
                        continue
                    rmse = pd.Series((a[-n:] - p[-n:])**2).rolling(window, min_periods=1).mean()**0.5
                    all_vals.extend(rmse.values)
        else:
            for _, (acts, preds, _) in outputs.items():
                a, p = np.array(acts), np.array(preds)
                n = min(len(a), len(p))
                if n == 0:
                    continue
                rmse = pd.Series((a[-n:] - p[-n:])**2).rolling(window, min_periods=1).mean()**0.5
                all_vals.extend(rmse.values)
    return np.nanmin(all_vals), np.nanmax(all_vals)

countries = ['Germany', 'France', 'Spain', 'Italy']
window = 4
start, end = '2017-01-01', '2019-12-31'


# Custom colormap: blue → yellow → red
custom_cmap = LinearSegmentedColormap.from_list(
    'custom_heat', ['darkblue', 'blue', 'cyan', 'yellow', 'orange', 'red', 'darkred']
)

# Dataset groupings
gdp_datasets = [
    all_raw_outputs_gdp_epu_pre_covid,
    all_raw_outputs_gdp_figas_pre_covid,
    all_raw_outputs_gdp_ashwin_indiv
]

inf_datasets = [
    all_raw_outputs_inf_epu_pre_covid,
    all_raw_outputs_inf_figas_pre_covid,
    all_raw_outputs_inf_ashwin_indiv
]

# Custom colormap
custom_cmap = LinearSegmentedColormap.from_list(
    'custom_heat', ['darkblue', 'blue', 'cyan', 'yellow', 'orange', 'red', 'darkred']
)

# 2) Plot per country with the two cbar positions tweaked:
for country in countries:
    vmin_inf, vmax_inf = collect_country_rmse_vals(inf_datasets, country, window)
    vmin_gdp, vmax_gdp = collect_country_rmse_vals(gdp_datasets, country, window)

    fig, axes = plt.subplots(5, 2, figsize=(14, 20), sharex='col')
    fig.suptitle(f"{country} — Pre-COVID Rolling RMSE Heatmaps", fontsize=16)

    # --- Inflation column (left) ---
    heatmap_on_ax(axes[0, 0], all_raw_outputs_inf_epu_pre_covid[country],
                  window, start, end, "Inflation – EPU",
                  vmin=vmin_inf, vmax=vmax_inf, cmap=custom_cmap)
    heatmap_on_ax(axes[1, 0], all_raw_outputs_inf_figas_pre_covid[country],
                  window, start, end, "Inflation – FIGAS",
                  vmin=vmin_inf, vmax=vmax_inf, cmap=custom_cmap)
    heatmap_on_ax(axes[2, 0], all_raw_outputs_inf_ashwin_indiv[country]['vader'],
                  window, start, end, "Inflation – VADER",
                  vmin=vmin_inf, vmax=vmax_inf, cmap=custom_cmap)
    heatmap_on_ax(axes[3, 0], all_raw_outputs_inf_ashwin_indiv[country]['stability'],
                  window, start, end, "Inflation – Stability",
                  vmin=vmin_inf, vmax=vmax_inf, cmap=custom_cmap)
    im_inf = heatmap_on_ax(axes[4, 0], all_raw_outputs_inf_ashwin_indiv[country]['econlex'],
                           window, start, end, "Inflation – EconLex",
                           vmin=vmin_inf, vmax=vmax_inf, cmap=custom_cmap)

    # --- GDP column (right) ---
    im_gdp = heatmap_on_ax(axes[0, 1], all_raw_outputs_gdp_epu_pre_covid[country],
                           window, start, end, "GDP – EPU",
                           vmin=vmin_gdp, vmax=vmax_gdp, cmap=custom_cmap)
    heatmap_on_ax(axes[1, 1], all_raw_outputs_gdp_figas_pre_covid[country],
                  window, start, end, "GDP – FIGAS",
                  vmin=vmin_gdp, vmax=vmax_gdp, cmap=custom_cmap)
    heatmap_on_ax(axes[2, 1], all_raw_outputs_gdp_ashwin_indiv[country]['vader'],
                  window, start, end, "GDP – VADER",
                  vmin=vmin_gdp, vmax=vmax_gdp, cmap=custom_cmap)
    heatmap_on_ax(axes[3, 1], all_raw_outputs_gdp_ashwin_indiv[country]['stability'],
                  window, start, end, "GDP – Stability",
                  vmin=vmin_gdp, vmax=vmax_gdp, cmap=custom_cmap)
    heatmap_on_ax(axes[4, 1], all_raw_outputs_gdp_ashwin_indiv[country]['econlex'],
                  window, start, end, "GDP – EconLex",
                  vmin=vmin_gdp, vmax=vmax_gdp, cmap=custom_cmap)
    heatmap_axes = [
        (axes[0, 0], all_raw_outputs_inf_epu_pre_covid[country]),
        (axes[1, 0], all_raw_outputs_inf_figas_pre_covid[country]),
        (axes[2, 0], all_raw_outputs_inf_ashwin_indiv[country]['vader']),
        (axes[3, 0], all_raw_outputs_inf_ashwin_indiv[country]['stability']),
        (axes[4, 0], all_raw_outputs_inf_ashwin_indiv[country]['econlex']),
        (axes[0, 1], all_raw_outputs_gdp_epu_pre_covid[country]),
        (axes[1, 1], all_raw_outputs_gdp_figas_pre_covid[country]),
        (axes[2, 1], all_raw_outputs_gdp_ashwin_indiv[country]['vader']),
        (axes[3, 1], all_raw_outputs_gdp_ashwin_indiv[country]['stability']),
        (axes[4, 1], all_raw_outputs_gdp_ashwin_indiv[country]['econlex']),
    ]
    for ax, data in heatmap_axes:
        labels = list(data.keys())[::-1]
        n = len(labels)
        # Move the ticks to the center of each heatmap row
        ax.set_yticks(np.arange(n) + 0.5)
        ax.set_yticklabels(labels, va='center', rotation=0)
    # Format x-axes
    for ax in axes.flatten():
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', which='major', labelrotation=45, labelsize=8)
        ax.set_xlabel("Year")

    # --- Colorbars ---
    # Inflation: BETWEEN the two columns
    cax_inf = fig.add_axes([0.45, 0.15, 0.015, 0.7])
    cb_inf = fig.colorbar(im_inf, cax=cax_inf)
    cb_inf.set_label("RMSE (Inflation)", rotation=270, labelpad=15)

    # GDP: RIGHT of GDP column
    cax_gdp = fig.add_axes([0.98, 0.15, 0.015, 0.7])
    cb_gdp = fig.colorbar(im_gdp, cax=cax_gdp)
    cb_gdp.set_label("RMSE (GDP)", rotation=270, labelpad=15)

    plt.tight_layout(rect=[0, 0.03, 0.90, 0.96])
    
    fig.subplots_adjust(
        left=0.05,    # keep a little breathing room on the left
        right=0.95,   # push the whole grid to use up to 88% of the width
        wspace=0.4,   # increase gap between col 0 and col 1
        top=0.96,
        bottom=0.03
    )
    plt.show()
#%%
for country in ['Germany', 'France', 'Spain', 'Italy']:
    plot_country_rolling_rmse(
        country=country,
        gdp_epu=all_raw_outputs_gdp_epu[country],
        gdp_figas=all_raw_outputs_gdp_figas[country],
      #  gdp_ashwin=all_raw_outputs_gdp_ashwin[country],
        inf_epu=all_raw_outputs_inf_epu[country],
        inf_figas=all_raw_outputs_inf_figas[country],
      #  inf_ashwin=all_raw_outputs_inf_ashwin[country],
        window=4,
        start='2017-01-01',
        end='2024-12-31'
    )
    
for country in ['Germany', 'France', 'Spain', 'Italy']:
    plot_country_rolling_rmse(
        country=country,
        gdp_epu=all_combo_outputs_gdp_epu[country],
        gdp_figas=all_combo_outputs_gdp_figas[country],
     #   gdp_ashwin=all_combo_outputs_gdp_ashwin[country],
        inf_epu=all_combo_outputs_inf_epu[country],
        inf_figas=all_combo_outputs_inf_figas[country],
      #  inf_ashwin=all_combo_outputs_inf_ashwin[country],
        window=4,
        start='2017-01-01',
        end='2024-12-31'
    )    
#%%

countries    = ["Germany","France","Spain","Italy"]
model_order  = ["ARIMA","ARIMAX","U-MIDAS","LASSO","RF","MIDAS-Net","r-MIDAS"]
sentiments   = ['loughran','stability','afinn','vader','econlex']
post_covid   = pd.to_datetime("2020-01-01")
output_path  = r"C:\Users\oskar\Desktop\Uni\4. Mastersemester\Master Thesis\Tables\all_tables.tex"

with open(output_path, "w") as f:
    # 1) Post-COVID single-model absolute
    for country in countries:
        df = pd.DataFrame({
            "GDP Figas": pd.Series(all_summary_rmse_gdp_figas[country]),
            "GDP EPU":   pd.Series(all_summary_rmse_gdp_epu[country]),
            "INF Figas": pd.Series(all_summary_rmse_inf_figas[country]),
            "INF EPU":   pd.Series(all_summary_rmse_inf_epu[country]),
        })
        f.write(f"% === {country} — Single-model RMSEs (post-COVID, absolute) ===\n")
        f.write(df.to_latex(float_format="%.4f", na_rep="--",
                            caption=f"Post-COVID single-model RMSEs for {country} (absolute)",
                            label=f"tab:{country.lower()}_rmse_single_post_abs"))
        f.write("\n\n")
    # 2) Post-COVID single-model relative
    for country in countries:
        df = pd.DataFrame({
            "GDP Figas": pd.Series(all_summary_rmse_gdp_figas[country]),
            "GDP EPU":   pd.Series(all_summary_rmse_gdp_epu[country]),
            "INF Figas": pd.Series(all_summary_rmse_inf_figas[country]),
            "INF EPU":   pd.Series(all_summary_rmse_inf_epu[country]),
        }).reindex(model_order)
        baseline = df.loc["ARIMA"]
        df_rel = df.div(baseline, axis=1)
        f.write(f"% === {country} — Single-model RMSEs (post-COVID, relative) ===\n")
        f.write(df_rel.to_latex(float_format="%.2f", na_rep="--",
                                caption=f"Post-COVID single-model RMSEs for {country} (relative to ARIMA)",
                                label=f"tab:{country.lower()}_rmse_single_post_rel"))
        f.write("\n\n")
    # 3) Post-COVID combo-model absolute
    for country in countries:
        combos = {
            "GDP Figas": all_combo_outputs_gdp_figas[country],
            "GDP EPU":   all_combo_outputs_gdp_epu[country],
            "INF Figas": all_combo_outputs_inf_figas[country],
            "INF EPU":   all_combo_outputs_inf_epu[country],
        }
        combo_names = sorted(
            set().union(*(d.keys() for d in combos.values()))
        )
        df_combo = pd.DataFrame(index=combo_names, columns=combos.keys(), dtype=float)
        for col, cd in combos.items():
            for name, (acts, preds, dates) in cd.items():
                a,p = np.array(acts), np.array(preds)
                n    = min(len(a), len(p))
                rmse = np.sqrt(((a[-n:]-p[-n:])**2).mean()) if n else np.nan
                df_combo.at[name, col] = rmse
                
        # replace underscores in the index with spaces
        df_combo.index = df_combo.index.str.replace("_", " ")

        f.write(f"% === {country} — Combination-model RMSEs (post-COVID, absolute) ===\n")
        f.write(df_combo.to_latex(float_format="%.4f", na_rep="--",
                                  caption=f"Post-COVID combination RMSEs for {country} (absolute)",
                                  label=f"tab:{country.lower()}_rmse_combo_post_abs"))
        f.write("\n\n")
    # 4) Post-COVID combo-model relative
    for country in countries:
        combos = {
            "GDP Figas": all_combo_outputs_gdp_figas[country],
            "GDP EPU":   all_combo_outputs_gdp_epu[country],
            "INF Figas": all_combo_outputs_inf_figas[country],
            "INF EPU":   all_combo_outputs_inf_epu[country],
        }
        combo_names = sorted(set().union(*(d.keys() for d in combos.values())))
        df_combo = pd.DataFrame(index=combo_names, columns=combos.keys(), dtype=float)

        for col, cd in combos.items():
            for name, (acts, preds, dates) in cd.items():
                a = np.array(acts)
                p = np.array(preds)
                n = min(len(a), len(p))
                rmse = np.sqrt(((a[-n:] - p[-n:])**2).mean()) if n else np.nan
                df_combo.at[name, col] = rmse

        # Normalize using same ARIMA summary values as single-model
        baseline = {
            "GDP Figas": all_summary_rmse_gdp_figas[country]["ARIMA"],
            "GDP EPU":   all_summary_rmse_gdp_epu[country]["ARIMA"],
            "INF Figas": all_summary_rmse_inf_figas[country]["ARIMA"],
            "INF EPU":   all_summary_rmse_inf_epu[country]["ARIMA"],
        }
        df_rel = df_combo.div(pd.Series(baseline), axis=1)

        df_combo.index = df_combo.index.str.replace("_", " ")
        df_rel.index = df_rel.index.str.replace("_", " ")

        f.write(f"% === {country} — Combination-model RMSEs (post-COVID, relative) ===\n")
        f.write(df_rel.to_latex(
            float_format="%.2f", na_rep="--",
            caption=f"Post-COVID combination RMSEs for {country} (relative to ARIMA)",
            label=f"tab:{country.lower()}_rmse_combo_post_rel"
        ))
        f.write("\n\n")
    
    # 5) Pre-COVID single-model absolute
    for country in countries:
        df = pd.DataFrame({
            "GDP Figas": pd.Series(all_summary_rmse_gdp_figas_pre_covid[country]),
            "GDP EPU":   pd.Series(all_summary_rmse_gdp_epu_pre_covid[country]),
            "INF Figas": pd.Series(all_summary_rmse_inf_figas_pre_covid[country]),
            "INF EPU":   pd.Series(all_summary_rmse_inf_epu_pre_covid[country]),
        })
        f.write(f"% === {country} — Single-model RMSEs (pre-COVID, absolute) ===\n")
        f.write(df.to_latex(float_format="%.4f", na_rep="--",
                            caption=f"Pre-COVID single-model RMSEs for {country} (absolute)",
                            label=f"tab:{country.lower()}_rmse_single_pre_abs"))
        f.write("\n\n")
    # 6) Pre-COVID single-model relative
    for country in countries:
        df = pd.DataFrame({
            "GDP Figas": pd.Series(all_summary_rmse_gdp_figas_pre_covid[country]),
            "GDP EPU":   pd.Series(all_summary_rmse_gdp_epu_pre_covid[country]),
            "INF Figas": pd.Series(all_summary_rmse_inf_figas_pre_covid[country]),
            "INF EPU":   pd.Series(all_summary_rmse_inf_epu_pre_covid[country]),
        }).reindex(model_order)
        baseline = df.loc["ARIMA"]
        df_rel = df.div(baseline, axis=1)
        f.write(f"% === {country} — Single-model RMSEs (pre-COVID, relative) ===\n")
        f.write(df_rel.to_latex(float_format="%.2f", na_rep="--",
                                caption=f"Pre-COVID single-model RMSEs for {country} (relative to ARIMA)",
                                label=f"tab:{country.lower()}_rmse_single_pre_rel"))
        f.write("\n\n")
    # 7) Pre-COVID combo-model absolute
    for country in countries:
        combos_pre = {
            "GDP Figas": all_combo_outputs_gdp_figas_pre_covid[country],
            "GDP EPU":   all_combo_outputs_gdp_epu_pre_covid[country],
            "INF Figas": all_combo_outputs_inf_figas_pre_covid[country],
            "INF EPU":   all_combo_outputs_inf_epu_pre_covid[country],
        }
        combo_names = sorted(
            set().union(*(d.keys() for d in combos_pre.values()))
        )
        df_pre = pd.DataFrame(index=combo_names, columns=combos_pre.keys(), dtype=float)
        for col, cd in combos_pre.items():
            for name, (acts, preds, dates) in cd.items():
                a,p = np.array(acts), np.array(preds)
                n    = min(len(a), len(p))
                rmse = np.sqrt(((a[-n:]-p[-n:])**2).mean()) if n else np.nan
                df_pre.at[name, col] = rmse
        # replace underscores in the index with spaces
        df_pre.index = df_pre.index.str.replace("_", " ")
        f.write(f"% === {country} — Combination-model RMSEs (pre-COVID, absolute) ===\n")
        f.write(df_pre.to_latex(float_format="%.4f", na_rep="--",
                                caption=f"Pre-COVID combination RMSEs for {country} (absolute)",
                                label=f"tab:{country.lower()}_rmse_combo_pre_abs"))
        f.write("\n\n")
    # 8) Pre-COVID combo-model relative
    for country in countries:
        combos_pre = {
            "GDP Figas": all_combo_outputs_gdp_figas_pre_covid[country],
            "GDP EPU":   all_combo_outputs_gdp_epu_pre_covid[country],
            "INF Figas": all_combo_outputs_inf_figas_pre_covid[country],
            "INF EPU":   all_combo_outputs_inf_epu_pre_covid[country],
        }
        combo_names = sorted(
            set().union(*(d.keys() for d in combos_pre.values()))
        )
        df_pre = pd.DataFrame(index=combo_names, columns=combos_pre.keys(), dtype=float)

        for col, combo_dict in combos_pre.items():
            for name, (acts, preds, dates) in combo_dict.items():
                a = np.array(acts)
                p = np.array(preds)
                n_min = min(len(a), len(p))
                # compute RMSE over the aligned pre-COVID series
                rmse = np.sqrt(np.mean((a[-n_min:] - p[-n_min:])**2)) if n_min else np.nan
                df_pre.at[name, col] = rmse

        baseline = {
            "GDP Figas": all_summary_rmse_gdp_figas_pre_covid[country]["ARIMA"],
            "GDP EPU":   all_summary_rmse_gdp_epu_pre_covid[country]["ARIMA"],
            "INF Figas": all_summary_rmse_inf_figas_pre_covid[country]["ARIMA"],
            "INF EPU":   all_summary_rmse_inf_epu_pre_covid[country]["ARIMA"],
        }
        df_rel = df_pre.div(pd.Series(baseline), axis=1)

        # replace underscores in the index with spaces
        df_rel.index = df_rel.index.str.replace("_", " ")

        f.write(f"% === {country} — Combination-model RMSEs (pre-COVID, relative) ===\n")
        f.write(df_rel.to_latex(
            float_format="%.2f", na_rep="--",
            caption=f"Pre-COVID combination RMSEs for {country}, relative to ARIMA",
            label=f"tab:{country.lower()}_rmse_combo_pre_rel",
            column_format="lcccc"
        ))
        f.write("\n\n")
    
    # 5) Post-COVID Ashwin sentiment RMSEs, split into GDP vs INF
    for country in countries:
        # --- GDP only ---
        # single-model absolute
        df_gdp_abs = pd.DataFrame(
            { s: pd.Series(all_summary_rmse_gdp_ashwin_indiv[country][s])
              for s in sentiments },
            index=model_order
        )
        # single-model relative
        df_gdp_rel = df_gdp_abs.div(df_gdp_abs.loc["ARIMA"], axis=1)

        # combo-model absolute
        combos_gdp = all_combo_outputs_gdp_ashwin_indiv[country]
        combo_names = sorted(set().union(*(d.keys() for d in combos_gdp.values())))
        df_gdp_combo_abs = pd.DataFrame(index=combo_names, columns=sentiments, dtype=float)
        for s, cd in combos_gdp.items():
            for name, (acts,preds,_) in cd.items():
                a,p = np.array(acts), np.array(preds)
                n   = min(len(a),len(p))
                df_gdp_combo_abs.at[name,s] = np.sqrt(((a[-n:]-p[-n:])**2).mean()) if n else np.nan

        # combo-model relative
        df_gdp_combo_rel = df_gdp_combo_abs.div(df_gdp_abs.loc["ARIMA"], axis=1)

        # write GDP tables
        f.write(f"% === {country} — Ashwin GDP (single, absolute) ===\n")
        f.write(df_gdp_abs.to_latex(
            float_format="%.4f", na_rep="--",
            caption=f"Post-COVID Ashwin GDP RMSEs for {country} (absolute)",
            label=f"tab:{country.lower()}_ashwin_gdp_post_abs",
            column_format="lccccc"
        )); f.write("\n\n")

        f.write(f"% === {country} — Ashwin GDP (single, relative) ===\n")
        f.write(df_gdp_rel.to_latex(
            float_format="%.2f", na_rep="--",
            caption=f"Post-COVID Ashwin GDP RMSEs for {country} (relative to ARIMA)",
            label=f"tab:{country.lower()}_ashwin_gdp_post_rel",
            column_format="lccccc"
        )); f.write("\n\n")
        
        # replace underscores in the index with spaces
        df_gdp_combo_abs.index = df_gdp_combo_abs.index.str.replace("_", " ")
        f.write(f"% === {country} — Ashwin GDP (combo, absolute) ===\n")
        f.write(df_gdp_combo_abs.to_latex(
            float_format="%.4f", na_rep="--",
            caption=f"Post-COVID Ashwin GDP combo RMSEs for {country} (absolute)",
            label=f"tab:{country.lower()}_ashwin_gdp_post_combo_abs",
            column_format="lccccc"
        )); f.write("\n\n")
        
        # replace underscores in the index with spaces
        df_gdp_combo_rel.index = df_gdp_combo_rel.index.str.replace("_", " ")
        f.write(f"% === {country} — Ashwin GDP (combo, relative) ===\n")
        f.write(df_gdp_combo_rel.to_latex(
            float_format="%.2f", na_rep="--",
            caption=f"Post-COVID Ashwin GDP combo RMSEs for {country} (relative to ARIMA)",
            label=f"tab:{country.lower()}_ashwin_gdp_post_combo_rel",
            column_format="lccccc"
        )); f.write("\n\n")

        # --- INF only ---
        # single-model absolute
        df_inf_abs = pd.DataFrame(
            { s: pd.Series(all_summary_rmse_inf_ashwin_indiv[country][s])
              for s in sentiments },
            index=model_order
        )
        # single-model relative
        df_inf_rel = df_inf_abs.div(df_inf_abs.loc["ARIMA"], axis=1)

        # combo-model absolute
        combos_inf = all_combo_outputs_inf_ashwin_indiv[country]
        combo_names_inf = sorted(set().union(*(d.keys() for d in combos_inf.values())))
        df_inf_combo_abs = pd.DataFrame(index=combo_names_inf, columns=sentiments, dtype=float)
        for s, cd in combos_inf.items():
            for name, (acts,preds,_) in cd.items():
                a,p = np.array(acts), np.array(preds)
                n   = min(len(a),len(p))
                df_inf_combo_abs.at[name,s] = np.sqrt(((a[-n:]-p[-n:])**2).mean()) if n else np.nan

        # combo-model relative
        df_inf_combo_rel = df_inf_combo_abs.div(df_inf_abs.loc["ARIMA"], axis=1)

        # write INF tables
        f.write(f"% === {country} — Ashwin INF (single, absolute) ===\n")
        f.write(df_inf_abs.to_latex(
            float_format="%.4f", na_rep="--",
            caption=f"Post-COVID Ashwin INF RMSEs for {country} (absolute)",
            label=f"tab:{country.lower()}_ashwin_inf_post_abs",
            column_format="lccccc"
        )); f.write("\n\n")

        f.write(f"% === {country} — Ashwin INF (single, relative) ===\n")
        f.write(df_inf_rel.to_latex(
            float_format="%.2f", na_rep="--",
            caption=f"Post-COVID Ashwin INF RMSEs for {country} (relative to ARIMA)",
            label=f"tab:{country.lower()}_ashwin_inf_post_rel",
            column_format="lccccc"
        )); f.write("\n\n")
        
        # replace underscores in the index with spaces
        df_inf_combo_abs.index = df_inf_combo_abs.index.str.replace("_", " ")
        f.write(f"% === {country} — Ashwin INF (combo, absolute) ===\n")
        f.write(df_inf_combo_abs.to_latex(
            float_format="%.4f", na_rep="--",
            caption=f"Post-COVID Ashwin INF combo RMSEs for {country} (absolute)",
            label=f"tab:{country.lower()}_ashwin_inf_post_combo_abs",
            column_format="lccccc"
        )); f.write("\n\n")
        
        # replace underscores in the index with spaces
        df_inf_combo_rel.index = df_inf_combo_rel.index.str.replace("_", " ")
        f.write(f"% === {country} — Ashwin INF (combo, relative) ===\n")
        f.write(df_inf_combo_rel.to_latex(
            float_format="%.2f", na_rep="--",
            caption=f"Post-COVID Ashwin INF combo RMSEs for {country} (relative to ARIMA)",
            label=f"tab:{country.lower()}_ashwin_inf_post_combo_rel",
            column_format="lccccc"
        )); f.write("\n\n")
        
    
print(f"All tables written to {output_path}")

#%%
import joblib
import os

def save_model_outputs(output_dir=r"C:\Users\oskar\Desktop\Uni\4. Mastersemester\Master Thesis\Tables", filename="model_outputs_H_12.joblib"):
    """
    Saves all relevant forecast and evaluation output variables to disk.
    """
    os.makedirs(output_dir, exist_ok=True)

    variables_to_save = {
        # # === Post-COVID ===
        "all_summary_rmse_gdp_figas": all_summary_rmse_gdp_figas,
        "all_raw_outputs_gdp_figas": all_raw_outputs_gdp_figas,
        "all_combo_outputs_gdp_figas": all_combo_outputs_gdp_figas,

        "all_summary_rmse_inf_figas": all_summary_rmse_inf_figas,
        "all_raw_outputs_inf_figas": all_raw_outputs_inf_figas,
        "all_combo_outputs_inf_figas": all_combo_outputs_inf_figas,

        "all_summary_rmse_gdp_epu": all_summary_rmse_gdp_epu,
        "all_raw_outputs_gdp_epu": all_raw_outputs_gdp_epu,
        "all_combo_outputs_gdp_epu": all_combo_outputs_gdp_epu,

        "all_summary_rmse_inf_epu": all_summary_rmse_inf_epu,
        "all_raw_outputs_inf_epu": all_raw_outputs_inf_epu,
        "all_combo_outputs_inf_epu": all_combo_outputs_inf_epu,

        "all_summary_rmse_gdp_ashwin_indiv": all_summary_rmse_gdp_ashwin_indiv,
        "all_raw_outputs_gdp_ashwin_indiv": all_raw_outputs_gdp_ashwin_indiv,
        "all_combo_outputs_gdp_ashwin_indiv": all_combo_outputs_gdp_ashwin_indiv,

        "all_summary_rmse_inf_ashwin_indiv": all_summary_rmse_inf_ashwin_indiv,
        "all_raw_outputs_inf_ashwin_indiv": all_raw_outputs_inf_ashwin_indiv,
        "all_combo_outputs_inf_ashwin_indiv": all_combo_outputs_inf_ashwin_indiv,

        # === Pre-COVID ===
        "all_summary_rmse_gdp_figas_pre_covid": all_summary_rmse_gdp_figas_pre_covid,
        "all_raw_outputs_gdp_figas_pre_covid": all_raw_outputs_gdp_figas_pre_covid,
        "all_combo_outputs_gdp_figas_pre_covid": all_combo_outputs_gdp_figas_pre_covid,

        "all_summary_rmse_inf_figas_pre_covid": all_summary_rmse_inf_figas_pre_covid,
        "all_raw_outputs_inf_figas_pre_covid": all_raw_outputs_inf_figas_pre_covid,
        "all_combo_outputs_inf_figas_pre_covid": all_combo_outputs_inf_figas_pre_covid,

        "all_summary_rmse_gdp_epu_pre_covid": all_summary_rmse_gdp_epu_pre_covid,
        "all_raw_outputs_gdp_epu_pre_covid": all_raw_outputs_gdp_epu_pre_covid,
        "all_combo_outputs_gdp_epu_pre_covid": all_combo_outputs_gdp_epu_pre_covid,

        "all_summary_rmse_inf_epu_pre_covid": all_summary_rmse_inf_epu_pre_covid,
        "all_raw_outputs_inf_epu_pre_covid": all_raw_outputs_inf_epu_pre_covid,
        "all_combo_outputs_inf_epu_pre_covid": all_combo_outputs_inf_epu_pre_covid,
    }

    full_path = os.path.join(output_dir, filename)
    joblib.dump(variables_to_save, full_path)
    print(f"✅ All model outputs saved to: {full_path}")
    
save_model_outputs()
#%%

import joblib

loaded_data = joblib.load(r"C:\Users\oskar\Desktop\Uni\4. Mastersemester\Master Thesis\Tables\model_outputs.joblib")

# Unpack each variable into the global namespace
globals().update(loaded_data)

print("✅ All variables successfully loaded.")