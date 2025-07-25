# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 14:35:16 2025

@author: oskar
"""

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import t
from matplotlib.colors import LinearSegmentedColormap

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
    """
    Generate a heatmap of rolling RMSE values for multiple models over time.

    model_outputs : dict
        Mapping from model name to a tuple of (actuals, predictions, dates).
    window : int
        Number of periods to include in each rolling RMSE calculation.
    start, end : str or datetime-like
        Time span for the heatmap’s horizontal axis.
    title : str
        Title for the resulting plot.
    """

    # 1) Calculate a time series of rolling RMSE for each model
    rmse_dict = {}
    for name, (actuals, preds, dates) in model_outputs.items():
        # compute per-period error, then rolling-window RMSE
        errs = np.array(actuals) - np.array(preds)
        idx  = pd.to_datetime(dates[-len(errs):])
        rmse = (pd.Series(errs**2, index=idx)
                   .rolling(window=window, min_periods=1)
                   .mean()
                   .pow(0.5))
        rmse_dict[name] = rmse

    # 2) Create a unified date index covering all models
    #    at the appropriate frequency (monthly/quarterly)
    sample_series = next(iter(rmse_dict.values()))
    freq          = pd.infer_freq(sample_series.index) or 'M'
    full_dates    = pd.date_range(start, end, freq=freq)

    # 3) Align each model’s RMSE onto this common grid and
    #    assemble into a 2D DataFrame (rows=models, cols=dates)
    df = pd.DataFrame({
        name: series.reindex(full_dates)
        for name, series in rmse_dict.items()
    }).T

    # 4) Prepare the data array and colormap,
    #    masking missing values so they render as white
    data = np.ma.masked_invalid(df.values)
    cmap = plt.get_cmap('viridis', 256)
    cmap.set_bad(color='white')

    # 5) Render the heatmap with a true date-based x-axis
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.5 + 1))
    x_nums = mdates.date2num(full_dates.to_pydatetime())
    extent = [x_nums[0], x_nums[-1], 0, len(df)]
    im     = ax.imshow(data, aspect='auto', interpolation='nearest',
                       cmap=cmap, extent=extent, origin='lower')

    # label models on the y-axis
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index)

    # show only year labels on the x-axis
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(extent[:2])

    # finalize the plot
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='RMSE')
    plt.tight_layout()
    plt.show()
    
    
    #%%
def heatmap_on_ax(ax, model_outputs, window=4, start='2017-01-01', end='2024-12-31',
                  title='', vmin=None, vmax=None, cmap=None):
    """
    Draws a colored grid showing how prediction errors evolve over time for multiple models.
    Each row is one model; each column is a time period, colored by rolling RMSE.

    Parameters:
      ax            - Matplotlib axis to draw on.
      model_outputs - Dict mapping model names to (actuals, predictions, dates).
      window        - Number of periods for computing moving RMSE.
      start, end    - Date range of the heatmap.
      title         - Text to show above the plot.
      vmin, vmax    - Min and max colorscale limits.
      cmap          - Color map to use; defaults to 'viridis'.

    Returns:
      The image object created, so a colorbar can be added if needed.
    """

    # 1) For each model, calculate a time series of rolling RMSE values.
    rmse_dict = {}
    for name, (acts, preds, dates) in model_outputs.items():
        # Turn lists of true and predicted values into arrays
        a = np.array(acts)
        p = np.array(preds)

        # Make sure both series have the same length to compare
        n = min(len(a), len(p))
        if n == 0:
            # Skip models with no data
            continue
        a = a[-n:]
        p = p[-n:]
        # Convert the last n dates into a datetime index
        idx = pd.to_datetime(dates[-n:])

        # Compute errors, square them, average over the window, then sqrt -> RMSE
        errs = a - p
        rmse = (
            pd.Series(errs**2, index=idx)
              .rolling(window=window, min_periods=1)
              .mean()**0.5
        )
        # Keep the result for this model
        rmse_dict[name] = rmse

    # 2) Stop if no model had data
    if not rmse_dict:
        return None
    # Pick one series to infer how often to place columns (e.g., daily, monthly)
    sample = next(iter(rmse_dict.values()))
    freq = pd.infer_freq(sample.index) or 'M'
    # Build a full calendar index from start to end at that frequency
    full_dates = pd.date_range(start, end, freq=freq)
    # Re-align every model's RMSE series onto the full grid, so they all share the same columns
    df = pd.DataFrame({m: s.reindex(full_dates) for m, s in rmse_dict.items()}).T

    # 3) Turn missing values into masked entries and choose a color map
    data = np.ma.masked_invalid(df.values)
    if cmap is None:
        cmap = plt.get_cmap('viridis', 256)
    # Show gaps as white
    cmap.set_bad(color='white')

    # Convert dates to numbers so imshow can position them correctly
    mvals = mdates.date2num(full_dates.to_pydatetime())
    extent = [mvals[0], mvals[-1], 0, len(df)]
    # Draw the 2D colored grid: models on y-axis, time on x-axis
    im = ax.imshow(
        data, aspect='auto', interpolation='nearest',
        cmap=cmap, extent=extent, origin='upper',
        vmin=vmin, vmax=vmax
    )

    # 4) Label the axes: model names on left, years at bottom, and add title
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(extent[0], extent[1])
    return im


def compute_rmse_range(*datasets, window=4):
    """
    Finds the smallest and largest rolling RMSE you get across several model outputs.
    Useful for setting a common color scale when plotting multiple heatmaps.

    Parameters:
      *datasets - Any number of dicts like those passed to heatmap_on_ax.
      window    - The window size over which to smooth the RMSE.

    Returns:
      A pair (min_rmse, max_rmse).
    """
    all_vals = []
    # Go through every provided dataset
    for data in datasets:
        for _, (acts, preds, _) in data.items():
            # Convert to arrays and align lengths
            a, p = np.array(acts), np.array(preds)
            n = min(len(a), len(p))
            if n == 0:
                continue
            # Calculate rolling RMSE values
            rmse = pd.Series((a[-n:] - p[-n:])**2).rolling(window, min_periods=1).mean()**0.5
            # Collect all numbers, including NaNs
            all_vals.extend(rmse.values)
    # Return the lowest and highest RMSE seen (ignoring missing data)
    return np.nanmin(all_vals), np.nanmax(all_vals)


def plot_country_rolling_rmse(country,
                              gdp_epu, gdp_figas,
                              inf_epu, inf_figas,
                              window=4, start='2017-01-01', end='2024-12-31'):
    """
    Creates a 2×2 panel of heatmaps showing how forecasting errors changed over time
    for a given country, covering both inflation and GDP predictions from two different sources.

    Parameters:
      country    - Name of the country (used for plot titles, if needed).
      gdp_epu    - Dict of EPU-based GDP model outputs: {model_name: (actuals, preds, dates)}.
      gdp_figas  - Dict of FIGAS-based GDP model outputs.
      inf_epu    - Dict of EPU-based inflation model outputs.
      inf_figas  - Dict of FIGAS-based inflation model outputs.
      window     - Rolling window size for smoothing RMSE (default 4 periods).
      start, end - Date range for the x-axis of all heatmaps.

    This function arranges four heatmaps:
      • Top row: RMSE heatmaps for EPU-based models (left: inflation, right: GDP).
      • Bottom row: RMSE heatmaps for FIGAS-based models (left: inflation, right: GDP).
    Each colored grid cell shows the rolling RMSE at that time for one model.
    """

    # 1) Find the min and max RMSE values across each pair of datasets
    #    These ranges let us use the same color scale for inflation vs. GDP plots.
    vmin_inf, vmax_inf = compute_rmse_range(inf_epu, inf_figas, window=window)
    vmin_gdp, vmax_gdp = compute_rmse_range(gdp_epu, gdp_figas, window=window)

    # 2) Build a custom blue-to-red color map for showing error severity
    custom_cmap = LinearSegmentedColormap.from_list(
        'custom_heat', ['darkblue', 'blue', 'cyan', 'yellow', 'orange', 'red', 'darkred']
    )

    # 3) Create a 2×2 grid of subplots, sharing the x-axis formatting
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex='col')

    # 4) Fill each subplot with a heatmap of rolling RMSE
    #    Left column: inflation models; Right column: GDP models.
    im_inf1 = heatmap_on_ax(
        axes[0, 0], inf_epu,    window, start, end,
        title="EPU Inflation", vmin=vmin_inf, vmax=vmax_inf,
        cmap=custom_cmap
    )
    im_gdp1 = heatmap_on_ax(
        axes[0, 1], gdp_epu,    window, start, end,
        title="EPU GDP", vmin=vmin_gdp, vmax=vmax_gdp,
        cmap=custom_cmap
    )
    im_inf2 = heatmap_on_ax(
        axes[1, 0], inf_figas, window, start, end,
        title="FIGAS Inflation", vmin=vmin_inf, vmax=vmax_inf,
        cmap=custom_cmap
    )
    im_gdp2 = heatmap_on_ax(
        axes[1, 1], gdp_figas, window, start, end,
        title="FIGAS GDP", vmin=vmin_gdp, vmax=vmax_gdp,
        cmap=custom_cmap
    )

    # 5) Center the model names on the y-axis of each heatmap
    #    Reverse labels so the first row appears at the top
    heatmap_axes = [
        (axes[0, 0], inf_epu), (axes[0, 1], gdp_epu),
        (axes[1, 0], inf_figas), (axes[1, 1], gdp_figas),
    ]
    for ax, data_dict in heatmap_axes:
        labels = list(data_dict.keys())[::-1]
        n_rows = len(labels)
        # Position labels at the middle of each row and keep them horizontal
        ax.set_yticks(np.arange(n_rows) + 0.5)
        ax.set_yticklabels(labels, va='center', rotation=0)
        
    # 6) Format the x-axis on all plots: show years, rotate labels diagonally
    for ax in axes.flatten():
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', labelrotation=45, labelsize=8)
        ax.set_xlabel("Year")

    # 7) Add two separate colorbars for inflation (left) and GDP (right)
    cax_inf = fig.add_axes([0.45, 0.15, 0.02, 0.7])  # placement [left, bottom, width, height]
    cb_inf = fig.colorbar(im_inf1, cax=cax_inf)
    cb_inf.set_label("RMSE (Inflation)", rotation=270, labelpad=15)

    cax_gdp = fig.add_axes([0.98, 0.15, 0.02, 0.7])
    cb_gdp = fig.colorbar(im_gdp1, cax=cax_gdp)
    cb_gdp.set_label("RMSE (GDP)", rotation=270, labelpad=15)

    # 8) Adjust overall layout to make space for titles and labels
    fig.subplots_adjust(
        left=0.05, right=0.97, wspace=0.4,
        top=0.96, bottom=0.03
    )

    # Finally, display everything on screen
    plt.show()
    
    
#%%

def plot_forecasts_grouped_by_model(
    country,
    gdp_pre_dicts,
    inf_pre_dicts,
    gdp_post_dicts,
    inf_post_dicts,
    ashwin_dict_gdp,
    ashwin_dict_inf,
    ashwin_sentiments=["vader", "stability", "econlex", "loughran"]
):
    """
    Plots time-series of actual vs. predicted values for each forecasting model,
    grouped by data source and period (pre- vs. post-COVID) for a given country.

    For each combination of period (Pre-COVID/Post-COVID) and variable (GDP/Inflation),
    this creates one figure per model, stacking lines from different data sources:
      • EPU-based forecasts
      • FIGAS-based forecasts
      • Various sentiment-based forecasts (e.g., VADER, Stability, EconLex, Loughran)

    Parameters:
      country         - The country name to filter data from the provided dicts.
      gdp_pre_dicts   - Dicts of GDP forecasts before COVID, keyed by 'epu' and 'figas'.
      inf_pre_dicts   - Dicts of inflation forecasts before COVID.
      gdp_post_dicts  - Dicts of GDP forecasts after COVID.
      inf_post_dicts  - Dicts of inflation forecasts after COVID.
      ashwin_dict_gdp - Nested dict of sentiment-based GDP forecasts: country → sentiment → model data.
      ashwin_dict_inf - Same as above for inflation.
      ashwin_sentiments - List of sentiment source names to include (defaults provided).

    Each model’s plot shows:
      - Black line: actual historical values
      - Blue line: model’s predicted values
      - Year ticks on the x-axis
      - Grid and legend for clarity
    """

    # Inner helper: make one plot for a single model across all data sources
    def plot_model_group(title_prefix, data_sources, var_type):
        # Create a vertical stack of subplots: one row per data source
        fig, axes = plt.subplots(
            len(data_sources), 1,
            figsize=(12, len(data_sources) * 2.5),
            sharex=True
        )
        # Ensure axes is iterable when only one subplot
        if len(data_sources) == 1:
            axes = [axes]

        # Loop through each data source (e.g. 'EPU', 'FIGAS', 'VADER', etc.)
        for ax, (src, model_data) in zip(axes, data_sources.items()):
            # Skip if this model name is not present in the source’s data
            if model_name not in model_data:
                continue
            acts, preds, dates = model_data[model_name]
            # Skip if missing any series
            if len(acts) == 0 or len(preds) == 0 or len(dates) == 0:
                continue
            # Align the dates to the length of the actual series
            acts = np.array(acts)
            preds = np.array(preds)
            dates = pd.to_datetime(dates[-len(acts):])

            # Draw the actual values in black and predictions in blue
            ax.plot(dates, acts, color='black', linewidth=1.5, label='Actual')
            ax.plot(
                dates, preds,
                linestyle='-', color='tab:blue',
                label='Prediction'
            )
            # Title each subplot by data source and model name
            ax.set_title(f"{src} – {model_name}", fontsize=10)
            ax.grid(True)
            ax.legend(fontsize=7)

        # Label the shared x-axis on the bottom plot
        axes[-1].set_xlabel("Date")
        # Format years and rotate labels for readability
        for ax in axes:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.tick_params(axis='x', rotation=45)

        # Add an overall title and tighten layout
        fig.suptitle(
            f"{country} – {title_prefix} {var_type} – {model_name}",
            fontsize=14
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    # Build a list of all sources to include: EPU, FIGAS, plus each sentiment in uppercase
    sources = ['EPU', 'FIGAS'] + [s.upper() for s in ashwin_sentiments]

    # 1) Pre-COVID GDP: gather data from each source into one dict
    gdp_pre_data = {}
    for src in sources:
        if src == 'EPU':
            gdp_pre_data[src] = gdp_pre_dicts['epu'].get(country, {})
        elif src == 'FIGAS':
            gdp_pre_data[src] = gdp_pre_dicts['figas'].get(country, {})
        else:
            # sentiment sources live under ashwin_dict_gdp[country][sentiment]
            gdp_pre_data[src] = (
                ashwin_dict_gdp
                .get(country, {})
                .get(src.lower(), {})
            )
    # Find every model name across all sources
    all_models = sorted(set().union(*(d.keys() for d in gdp_pre_data.values())))
    # Make one combined plot per model
    for model_name in all_models:
        plot_model_group("Pre-COVID", gdp_pre_data, "GDP")

    # 2) Pre-COVID Inflation: same process but using inflation dicts
    inf_pre_data = {}
    for src in sources:
        if src == 'EPU':
            inf_pre_data[src] = inf_pre_dicts['epu'].get(country, {})
        elif src == 'FIGAS':
            inf_pre_data[src] = inf_pre_dicts['figas'].get(country, {})
        else:
            inf_pre_data[src] = (
                ashwin_dict_inf
                .get(country, {})
                .get(src.lower(), {})
            )
    all_models = sorted(set().union(*(d.keys() for d in inf_pre_data.values())))
    for model_name in all_models:
        plot_model_group("Pre-COVID", inf_pre_data, "Inflation")

    # 3) Post-COVID GDP: only EPU and FIGAS available here
    gdp_post_data = {
        'EPU': gdp_post_dicts['epu'].get(country, {}),
        'FIGAS': gdp_post_dicts['figas'].get(country, {})
    }
    all_models = sorted(set().union(*(d.keys() for d in gdp_post_data.values())))
    for model_name in all_models:
        plot_model_group("Post-COVID", gdp_post_data, "GDP")

    # 4) Post-COVID Inflation: same as GDP but for inflation
    inf_post_data = {
        'EPU': inf_post_dicts['epu'].get(country, {}),
        'FIGAS': inf_post_dicts['figas'].get(country, {})
    }
    all_models = sorted(set().union(*(d.keys() for d in inf_post_data.values())))
    for model_name in all_models:
        plot_model_group("Post-COVID", inf_post_data, "Inflation")