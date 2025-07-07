# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 14:31:55 2025

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
from Functions import (
    construct_midas_features_pca,
    build_lagged_sentiment_matrix,
    exponential_almon_weights,
    rolling_rmse_plot,
    scatter_actual_vs_pred,
    rolling_rmse_heatmap
)

#%%

def forecast_with_sentiment_models_qd(series, sentiment_df_quarterly, sentiment_df_daily,
                                    country_code, sentiment_vars, sentiment_cols, order,
                                    window_split=(0.4, 0.4, 0.2), forecast_horizon=1, lags=90, plot=False):
    
    """
    Run a comparative rolling forecast evaluation using multiple time series and machine learning models.

    Models included:
    - ARIMA: Autoregressive Integrated Moving Average (univariate)
    - ARIMAX: ARIMA with exogenous variables (quarterly sentiment indicators)
    - U-MIDAS: Unrestricted MIDAS regression using daily sentiment lags
    - r-MIDAS: Restricted MIDAS using exponential Almon lag polynomial
    - LASSO: Penalized regression using all sentiment topics and their daily lags

    Parameters
    ----------
    series : pd.Series
        Quarterly log-differenced GDP time series.
    sentiment_df_quarterly : pd.DataFrame
        Quarterly-aggregated sentiment data with topic-country combinations as columns.
    sentiment_df_daily : pd.DataFrame
        Daily sentiment data (wide format: one column per topic-country).
    country_code : str
        ISO 2-letter country code (e.g., 'DE', 'IT').
    sentiment_vars : list of str
        Top 2 sentiment topics for ARIMAX and MIDAS models.
    sentiment_cols : list of str
        All 6 sentiment topic names for use in Lasso model.
    order : tuple
        ARIMA order (p, d, q) to use for ARIMA/ARIMAX.
    window_split : tuple of floats, optional
        (train %, adjust %, eval %) — rolling window splits, by default (0.4, 0.4, 0.2)
    forecast_horizon : int, optional
        Forecast step size, by default 1 (i.e., next quarter).
    lags : int, optional
        Number of daily sentiment lags to include in MIDAS and Lasso models, by default 90.
    plot : bool, optional
        Whether to generate comparison plot, by default False.

    Returns
    -------
    dict
        Dictionary with RMSE values from each model:
        {
            "ARIMA": float,
            "ARIMAX": float,
            "U-MIDAS": float,
            "r-MIDAS": float,
            "LASSO": float
        }
    """
    # Prepare and align GDP
    series = series.dropna()
    series.index = series.index.to_period('Q')
    gdp_temp = series.copy()
    gdp_temp.index = gdp_temp.index.to_timestamp() + QuarterEnd(0)

    p = order[0]  # number of quarterly AR lags
    H = forecast_horizon  # multi-step horizon

    y = gdp_temp

    # === ARIMA direct H-step ===
    preds_arima, acts_arima, eval_dates_arima = [], [], []
    train_end_ar = int((window_split[0] + window_split[1]) * len(y))
    for i in range(train_end_ar, len(y) - H + 1):
        fit_ar = SARIMAX(y[:i], order=order).fit(disp=False)
        fc = fit_ar.get_forecast(steps=H)
        preds_arima.append(fc.predicted_mean.iloc[-1])
        acts_arima.append(y.iloc[i + H - 1])
        eval_dates_arima.append(y.index[i + H - 1])
    rmse_arima = np.sqrt(mean_squared_error(acts_arima, preds_arima))

    # === ARIMAX direct H-step ===
    # build quarterly exogenous from sentiment_df_quarterly
    exog_q_df = sentiment_df_quarterly.set_index('date').reindex(y.index)
    exog_vars = [f"{country_code}_{var}" for var in sentiment_vars]
    exog_q_df = exog_q_df[exog_vars]
    # error handling for missing exogenous values
    if exog_q_df.isna().any().any():
        missing = exog_q_df.isna()
        missing_info = [(str(date.date()), var) for (date, var) in missing.stack()[missing.stack()].index]
        raise ValueError(f"Missing exogenous sentiment data at: {missing_info}")
    exog_q = exog_q_df.values

    preds_arimax, acts_arimax, eval_dates_arimax = [], [], []
    for i in range(train_end_ar, len(y) - H + 1):
        # fit with data up to time i-1
        fit_ax = SARIMAX(y[:i], exog=exog_q[:i], order=order).fit(disp=False)
        # freeze last known exog row and repeat for H steps
        last_exog = exog_q[i-1].reshape(1, -1)
        future_exog = np.repeat(last_exog, H, axis=0)
        fc_ax = fit_ax.get_forecast(steps=H, exog=future_exog)
        preds_arimax.append(fc_ax.predicted_mean.iloc[-1])
        acts_arimax.append(y.iloc[i + H - 1])
        eval_dates_arimax.append(y.index[i + H - 1])
    rmse_arimax = np.sqrt(mean_squared_error(acts_arimax, preds_arimax))

    # === U-MIDAS with AR lags (direct H-step) ===
    midas_col = f"{country_code}_{sentiment_vars[0]}"
    daily_series = sentiment_df_daily.set_index("date")[midas_col].dropna()
    X_midas_df = construct_midas_features_pca(
        sentiment_df_daily, country_code, sentiment_cols,
        gdp_temp.index, lags=lags
    )
    dates_midas = X_midas_df.index

    # Build AR(p) features and align
    y_series = gdp_temp
    y_lags_df = pd.DataFrame(
        {f"y_lag_{lag}": y_series.shift(lag) for lag in range(1, p+1)},
        index=y_series.index
    ).reindex(dates_midas)
    X_mid = pd.concat([y_lags_df, X_midas_df], axis=1).dropna()
    valid_mid = X_mid.index
    X_mid_vals = X_mid.values
    y_mid_vals = y_series.reindex(valid_mid).values

    preds_mid, acts_mid, eval_dates_mid = [], [], []
    train_end_mid = int((window_split[0] + window_split[1]) * len(y_mid_vals))
    for i in range(train_end_mid, len(y_mid_vals) - H + 1):
        model_mid = LinearRegression().fit(X_mid_vals[:i], y_mid_vals[:i])
        X_te = X_mid_vals[i + H - 1].reshape(1, -1)
        preds_mid.append(model_mid.predict(X_te)[0])
        acts_mid.append(y_mid_vals[i + H - 1])
        eval_dates_mid.append(valid_mid[i + H - 1])
    rmse_midas = np.sqrt(mean_squared_error(acts_mid, preds_mid))

    # === LASSO with AR lags (direct H-step) ===
    X_lasso_df = build_lagged_sentiment_matrix(
        sentiment_df_daily, country_code, sentiment_cols,
        gdp_temp.index, lags=lags
    )
    dates_lasso = X_lasso_df.index
    y_lags_l = pd.DataFrame(
        {f"y_lag_{lag}": y_series.shift(lag) for lag in range(1, p+1)},
        index=y_series.index
    ).reindex(dates_lasso)
    X_lasso_full = pd.concat([y_lags_l, X_lasso_df], axis=1).dropna()
    valid_lasso = X_lasso_full.index
    X_lasso_vals = StandardScaler().fit_transform(X_lasso_full.values)
    y_lasso_vals = y_series.reindex(valid_lasso).values

    preds_lasso, acts_lasso, eval_dates_lasso = [], [], []
    train_end_l = int((window_split[0] + window_split[1]) * len(y_lasso_vals))
    for i in range(train_end_l, len(y_lasso_vals) - H + 1):
        model_l = LassoCV(cv=5).fit(X_lasso_vals[:i], y_lasso_vals[:i])
        X_te = X_lasso_vals[i + H - 1].reshape(1, -1)
        preds_lasso.append(model_l.predict(X_te)[0])
        acts_lasso.append(y_lasso_vals[i + H - 1])
        eval_dates_lasso.append(valid_lasso[i + H - 1])
    rmse_lasso = np.sqrt(mean_squared_error(acts_lasso, preds_lasso))
    
    # === Random Forest ===
    # 1) pick the quarterly vars you need (already in sentiment_df_quarterly)
    rf_vars = [f"{country_code}_{topic}" for topic in sentiment_cols]
    sent_q = (
        sentiment_df_quarterly
        .set_index("date")[rf_vars]
        .reindex(y.index)        # align to your GDP dates
    )

    # 2) build AR(p) lags for y, aligned to same index
    y_lags_rf = pd.DataFrame(
        {f"y_lag_{lag}": y.shift(lag) for lag in range(1, p+1)},
        index=y.index
    ).reindex(sent_q.index)

    # 3) combine sentiment + AR features, drop any NaNs
    rf_df = pd.concat([sent_q, y_lags_rf], axis=1).dropna()
    valid_rf = rf_df.index

    # 4) standardize and split X/y
    X_rf = StandardScaler().fit_transform(rf_df.values)
    y_rf = y.loc[valid_rf].values

    # 5) rolling direct‐H forecasting with RandomForest
    preds_rf, acts_rf, dates_rf = [], [], []
    train_end_rf = int((window_split[0] + window_split[1]) * len(y_rf))

    for i in range(train_end_rf, len(y_rf) - H + 1):
        model_rf = RandomForestRegressor(n_estimators=100, random_state=0)
        model_rf.fit(X_rf[:i], y_rf[:i])
        X_te = X_rf[i + H - 1].reshape(1, -1)
        preds_rf.append(model_rf.predict(X_te)[0])
        acts_rf.append(y_rf[i + H - 1])
        dates_rf.append(valid_rf[i + H - 1])

    rmse_rf = np.sqrt(mean_squared_error(acts_rf, preds_rf))


    # === MIDAS-Net (MLP) ===
    X_mlp = StandardScaler().fit_transform(X_mid_vals)
    preds_mlp, acts_mlp, dates_mlp = [], [], []
    train_end_mlp = train_end_mid
    
    for i in range(train_end_mlp, len(y_mid_vals) - H + 1):
        mlp = MLPRegressor(
            hidden_layer_sizes=(5,2),
            alpha=0.1,
            early_stopping=True,
            n_iter_no_change=20,
            max_iter=500,
            random_state=0
        )
        mlp.fit(X_mlp[:i], y_mid_vals[:i])
        preds_mlp.append(mlp.predict(X_mlp[i+H-1].reshape(1,-1))[0])
        acts_mlp.append(y_mid_vals[i+H-1])
        dates_mlp.append(valid_mid[i+H-1])

    rmse_mlp = np.sqrt(mean_squared_error(acts_mlp, preds_mlp))
    
    
    # === r-MIDAS with AR lags (direct H-step) ===
    X_r_list, dates_r = [], []
    for d in gdp_temp.index:
        window = daily_series.loc[d - pd.Timedelta(days=lags): d]
        if len(window) >= lags:
            X_r_list.append(window.tail(lags).values[::-1])
            dates_r.append(d)
    X_r = np.array(X_r_list)
    valid_r = dates_r
    y_lags_r = pd.DataFrame(
        {f"y_lag_{lag}": y_series.shift(lag) for lag in range(1, p+1)},
        index=y_series.index
    ).reindex(valid_r)
    # estimate Almon weights only on training subsample
    T_r = len(valid_r)
    train_end_r = int((window_split[0] + window_split[1]) * T_r)
    def r_loss(theta):
        w = exponential_almon_weights(theta, lags)
        Xp = X_r[:train_end_r] @ w
        y_tr = y_series.reindex(valid_r)[:train_end_r].values
        return np.mean((y_tr - np.poly1d(np.polyfit(Xp, y_tr, 1))(Xp))**2)
    res = minimize(r_loss, x0=np.array([-0.1, 0.01]), method='Nelder-Mead')
    w_hat = res.x if res.success else np.array([-0.1, 0.01])
    df_proj = pd.DataFrame({'X_proj': X_r @ exponential_almon_weights(w_hat, lags)}, index=valid_r)
    df_r_full = pd.concat([df_proj, y_lags_r], axis=1).dropna()
    X_r_vals = df_r_full.values
    y_r_vals = y_series.reindex(df_r_full.index).values

    preds_r, acts_r, eval_dates_r = [], [], []
    train_end_r2 = int((window_split[0] + window_split[1]) * len(y_r_vals))
    for i in range(train_end_r2, len(y_r_vals) - H + 1):
        model_r = LinearRegression().fit(X_r_vals[:i], y_r_vals[:i])
        X_te = X_r_vals[i + H - 1].reshape(1, -1)
        preds_r.append(model_r.predict(X_te)[0])
        acts_r.append(y_r_vals[i + H - 1])
        eval_dates_r.append(df_r_full.index[i + H - 1])
    rmse_r_midas = np.sqrt(mean_squared_error(acts_r, preds_r))
    
    # --- ALIGN ALL MODELS TO THE SHORTEST SERIES ---
    models = {
        "ARIMA":     {"acts": acts_arima,   "preds": preds_arima,   "dates": eval_dates_arima},
        "ARIMAX":    {"acts": acts_arimax,  "preds": preds_arimax,  "dates": eval_dates_arimax},
        "U-MIDAS":   {"acts": acts_mid,     "preds": preds_mid,     "dates": eval_dates_mid},
        "LASSO":     {"acts": acts_lasso,   "preds": preds_lasso,   "dates": eval_dates_lasso},
        "RF":        {"acts": acts_rf,      "preds": preds_rf,      "dates": dates_rf},
        "MIDAS-Net": {"acts": acts_mlp,     "preds": preds_mlp,     "dates": dates_mlp},
        "r-MIDAS":   {"acts": acts_r,       "preds": preds_r,       "dates": eval_dates_r},
    }

    # 1) find the minimum length among all preds
    min_len = min(len(v["preds"]) for v in models.values())

    # 2) truncate each series to its last min_len points
    for v in models.values():
        v["acts"]  = np.array(v["acts"])[-min_len:]
        v["preds"] = np.array(v["preds"])[-min_len:]
        v["dates"] = v["dates"][-min_len:]

    # 3) reassign back to your variables
    acts_arima,   preds_arima,   eval_dates_arima   = models["ARIMA"].values()
    acts_arimax,  preds_arimax,  eval_dates_arimax  = models["ARIMAX"].values()
    acts_mid,     preds_mid,     eval_dates_mid     = models["U-MIDAS"].values()
    acts_lasso,   preds_lasso,   eval_dates_lasso   = models["LASSO"].values()
    acts_rf,      preds_rf,      dates_rf           = models["RF"].values()
    acts_mlp,     preds_mlp,     dates_mlp          = models["MIDAS-Net"].values()
    acts_r,       preds_r,       eval_dates_r       = models["r-MIDAS"].values()



    # Collect all model predictions aligned on ARIMA evaluation dates
    all_preds = {
        "ARIMA":      np.array(preds_arima),
        "ARIMAX":     np.array(preds_arimax),
        "U-MIDAS":    np.array(preds_mid),
        "LASSO":      np.array(preds_lasso),
        "RF":         np.array(preds_rf),
    #    "MIDAS-Net":  np.array(preds_mlp),
        "r-MIDAS":    np.array(preds_r)
    }
    acts = np.array(acts_arima)  # use ARIMA acts as reference

    # Define three unweighted combinations
    unw_combos = {
        "Comb1_ARIMA_ARIMAX_RF":    ["ARIMA", "ARIMAX", "RF"],
        "Comb2_UMIDAS_LASSO_ARIMA":    ["U-MIDAS", "LASSO", "ARIMA"],
        "Comb3_LASSO_RMIDAS_RF":   ["LASSO", "r-MIDAS", "RF"]
    }

    comb_rmse = {}
    comb_preds = {}
    for name, members in unw_combos.items():
    # 1. find minimum length among members
        lengths = [len(all_preds[m]) for m in members]
        n_min   = min(lengths)
        
        # 2. truncate each model’s preds to last n_min points
        aligned_preds = [np.array(all_preds[m])[-n_min:] for m in members]
        stack         = np.vstack(aligned_preds)       # now all same shape
        
        # 3. align actuals as well
        acts_arr       = np.array(acts)[-n_min:]
        
        # 4. average & compute RMSE
        avg_pred       = stack.mean(axis=0)
        comb_preds[name] = avg_pred
        comb_rmse[name]  = np.sqrt(mean_squared_error(acts_arr, avg_pred))

    # Define inverse-RMSE weighted combinations
    ind_rmse = {
        "ARIMA": rmse_arima,
        "ARIMAX": rmse_arimax,
        "U-MIDAS": rmse_midas,
        "LASSO": rmse_lasso,
        "RF": rmse_rf,
      #  "MIDAS-Net": rmse_mlp,
        "r-MIDAS": rmse_r_midas
    }
    wtd_rmse = {}
    wtd_preds = {}
    for name, members in unw_combos.items():
    # 1. same alignment
        lengths = [len(all_preds[m]) for m in members]
        n_min   = min(lengths)
        aligned = [np.array(all_preds[m])[-n_min:] for m in members]
        stack   = np.vstack(aligned)

        # 2. weights
        inv      = np.array([1.0/ind_rmse[m] for m in members])
        weights  = inv / inv.sum()

        # 3. weighted avg & align acts
        wpred         = (weights[:,None] * stack).sum(axis=0)
        acts_arr      = np.array(acts)[-n_min:]
        wtd_preds[name] = wpred
        wtd_rmse[name]  = np.sqrt(mean_squared_error(acts_arr, wpred))
    
    # Rolling RMSE plot
    model_outputs = {
        f"ARIMA H={H}": (acts_arima, preds_arima, eval_dates_arima),
        f"ARIMAX H={H}": (acts_arimax, preds_arimax, eval_dates_arimax),
   #     f"U-MIDAS H={H}": (acts_mid, preds_mid, eval_dates_mid),
        f"LASSO H={H}": (acts_lasso, preds_lasso, eval_dates_lasso),
        f"RF H={H}":(acts_rf,preds_rf,dates_rf),
   #     f"MIDAS-Net H={H}":(acts_mlp,preds_mlp,dates_mlp),
        f"r-MIDAS H={H}": (acts_r, preds_r, eval_dates_r)
    }
    
    # Rolling RMSE for combined models
    combo_outputs = {}
    for name in comb_preds:
        combo_outputs[name + ' (unw)'] = (acts, comb_preds[name], eval_dates_arima)
    for name in wtd_preds:
        combo_outputs[name + ' (wtd)'] = (acts, wtd_preds[name], eval_dates_arima)
        
    # Plotting
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(eval_dates_arima, acts_arima, '-', color='black', linewidth=1.5, label='Actual')
        plt.plot(eval_dates_arima, preds_arima, '--', label='ARIMA')
        plt.plot(eval_dates_arimax, preds_arimax, '-.', label='ARIMAX')
        # plt.plot(eval_dates_mid, preds_mid, ':', label=f'U-MIDAS H={H}')
        plt.plot(eval_dates_lasso, preds_lasso, ':', label=f'LASSO H={H}')
        plt.plot(dates_rf,preds_rf,'-x',label=f'RF H={H}')
       # plt.plot(dates_mlp,preds_mlp,'-o',label=f'MIDAS-Net H={H}')
        plt.plot(eval_dates_r, preds_r, '--', label=f'r-MIDAS H={H}')
        plt.title(f"Forecast Comparison H={H} - {country_code}")
        plt.xlabel("Date")
        plt.ylabel("Log-diff GDP")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Plot combined model forecasts: unweighted and weighted
        plt.figure(figsize=(10,6))
        # Actual values
        plt.plot(eval_dates_arima, acts_arima, '-', color='black', linewidth=1.5, label='Actual')
        # Unweighted
        for name, pred in comb_preds.items():
            plt.plot(eval_dates_arima, pred, linestyle='-', label=f'{name} (unw)')
        # Weighted
        for name, pred in wtd_preds.items():
            plt.plot(eval_dates_arima, pred, linestyle='--', label=f'{name} (wtd)')
        plt.title(f"Combined Forecasts vs Actuals H={H} - {country_code}")
        plt.xlabel("Date")
        plt.ylabel("Log-diff GDP")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        
        rolling_rmse_plot(model_outputs, eval_dates_arima, title=f"Rolling RMSE H={H} — {country_code}")
        
        
        rolling_rmse_plot(combo_outputs, eval_dates_arima, title=f"Rolling RMSE Combined H={H} — {country_code}")
        
        
        # 4. scatter plots
        scatter_actual_vs_pred(model_outputs, title_prefix=f'Actual vs Predicted H={H}')

       # 5. heatmap
        rolling_rmse_heatmap(model_outputs, window=4, 
                        title=f'Rolling 4-Quarter RMSE Heatmap H={H}')
        
    summary_rmse = {
        "ARIMA":    rmse_arima,
        "ARIMAX":   rmse_arimax,
        "U-MIDAS":  rmse_midas,
        "LASSO":    rmse_lasso,
        "RF":       rmse_rf,
        "MIDAS-Net":  rmse_mlp,
        "r-MIDAS":  rmse_r_midas
    }

    return {
        "summary_rmse":     summary_rmse,
        "raw_outputs":      model_outputs,
        "combo_outputs":    combo_outputs
    }

