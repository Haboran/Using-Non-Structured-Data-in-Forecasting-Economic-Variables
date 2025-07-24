# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 2025

@author: oskar

# -*- coding: utf-8 -*-

Function for monthly-frequency forecasting using monthly sentiment indicators.
"""
#%%
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from pandas.tseries.offsets import MonthEnd, DateOffset

#%%

def forecast_with_sentiment_models_mm(
    series: pd.Series,
    sentiment_df_monthly: pd.DataFrame,
    country_code: str,
    sentiment_vars: list,
    sentiment_cols: list,
    order: tuple,
    window_split=(0.4, 0.4, 0.2),
    forecast_horizon: int = 1,
    lags: int = 1,
    plot: bool = False
):
    """
    Run rolling direct-horizon forecasts for a monthly series using monthly sentiments.

    Models:
      - ARIMA (univariate)
      - ARIMAX (with top sentiment indicators)
      - Unrestricted distributed-lag (U-DL) on one sentiment
      - Restricted DL (r-DL) using exponential Almon weights
      - LASSO using all sentiment series
      - Random Forest using all sentiment series
      - MIDAS-Net (MLP)
      - r-MIDAS (direct H-step with AR lags)

    Returns
    -------
    dict
        {"summary_rmse": {...}, "raw_outputs": {...}, "combo_outputs": {...}}
    """
    # Prepare target
    y = series.dropna().copy()
    y.index = y.index.to_period('M').to_timestamp() + MonthEnd(0)
    y = pd.to_numeric(y, errors='coerce').dropna()
    p = order[0]
    # force at least one AR-lag so p is never zero
    p = max(p, forecast_horizon)

    H = forecast_horizon

    # Split index for rolling
    idx_split = int((window_split[0] + window_split[1]) * len(y))

    # --- ARIMA ---
    preds_arima, acts_arima, dates_arima = [], [], []
    for i in range(idx_split, len(y) - H + 1):
        fit = SARIMAX(y[:i], order=order).fit(disp=False)
        fc = fit.get_forecast(steps=H)
        preds_arima.append(fc.predicted_mean.iloc[-1])
        acts_arima.append(y.iloc[i+H-1])
        dates_arima.append(y.index[i+H-1])
    rmse_arima = np.sqrt(mean_squared_error(acts_arima, preds_arima))

    # --- ARIMAX ---
    exog = (
        sentiment_df_monthly
        .set_index('date')
        .reindex(y.index)[[f"{country_code}_{v}" for v in sentiment_vars]]
    )
    preds_arimax, dates_arimax = [], []
    for i in range(idx_split, len(y) - H + 1):
        fit_ax = SARIMAX(y[:i], exog=exog[:i], order=order).fit(disp=False)
        future_exog = np.repeat(exog.iloc[[i-1]].values, H, axis=0)
        fc_ax = fit_ax.get_forecast(steps=H, exog=future_exog)
        preds_arimax.append(fc_ax.predicted_mean.iloc[-1])
        dates_arimax.append(y.index[i+H-1])
    rmse_arimax = np.sqrt(mean_squared_error(acts_arima, preds_arimax))

    # --- U-DL (Unrestricted distributed lag) ---
    sent = sentiment_df_monthly.set_index('date')[f"{country_code}_{sentiment_vars[0]}"].reindex(y.index)
    lagged = pd.concat(
        {f"sent_lag_{lag}": sent.shift(lag) for lag in range(1, lags+1)},
        axis=1
    )
    y_lags = pd.concat(
        {f"y_lag_{lag}": y.shift(lag) for lag in range(1, p+1)},
        axis=1
    )
    X_udl = pd.concat([y_lags, lagged], axis=1).dropna()
    valid_mid = X_udl.index
    X_mid_vals = X_udl.values
    y_mid_vals = y.reindex(valid_mid).values

    preds_mid, acts_mid, dates_mid = [], [], []
    train_end_mid = int((window_split[0] + window_split[1]) * len(y_mid_vals))
    for i in range(train_end_mid, len(y_mid_vals) - H + 1):
        model_mid = LinearRegression().fit(X_mid_vals[:i], y_mid_vals[:i])
        preds_mid.append(model_mid.predict(X_mid_vals[i+H-1].reshape(1,-1))[0])
        acts_mid.append(y_mid_vals[i+H-1])
        dates_mid.append(valid_mid[i+H-1])
    rmse_midas = np.sqrt(mean_squared_error(acts_mid, preds_mid))

    # --- LASSO ---
    sent_full = sentiment_df_monthly.set_index('date')[[f"{country_code}_{c}" for c in sentiment_cols]].reindex(y.index)
    X_l = pd.concat([y_lags, sent_full.shift(0)], axis=1).dropna()
    X_lv = StandardScaler().fit_transform(X_l.values)
    y_lv = y.reindex(X_l.index).values
    preds_lasso, acts_lasso, dates_lasso = [], [], []
    for i in range(int((window_split[0]+window_split[1])*len(y_lv)), len(y_lv)-H+1):
        m = LassoCV(cv=5).fit(X_lv[:i], y_lv[:i])
        preds_lasso.append(m.predict(X_lv[i+H-1].reshape(1,-1))[0])
        acts_lasso.append(y_lv[i+H-1])
        dates_lasso.append(X_l.index[i+H-1])
    rmse_lasso = np.sqrt(mean_squared_error(acts_lasso, preds_lasso))

    # --- Random Forest ---
    X_rf = X_lv.copy()
    preds_rf, acts_rf, dates_rf = [], [], []
    for i in range(int((window_split[0]+window_split[1])*len(y_lv)), len(y_lv)-H+1):
        rf = RandomForestRegressor(n_estimators=100, random_state=0)
        rf.fit(X_rf[:i], y_lv[:i])
        preds_rf.append(rf.predict(X_rf[i+H-1].reshape(1,-1))[0])
        acts_rf.append(y_lv[i+H-1])
        dates_rf.append(X_l.index[i+H-1])
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

    # === r-MIDAS with AR lags (direct H-step), monthly only ===
    X_r_list, dates_r = [], []
    
    # 1) grab the monthly exogenous series at month-ends
    exog_monthly = (
        sentiment_df_monthly
          .set_index('date')[f"{country_code}_{sentiment_vars[0]}"]
          # reindex to your monthly target index if you like:
          .reindex(series.index)    
      )
    
    # 2) valid_mid should be the list of dates you want to forecast on
    #    e.g. the same as `dates_mid` or your y.index
    valid_mid = series.index  
    
    for d in valid_mid:
        # define the lags-window: last `lags` months up to and including d
        start = d - DateOffset(months=lags-1)
        window = exog_monthly.loc[start:d]
    
        # only keep if we actually have >= lags months of data
        if len(window) >= lags:
            # reverse order so lag0 is the most recent
            X_r_list.append(window.values[::-1])
            dates_r.append(d)
        
    # stack into array
    X_r = np.array(X_r_list)
    
    # build your AR lags of y
    y_lags_r = pd.DataFrame(
        {f"y_lag_{lag}": series.shift(lag) for lag in range(1, p+1)},
        index=series.index
    ).reindex(dates_r)
    
    # concat projected exog and y lags
    df_r_full = (
        pd.concat([pd.DataFrame(X_r @ np.ones(lags), index=dates_r, columns=['X_proj']), 
                   y_lags_r], axis=1)
        .dropna()
    )
    
    X_r_vals = df_r_full[['X_proj'] + [f"y_lag_{lag}" for lag in range(1,p+1)]].values
    y_r_vals = series.reindex(df_r_full.index).values
    
    # now your usual training/forecast loop
    preds_r, acts_r, eval_dates_r = [], [], []
    T_r = len(y_r_vals)
    train_end_r2 = int((window_split[0] + window_split[1]) * T_r)
    
    for i in range(train_end_r2, T_r - H + 1):
        model_r = LinearRegression().fit(X_r_vals[:i], y_r_vals[:i])
        preds_r.append(model_r.predict(X_r_vals[i+H-1].reshape(1,-1))[0])
        acts_r.append(y_r_vals[i+H-1])
        eval_dates_r.append(df_r_full.index[i+H-1])

    rmse_r_midas = np.sqrt(mean_squared_error(acts_r, preds_r))
    
    
    # --- ALIGN ALL MODELS TO THE SHORTEST SERIES ---
    models = {
        "ARIMA":     {"acts": acts_arima,   "preds": preds_arima,   "dates": dates_arima},
        "ARIMAX":    {"acts": acts_arima,  "preds": preds_arimax,  "dates": dates_arimax},
        "U-MIDAS":   {"acts": acts_mid,     "preds": preds_mid,     "dates": dates_mid},
        "LASSO":     {"acts": acts_lasso,   "preds": preds_lasso,   "dates": dates_lasso},
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

    
    # Collect all model predictions aligned on ARIMA dates
    all_preds = {
        "ARIMA":      np.array(preds_arima),
        "ARIMAX":     np.array(preds_arimax),
        "U-MIDAS":    np.array(preds_mid),
        "LASSO":      np.array(preds_lasso),
        "RF":         np.array(preds_rf),
        "MIDAS-Net":  np.array(preds_mlp),
        "r-MIDAS":    np.array(preds_r)
    }
    acts = np.array(acts_arima)

    # Define unweighted combos
    unw_combos = {
        "Comb1_ARIMA_ARIMAX_RF":    ["ARIMA", "ARIMAX", "RF"],
        "Comb2_UMIDAS_LASSO_ARIMA":    ["U-MIDAS", "LASSO", "ARIMA"],
        "Comb3_LASSO_RMIDAS_RF":   ["LASSO", "r-MIDAS", "RF"]
    }
    comb_rmse = {}
    comb_preds = {}
    for name, members in unw_combos.items():
        # 1) gather raw arrays
        arrs = [ np.array(all_preds[m]) for m in members ]
        # 2) compute the minimum length
        min_len = min(arr.shape[0] for arr in arrs)
        # 3) truncate each to its last min_len elements
        aligned = [ arr[-min_len:] for arr in arrs ]
        # 4) stack & average
        stack = np.vstack(aligned)
        avg   = stack.mean(axis=0)
        # 5) align your acts the same way
        aligned_acts = np.array(acts)[-min_len:]
    
        comb_preds[name] = avg
        comb_rmse[name]  = np.sqrt(mean_squared_error(aligned_acts, avg))


    # Weighted combos
    ind_rmse = {
        "ARIMA": rmse_arima,
        "ARIMAX": rmse_arimax,
        "U-MIDAS": rmse_midas,
        "LASSO": rmse_lasso,
        "RF": rmse_rf,
        "MIDAS-Net": rmse_mlp,
        "r-MIDAS": rmse_r_midas
    }
    wtd_preds = {}
    wtd_rmse  = {}
    for name, members in unw_combos.items():
        arrs    = [ np.array(all_preds[m]) for m in members ]
        min_len = min(arr.shape[0] for arr in arrs)
        aligned = [ arr[-min_len:] for arr in arrs ]
    
        inv     = np.array([1/ind_rmse[m] for m in members])
        weights = inv / inv.sum()
    
        stack = np.vstack(aligned)
        wpred = (weights[:,None] * stack).sum(axis=0)
    
        aligned_acts = np.array(acts)[-min_len:]
        wtd_preds[name] = wpred
        wtd_rmse[name]  = np.sqrt(mean_squared_error(aligned_acts, wpred))

    # Gather model outputs
    model_outputs = {
        f"ARIMA H={H}": (acts_arima, preds_arima, dates_arima),
        f"ARIMAX H={H}": (acts_arima, preds_arimax, dates_arimax),
        f"U-MIDAS H={H}": (acts_mid, preds_mid, dates_mid),
        f"LASSO H={H}": (acts_lasso, preds_lasso, dates_lasso),
        f"RF H={H}": (acts_rf, preds_rf, dates_rf),
        f"MIDAS-Net H={H}": (acts_mlp, preds_mlp, dates_mlp),
        f"r-MIDAS H={H}": (acts_r, preds_r, eval_dates_r)
    }
    combo_outputs = {}
    for name in comb_preds:
        combo_outputs[name + ' (unw)'] = (acts, comb_preds[name], dates_arima)
    for name in wtd_preds:
        combo_outputs[name + ' (wtd)'] = (acts, wtd_preds[name], dates_arima)

    return {
        "summary_rmse": {
            "ARIMA": rmse_arima,
            "ARIMAX": rmse_arimax,
            "U-MIDAS": rmse_midas,
            "LASSO": rmse_lasso,
            "RF": rmse_rf,
            "MIDAS-Net": rmse_mlp,
            "r-MIDAS": rmse_r_midas
        },
        "raw_outputs": model_outputs,
        "combo_outputs": combo_outputs
    }
