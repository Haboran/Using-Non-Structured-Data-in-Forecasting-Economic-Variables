def forecast_with_sentiment_models(
    series: pd.Series,
    sentiment_df_quarterly: pd.DataFrame,
    sentiment_df_daily: pd.DataFrame,
    country_code: str,
    sentiment_vars: list,
    sentiment_cols: list,
    order: tuple,
    window_split: tuple = (0.4, 0.4, 0.2),
    forecast_horizon: int = 1,
    freq_target: str = 'Q',          # 'Q' or 'M'
    lags: dict = None,              # exogenous-specific lags, e.g. {'daily':90,'monthly':3}
    use_pca: bool = False,
    select_top_k: int = None,
    plot: bool = False
):
    
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
    # Align target
    if freq_target.upper() == 'Q':
        y = series.resample('Q').mean()
        period_offset = QuarterEnd(0)
    elif freq_target.upper() == 'M':
        y = series.resample('M').mean()
        period_offset = MonthEnd(0)
    else:
        raise ValueError("freq_target must be 'Q' or 'M'")
    y = np.log(y).diff().dropna()
    y.index = y.index.to_period(freq_target.upper()).to_timestamp() + period_offset

    # default lags
    default_lags = {'daily':90, 'monthly':3, 'quarterly':1}
    if lags is None:
        lags = default_lags
    else:
        for k in default_lags:
            lags.setdefault(k, default_lags[k])

    p = order[0]
    H = forecast_horizon
    N = len(y)
    train_end = int((window_split[0] + window_split[1]) * N)

    # ARIMA
    preds_arima, acts_arima = [], []
    for i in range(train_end, N - H + 1):
        m = SARIMAX(y[:i], order=order).fit(disp=False)
        f = m.get_forecast(steps=H).predicted_mean.iloc[-1]
        preds_arima.append(f)
        acts_arima.append(y.iloc[i+H-1])
    rmse_arima = np.sqrt(mean_squared_error(acts_arima, preds_arima))

    # ARIMAX
    if freq_target.upper()=='Q':
        exog_full = sentiment_df_quarterly.set_index('date')
    else:
        exog_full = sentiment_df_daily.set_index('date').resample('M').mean()
    exog_full.index = exog_full.index.to_period(freq_target.upper()).to_timestamp() + period_offset
    exog_q = exog_full.reindex(y.index)
    exog_vars = [f"{country_code}_{v}" for v in sentiment_vars]
    X_exog = exog_q[exog_vars].values
    preds_ax, acts_ax = [], []
    for i in range(train_end, N - H + 1):
        m2 = SARIMAX(y[:i], exog=X_exog[:i], order=order).fit(disp=False)
        last = X_exog[i-1].reshape(1,-1)
        fut = np.repeat(last, H, axis=0)
        f2 = m2.get_forecast(steps=H, exog=fut).predicted_mean.iloc[-1]
        preds_ax.append(f2)
        acts_ax.append(y.iloc[i+H-1])
    rmse_arimax = np.sqrt(mean_squared_error(acts_ax, preds_ax))

    # U-MIDAS
    # pick first sentiment var for U-MIDAS
    var0 = sentiment_vars[0]
    col0 = f"{country_code}_{var0}"
    # get base series and aggregate
    base = sentiment_df_daily.set_index('date')[col0]
    if freq_target.upper()=='Q':
        ex = base.resample('Q').mean()
    else:
        ex = base.resample('M').mean()
    ex.index = ex.index.to_period(freq_target.upper()).to_timestamp() + period_offset
    lag_n = lags['daily'] if freq_target.upper()=='Q' else lags['monthly']
    # build MIDAS features
    X_mid = pd.concat([
        pd.concat({f"y_lag_{lag}": y.shift(lag) for lag in range(1,p+1)}, axis=1),
        pd.concat({f"exog_lag_{lag}": ex.shift(lag) for lag in range(1,lag_n+1)}, axis=1)
    ], axis=1).dropna()
    y_mid = y.reindex(X_mid.index)
    # PCA or top-k
    if use_pca:
        comps = PCA(n_components=min(5, X_mid.shape[1])).fit_transform(X_mid)
        X_mid = pd.DataFrame(comps, index=X_mid.index)
    if select_top_k:
        corrs = X_mid.corrwith(y_mid).abs().sort_values(ascending=False)
        X_mid = X_mid[corrs.index[:select_top_k]]
    # rolling forecast
    preds_mid, acts_mid = [], []
    Ym = y_mid.values
    Xm = X_mid.values
    M = len(Ym)
    train_end_m = int((window_split[0] + window_split[1]) * M)
    for i in range(train_end_m, M - H + 1):
        lr = LinearRegression().fit(Xm[:i], Ym[:i])
        f3 = lr.predict(Xm[i+H-1].reshape(1,-1))[0]
        preds_mid.append(f3)
        acts_mid.append(Ym[i+H-1])
    rmse_midas = np.sqrt(mean_squared_error(acts_mid, preds_mid))

    # LASSO
    # build full sentiment lag matrix
    dfs = []
    for col in sentiment_cols:
        name = f"{country_code}_{col}"
        s = sentiment_df_daily.set_index('date')[name]
        if freq_target.upper()=='Q':
            s2 = s.resample('Q').mean()
        else:
            s2 = s.resample('M').mean()
        s2.index = s2.index.to_period(freq_target.upper()).to_timestamp() + period_offset
        l_n = lags['daily'] if freq_target.upper()=='Q' else lags['monthly']
        dfs.append(pd.concat({f"{col}_lag{lag}": s2.shift(lag) for lag in range(1, l_n+1)}, axis=1))
    X_las = pd.concat(dfs, axis=1)
    y_las = pd.concat({f"y_lag_{lag}": y.shift(lag) for lag in range(1,p+1)}, axis=1)
    X_full = pd.concat([y_las, X_las], axis=1).dropna()
    yf = y.reindex(X_full.index)
    Xfv = StandardScaler().fit_transform(X_full.values)
    # PCA or top-k
    if use_pca:
        comps2 = PCA(n_components=min(5, Xfv.shape[1])).fit_transform(Xfv)
        Xfv = comps2
    if select_top_k:
        corrs2 = pd.DataFrame(Xfv, index=X_full.index).corrwith(yf).abs().sort_values(ascending=False)
        keep_idx = corrs2.index[:select_top_k]
        Xfv = pd.DataFrame(Xfv, index=X_full.index)[keep_idx].values
    # rolling Lasso
    preds_las, acts_las = [], []
    Yl = yf.values
    Ml = len(Yl)
    train_end_l = int((window_split[0] + window_split[1]) * Ml)
    for i in range(train_end_l, Ml - H + 1):
        las = LassoCV(cv=5).fit(Xfv[:i], Yl[:i])
        f4 = las.predict(Xfv[i+H-1].reshape(1,-1))[0]
        preds_las.append(f4)
        acts_las.append(Yl[i+H-1])
    rmse_lasso = np.sqrt(mean_squared_error(acts_las, preds_las))

    
    # === Random Forest ===
    # aggregate daily to quarterly sentiment aligned to GDP dates
    rf_vars = [f"{country_code}_{topic}" for topic in sentiment_cols]
    sent_q = sentiment_df_daily.set_index('date')[rf_vars].resample('Q').mean()
    sent_q.index = sent_q.index.to_period('Q').to_timestamp() + QuarterEnd(0)
    # align to GDP index to avoid out-of-sample quarters
    sent_q = sent_q.reindex(y.index)

    # build AR(p) lags for y aligned to GDP dates
    y_lags_rf = pd.DataFrame(
        {f"y_lag_{lag}": y.shift(lag) for lag in range(1, p+1)},
        index=y.index
    )

    # combine features and drop rows with any NaNs
    rf_df = pd.concat([sent_q, y_lags_rf], axis=1).dropna()
    valid_rf = rf_df.index
    X_rf = StandardScaler().fit_transform(rf_df.values)
    y_rf = y.loc[valid_rf].values

    # rolling direct H-step forecast
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
    # reuse X_mid_vals, y_mid_vals
    X_mlp = StandardScaler().fit_transform(X_mid_vals)
    preds_mlp, acts_mlp, dates_mlp = [], [], []
    train_end_mlp = train_end_mid
    for i in range(train_end_mlp, len(y_mid_vals) - H + 1):
        mlp = MLPRegressor(hidden_layer_sizes=(50,10), max_iter=500, random_state=0)
        mlp.fit(X_mlp[:i], y_mid_vals[:i])
        preds_mlp.append(mlp.predict(X_mlp[i+H-1].reshape(1,-1))[0])
        acts_mlp.append(y_mid_vals[i+H-1]); dates_mlp.append(valid_mid[i+H-1])
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
    
    # Collect all model predictions aligned on ARIMA evaluation dates
    all_preds = {
        "ARIMA":      np.array(preds_arima),
        "ARIMAX":     np.array(preds_arimax),
        "U-MIDAS":    np.array(preds_mid),
        "LASSO":      np.array(preds_lasso),
        "RF":         np.array(preds_rf),
        "MIDAS-Net":  np.array(preds_mlp),
        "r-MIDAS":    np.array(preds_r)
    }
    acts = np.array(acts_arima)  # use ARIMA acts as reference

    # Define three unweighted combinations
    unw_combos = {
        "Comb1_ARIMA_ARIMAX_RF":    ["ARIMA", "ARIMAX", "RF"],
        "Comb2_UMIDAS_LASSO_RF":    ["U-MIDAS", "LASSO", "ARIMA"],
        "Comb3_MLP_RMIDAS_ARIMA":   ["LASSO", "r-MIDAS", "RF"]
    }

    comb_rmse = {}
    comb_preds = {}
    for name, members in unw_combos.items():
        stack = np.vstack([all_preds[m] for m in members])
        avg_pred = stack.mean(axis=0)
        comb_preds[name] = avg_pred
        comb_rmse[name] = np.sqrt(mean_squared_error(acts, avg_pred))

    # Define inverse-RMSE weighted combinations
    ind_rmse = {
        "ARIMA": rmse_arima,
        "ARIMAX": rmse_arimax,
        "U-MIDAS": rmse_midas,
        "LASSO": rmse_lasso,
        "RF": rmse_rf,
        "MIDAS-Net": rmse_mlp,
        "r-MIDAS": rmse_r_midas
    }
    wtd_rmse = {}
    wtd_preds = {}
    for name, members in unw_combos.items():
        inv = np.array([1.0/ind_rmse[m] for m in members])
        weights = inv / inv.sum()
        stack = np.vstack([all_preds[m] for m in members])
        wpred = (weights[:, None] * stack).sum(axis=0)
        wtd_preds[name] = wpred
        wtd_rmse[name] = np.sqrt(mean_squared_error(acts, wpred))
    # Plotting
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(eval_dates_arima, acts_arima, '-', color='black', linewidth=1.5, label='Actual')
        plt.plot(eval_dates_arima, preds_arima, '--', label='ARIMA')
        plt.plot(eval_dates_arimax, preds_arimax, '-.', label='ARIMAX')
        plt.plot(eval_dates_mid, preds_mid, ':', label=f'U-MIDAS H={H}')
        plt.plot(eval_dates_lasso, preds_lasso, ':', label=f'LASSO H={H}')
        plt.plot(dates_rf,preds_rf,'-x',label=f'RF H={H}')
        plt.plot(dates_mlp,preds_mlp,'-o',label=f'MIDAS-Net H={H}')
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
        
        # Rolling RMSE plot
        model_outputs = {
            f"ARIMA H={H}": (acts_arima, preds_arima, eval_dates_arima),
            f"ARIMAX H={H}": (acts_arimax, preds_arimax, eval_dates_arimax),
            f"U-MIDAS H={H}": (acts_mid, preds_mid, eval_dates_mid),
            f"LASSO H={H}": (acts_lasso, preds_lasso, eval_dates_lasso),
            f"RF H={H}":(acts_rf,preds_rf,dates_rf),
            f"MIDAS-Net H={H}":(acts_mlp,preds_mlp,dates_mlp),
            f"r-MIDAS H={H}": (acts_r, preds_r, eval_dates_r)
        }
        rolling_rmse_plot(model_outputs, eval_dates_arima, title=f"Rolling RMSE H={H} — {country_code}")
        
        # Rolling RMSE for combined models
        combo_outputs = {}
        for name in comb_preds:
            combo_outputs[name + ' (unw)'] = (acts, comb_preds[name], eval_dates_arima)
        for name in wtd_preds:
            combo_outputs[name + ' (wtd)'] = (acts, wtd_preds[name], eval_dates_arima)
        rolling_rmse_plot(combo_outputs, eval_dates_arima, title=f"Rolling RMSE Combined H={H} — {country_code}")
        
        
        # 4. scatter plots
        scatter_actual_vs_pred(model_outputs, title_prefix=f'Actual vs Predicted H={H}')

       # 5. heatmap
        rolling_rmse_heatmap(model_outputs, window=4, 
                        title=f'Rolling 4-Quarter RMSE Heatmap H={H}')
        
    return {
        "ARIMA":rmse_arima,
        "ARIMAX":rmse_arimax,
        "U-MIDAS":rmse_midas,
        "LASSO":rmse_lasso,
        "RF":rmse_rf,
        "MIDAS-Net":rmse_mlp,
        "r-MIDAS":rmse_r_midas
    }

