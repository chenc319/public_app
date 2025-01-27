### ------------------------------------------------------------------------- PACKAGES ------------------------------------------------------------------------- ###

### PACKAGES ###
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import functools as ft
from tabulate import tabulate
import numpy as np
from datetime import datetime
import datetime as dt
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import arch
from statsmodels.regression.rolling import RollingOLS
import matplotlib.pyplot as plt
import sklearn.metrics as skmetrics
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats as st
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from scipy.stats import kurtosis
from scipy.stats import skew
import refinitiv.data as rd

### --------------------------------------------------------------------- CUSTOM FUNCTIONS --------------------------------------------------------------------- ###

### LINE GRAPH ###
def line_graph(data,
               xlabel = '',
               ylabel = '',
               title = '',
               fontsize = 6,
               x_size = 12,
               y_size = 8):
    plt.figure(figsize=(x_size,y_size))
    data = pd.DataFrame(data)
    plot_variable = plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(plot_variable, data.columns, fontsize=fontsize)
    plt.show()

### BAR GRAPH ###
def bar_graph(data,
              start_date_string,
               x_size = 20,
               y_size = 15):
    plt.rcParams["figure.figsize"] = (x_size, y_size)
    data['dates'] = data.index
    ax = data.loc[datetime.strptime(start_date_string,
                                    '%Y-%m-%d').date():].plot(x="dates",
                                                              y=data.columns[:len(data.columns) - 1],
                                                              kind="bar",
                                                              rot=0)
    plt.show()

### ACF PLOT ###
def rolling_acf(return_data,
                  num_lags):
    for x in range(0,num_lags):
        return_data

### CALCULATE BASIC METRICS ###
def return_metrics(backtest_returns_data,
                   benchmark_data):
    '''
    backtest_returns_data = roll_bt_ret
    benchmark_data = sofr_fly_benchmark
    '''
    return_metrics_df = pd.DataFrame(
        columns = ['Total Return',
                   'Avg Return',
                   'Avg Upside Return',
                   'Avg Downside Return',
                   'Win Ratio',
                   'Ann. Return',
                   'Ann. Volatility',
                   'Return/Risk',
                   'Max Return','Max Return Date',
                   'Min Return','Min Return Date',
                   'Beta']
    )
    merged_data = merge_dfs([backtest_returns_data,benchmark_data]).dropna()
    for x in range(0,len(backtest_returns_data.columns)):
        col = backtest_returns_data.columns[x]
        data = pd.DataFrame(backtest_returns_data[col]).ffill().dropna()
        data.columns = ['daily returns']
        total_return = data['daily returns'].sum()
        mean_return = data['daily returns'].mean()
        avg_win_return = data[data['daily returns'] > 0].mean().iloc[0]
        avg_lose_return = data[data['daily returns'] < 0].mean().iloc[0]
        win_ratio = len(data[data['daily returns'] > 0]) / len(data)
        ann_return = mean_return * 252
        ann_vol = data['daily returns'].std() * (252**0.5)
        return_risk = ann_return / ann_vol
        max_return = data['daily returns'].max()
        max_return_date = data[data['daily returns'] == max_return].index[0]
        min_return = data['daily returns'].min()
        min_return_date = data[data['daily returns'] == min_return].index[0]
        beta = static_beta(benchmark_data,data['daily returns'])
        return_metrics_df.loc[col] = [total_return,mean_return,
                                      avg_win_return,avg_lose_return,
                                      win_ratio,ann_return,
                                      ann_vol,return_risk,
                                      max_return,max_return_date,
                                      min_return,min_return_date,beta]
    print(tabulate(return_metrics_df, headers='keys'))
    return(return_metrics_df)

def return_metrics_t(backtest_returns_data, benchmark_data):
    '''
    backtest_returns_data = roll_bt_ret
    benchmark_data = sofr_fly_benchmark
    '''
    return_metrics_df = pd.DataFrame(
        columns=['Total Return',
                 'Avg Return',
                 'Avg Upside Return',
                 'Avg Downside Return',
                 'Win Ratio',
                 'Ann. Return',
                 'Ann. Volatility',
                 'Return/Risk',
                 'Max Return', 'Max Return Date',
                 'Min Return', 'Min Return Date',
                 'Beta']
    )

    merged_data = merge_dfs([backtest_returns_data, benchmark_data]).dropna()

    for x in range(0, len(backtest_returns_data.columns)):
        col = backtest_returns_data.columns[x]
        data = pd.DataFrame(backtest_returns_data[col]).ffill().dropna()
        data.columns = ['daily returns']

        total_return = data['daily returns'].sum()
        mean_return = data['daily returns'].mean()
        avg_win_return = data[data['daily returns'] > 0].mean().iloc[0]
        avg_lose_return = data[data['daily returns'] < 0].mean().iloc[0]
        win_ratio = len(data[data['daily returns'] > 0]) / len(data)
        ann_return = mean_return * 252
        ann_vol = data['daily returns'].std() * (252 ** 0.5)
        return_risk = ann_return / ann_vol
        max_return = data['daily returns'].max()
        max_return_date = data[data['daily returns'] == max_return].index[0]
        min_return = data['daily returns'].min()
        min_return_date = data[data['daily returns'] == min_return].index[0]
        beta = static_beta(benchmark_data, data['daily returns'])

        return_metrics_df.loc[col] = [total_return, mean_return,
                                      avg_win_return, avg_lose_return,
                                      win_ratio, ann_return,
                                      ann_vol, return_risk,
                                      max_return, max_return_date,
                                      min_return, min_return_date, beta]
    print(tabulate(return_metrics_df.T, headers='keys'))
    return return_metrics_df.T

### NEW CUSTOM MERGE DF BASED ON LAMBDA REDUCE FUNCTIONAL TOOLS ###
def merge_dfs(array_of_dfs):
    new_df = ft.reduce(lambda left, right: pd.merge(left,
                                                    right,
                                                    left_index=True,
                                                    right_index=True,
                                                    how='outer'), array_of_dfs)
    return(new_df)

### Z SCORE FUNCTION ###
def z_score(data,
            rolling_window):
    data = pd.DataFrame(data)
    data.columns = ['signal']
    data['mean'] = data['signal'].rolling(rolling_window).mean()
    data['std'] = data['signal'].rolling(rolling_window).std()
    data['z'] = (data['signal'] - data['mean']) / data['std']
    final_data = pd.DataFrame(data['z'])
    final_data.columns = ['z']
    return(final_data)

### MONTHLY RETURNS ###
def monthly_returns(df):
    '''
    df = oi_bt_ret
    '''
    df.index = pd.to_datetime(df.index)
    month_data = df.resample('1M').sum()
    month_data.index = pd.to_datetime(month_data.index, format="%Y-%m-%d")
    return(month_data)

### ANNUAL RETURNS ###
def annual_returns(df):
    '''
    df = oi_bt_ret
    '''
    df.index = pd.to_datetime(df.index)
    annual_data = df.resample('1Y').sum()
    annual_data.index = pd.to_datetime(annual_data.index, format="%Y-%m-%d")
    return(annual_data)

### VOL FACTOR RETURNS WITH TRANSACTION COST ###
def vol_factor_pnl(signal_data,
                   underlying_levels_delta_data,
                   backtest_notional,
                   target_vol_factor,
                   rolling_vol_window,
                   tcost,
                   fut_underlying_dv01,
                   tick_value,
                   com_fees):
    '''
    signal_data = -1 * final_data.shift(1)
    underlying_levels_delta_data = underlying_data[col2]
    backtest_notional = 5000000
    target_vol_factor = 3
    rolling_vol_window = 21
    tcost = -0.25
    fut_underlying_dv01 = 25
    tick_value = 12.5
    com_fees = -1.5
    '''
    bt_df = merge_dfs([signal_data,underlying_levels_delta_data])
    bt_df.columns = ['signal','levels_delta']
    bt_df['Target Ann Volatility'] = bt_df['signal'] * target_vol_factor
    bt_df['Target Daily Volatility'] = bt_df['Target Ann Volatility'] / (252 ** 0.5)
    bt_df['Daily Dollar Volatility'] = bt_df['Target Daily Volatility'] * backtest_notional
    bt_df['Rolling Vol'] = bt_df['signal'].rolling(rolling_vol_window).std()
    bt_df['Target DV01'] = bt_df['Daily Dollar Volatility'] / bt_df['Rolling Vol']
    bt_df['Recommended Lots EOD'] = bt_df['Target DV01'] / fut_underlying_dv01

    bt_df['Pre-Tcost PnL'] = (tick_value * 100 * bt_df['levels_delta'].shift(-1) * bt_df['Recommended Lots EOD']).shift(1)
    bt_df['Change in Lots'] = bt_df['Recommended Lots EOD'] - bt_df['Recommended Lots EOD'].shift(1)
    bt_df['Tcost in USD'] = fut_underlying_dv01 * abs(bt_df['Change in Lots']) * tcost
    bt_df['Commission and Fees in USD'] = abs(bt_df['Change in Lots']) * com_fees
    bt_df['pnl'] = (bt_df['Commission and Fees in USD'] + bt_df['Tcost in USD'] + bt_df['Pre-Tcost PnL']) / backtest_notional
    bt_df['tcost perc'] = (bt_df['Tcost in USD'] + bt_df['Commission and Fees in USD']) / backtest_notional
    return(pd.DataFrame(bt_df['pnl']).dropna(),
           pd.DataFrame(bt_df['tcost perc']).dropna())

### VOL FACTOR RETURNS WITH TRANSACTION COST ###
def signal_pnl(signal_data,
               underlying_levels_delta_data,
               backtest_notional,
               rolling_vol_window,
               tcost,
               fut_underlying_dv01,
               tick_value,
               com_fees):
    '''
    signal_data = -1 * final_data.shift(1)
    underlying_levels_delta_data = underlying_data[col2]
    backtest_notional = 5000000
    target_vol_factor = 3
    rolling_vol_window = 21
    tcost = -0.25
    fut_underlying_dv01 = 25
    tick_value = 12.5
    com_fees = -1.5
    '''
    bt_df = merge_dfs([signal_data,underlying_levels_delta_data])
    bt_df.columns = ['signal','levels_delta']
    bt_df['Target Ann Volatility'] = bt_df['signal']
    bt_df['Target Daily Volatility'] = bt_df['Target Ann Volatility'] / (252 ** 0.5)
    bt_df['Daily Dollar Volatility'] = bt_df['Target Daily Volatility'] * backtest_notional
    bt_df['Target DV01'] = bt_df['Daily Dollar Volatility']
    bt_df['Recommended Lots EOD'] = bt_df['Target DV01'] / fut_underlying_dv01

    bt_df['Pre-Tcost PnL'] = (tick_value * 100 * bt_df['levels_delta'].shift(-1) * bt_df['Recommended Lots EOD']).shift(1)
    bt_df['Change in Lots'] = bt_df['Recommended Lots EOD'] - bt_df['Recommended Lots EOD'].shift(1)
    bt_df['Tcost in USD'] = fut_underlying_dv01 * abs(bt_df['Change in Lots']) * tcost
    bt_df['Commission and Fees in USD'] = abs(bt_df['Change in Lots']) * com_fees
    bt_df['pnl'] = (bt_df['Commission and Fees in USD'] + bt_df['Tcost in USD'] + bt_df['Pre-Tcost PnL']) / backtest_notional
    bt_df['tcost perc'] = (bt_df['Tcost in USD'] + bt_df['Commission and Fees in USD']) / backtest_notional
    return(pd.DataFrame(bt_df['pnl']).dropna(),
           pd.DataFrame(bt_df['tcost perc']).dropna())

### DRAWDOWN ###
def drawdown(cumulative_daily_return_data):
    '''
    cumulative_daily_return_data = roll_bt_ret
    '''
    drawdown_array = []
    for col in cumulative_daily_return_data.columns:
        data = pd.DataFrame(cumulative_daily_return_data[col])
        data.columns = [col]
        data[col + ' drawdown'] = np.nan
        for row in range(0, (len(data.index) - 1)):
            if (data.iloc[row, 0] < data.iloc[(row - 1), 0]):
                data.iloc[row, 1] = (data.iloc[row, 0] - data.iloc[(row - 1), 0])
            else:
                data.iloc[row, 1] = 0
        drawdown_array.append(pd.DataFrame(data[col + ' drawdown']).dropna())
    drawdown_df = merge_dfs(drawdown_array)
    drawdown_df.columns = cumulative_daily_return_data.columns
    return(drawdown_df)

### BETA ###
def rolling_beta(return_ts, benchmark_ts, roll_window):
    returns = merge_dfs([return_ts,benchmark_ts])
    rolling_cov = returns.iloc[:,0].rolling(roll_window).cov(returns.iloc[:,1])
    rolling_var = returns.iloc[:,1].rolling(roll_window).var()
    individual_beta = rolling_cov / rolling_var
    return individual_beta
def static_beta(return_ts, benchmark_ts,):
    returns = merge_dfs([return_ts,benchmark_ts])
    rolling_cov = returns.iloc[:,0].cov(returns.iloc[:,1])
    rolling_var = returns.iloc[:,1].var()
    individual_beta = rolling_cov / rolling_var
    return individual_beta

### RMSE ###
def rolling_rmse(predicted,actual,rolling_window):
    '''
    predicted = comb_data['Roll/Vol'].shift(1)
    actual = comb_data['Roll/Vol']
    rolling_window = 63
    '''
    predicted = pd.DataFrame(predicted)
    predicted.columns = ['pred']
    actual = pd.DataFrame(actual)
    actual.columns = ['actual']
    merge = merge_dfs([predicted,actual]).dropna()
    merge['diff'] = merge['pred'] - merge['actual']
    merge['diff^2'] = merge['diff'] ** 2
    merge['rolling_mean'] = merge['diff^2'].rolling(rolling_window).mean()
    merge['rmse'] = merge['rolling_mean'] ** 0.5
    return(pd.DataFrame(merge['rmse']).dropna())

### UNITY NORMALIZATION ###
def rolling_max_min_normalize(data,
                              rolling_window,
                              lower_bound = 0,
                              upper_bound = 1):
    ### THIS USES A BRANCH OF NORMALIZATION TECHNIQUES SIMILAR TO MAX MIN FEATURE SCALING ###
    '''
    data = sofr_fly_6m_yield_change['SFRH19 | SFRM19 | SFRU19']
    rolling_window = 21
    '''
    data = pd.DataFrame(data.dropna())
    data.columns = ['returns']
    data['norm'] = np.nan
    if((lower_bound == 0) & (upper_bound == 1)):
        for x in range((rolling_window + 1), len(data)):
            subset_data = data.iloc[(x - rolling_window):x, :]
            min = subset_data['returns'].min()
            max = subset_data['returns'].max()
            data.at[data.index[x],'norm'] = (data.at[data.index[x],'returns'] - min) / (max-min)
        return(pd.DataFrame(1 - data['norm']).dropna())
    elif((lower_bound == -1) & (upper_bound == 1)):
        for x in range((rolling_window + 1), len(data)):
            subset_data = data.iloc[(x - rolling_window):x, :]
            min = subset_data['returns'].min()
            max = subset_data['returns'].max()
            data.at[data.index[x], 'norm'] = (2 * (data.at[data.index[x], 'returns'] - min) / (max - min)) - 1
        return (pd.DataFrame(1 - data['norm']).dropna())

### ASYMMETRIC VOLATILITY FUNCTION ###
def rolling_asym_vol_ratio(data,
                           rolling_window):
    '''
    data = daily_return
    rolling_window = 21
    '''
    data = pd.DataFrame(data.dropna())
    data.columns = ['returns']
    data['asym_vol_ratio'] = np.nan
    for x in range((rolling_window+1),len(data)):
        subset_data = data.iloc[(x-rolling_window):x,:]
        upside_data = subset_data[subset_data['returns'] > 0]
        downside_data = subset_data[subset_data['returns'] < 0]
        if((len(upside_data) == 0 or len(upside_data) == 1) and len(downside_data) > 0):
            downside_volatility = downside_data['returns'].std() * (252 ** 0.5)
            ratio = downside_volatility - 1
        elif((len(downside_data) == 0 or len(downside_data) == 1) and len(upside_data) > 0):
            upside_volatility = upside_data['returns'].std() * (252 ** 0.5)
            downside_volatility = upside_volatility
            ratio = upside_volatility + 1
        else:
            downside_volatility = downside_data['returns'].std() * (252 ** 0.5)
            upside_volatility = upside_data['returns'].std() * (252 ** 0.5)
            ratio = upside_volatility / downside_volatility
        data.at[data.index[x],'asym_vol_ratio'] = ratio
    return(pd.DataFrame(data['asym_vol_ratio']).dropna())

### ASYMMETRIC VOLATILITY FUNCTION ###
def rolling_asym_std(data,
                     rolling_window):
    '''
    data = daily_return
    rolling_window = 21
    '''
    data = pd.DataFrame(data.dropna())
    data.columns = ['returns']
    data['asym_std'] = np.nan
    for x in range((rolling_window+1),len(data)):
        subset_data = data.iloc[(x-rolling_window):x,:]
        upside_data = subset_data[subset_data['returns'] > 0]
        downside_data = subset_data[subset_data['returns'] < 0]
        if((len(upside_data) == 0 or len(upside_data) == 1) and len(downside_data) > 0):
            downside_volatility = downside_data['returns'].std()
        elif((len(downside_data) == 0 or len(downside_data) == 1) and len(upside_data) > 0):
            upside_volatility = upside_data['returns'].std()
            downside_volatility = upside_volatility
        else:
            downside_volatility = downside_data['returns'].std()
            upside_volatility = upside_data['returns'].std()
            ratio = upside_volatility / downside_volatility
        data.at[data.index[x],'asym_std'] = (upside_volatility * len(upside_data)) + (downside_volatility * len(downside_data)) / len(subset_data)
    return(pd.DataFrame(data['asym_std']).dropna())

### ASYMMETRIC Z SCORE FUNCTION ###
def asym_z_score(data,
                 underlying_data,
                 rolling_window):
    data = pd.DataFrame(data)
    data.columns = ['signal']
    data['mean'] = data['signal'].rolling(rolling_window).mean()
    data['std'] = rolling_asym_std(underlying_data,rolling_window) / (252**0.5)
    data['z'] = (data['signal'] - data['mean']) / data['std']
    final_data = pd.DataFrame(data['z'])
    final_data.columns = ['z']
    return(final_data)

### HURST EXPONENT TO CALCULATE MEAN REVERSION ###
def get_hurst_exponent(time_series,
                       max_lag=20):
    '''
    time_series = wc_metrics_data[0]['3-5pm Change (%)']
    max_lag = 20
    '''
    lags = range(1, max_lag)
    lag = 1
    # variances of the lagged differences
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag]).dropna()) for lag in lags]
    # calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 1)
    return reg[0]

### CALCUALTE ACF TO GENERATE PLOT ###
def acf(data_series,lag_num):
    pcc_df = pd.DataFrame(columns = ['Lags','PCC'])
    for x in range(1,lag_num+1):
        lag_data = data_series.shift(x)
        unlag_lag_data = pd.concat([data_series,lag_data],axis = 1).dropna()
        x_mean = data_series.mean()
        y_mean = lag_data.mean()
        unlag_lag_data['x dif'] = data_series - x_mean
        unlag_lag_data['y dif'] = lag_data - y_mean
        unlag_lag_data['x dif * y dif'] = unlag_lag_data['x dif'] * unlag_lag_data['y dif']
        unlag_lag_data['x dif^2'] = unlag_lag_data['x dif'] * unlag_lag_data['x dif']
        unlag_lag_data['y dif^2'] = unlag_lag_data['y dif'] * unlag_lag_data['y dif']
        pcc = unlag_lag_data['x dif * y dif'].sum() / (unlag_lag_data['x dif^2'].sum() * unlag_lag_data['y dif^2'].sum()) ** 0.5
        row_df = pd.DataFrame([x,pcc]).T
        row_df.columns = ['Lags','PCC']
        pcc_df = pd.concat([pcc_df,row_df])
    return(pcc_df)
# test_acf = acf(ts,50)
# plt.plot(test_acf['Lags'], test_acf['PCC'])
# plt.xlabel('Lags')
# plt.ylabel('Pearson Correlation Coefficient')
# plt.title('ACF Plot')
# plt.show()

### ROLLING VARIANCE CALCULATION ###
def rolling_variance(data,
                     rolling_window):
    '''
    data = us_ylds_change_df
    rolling_window = 63
    '''
    final_return_df = pd.DataFrame(columns = data.columns,
                                   index = data.index)
    for row in range(rolling_window+1,len(data)):
        subset_data = data.iloc[(row-rolling_window):row,]
        var_cov_matrix = np.cov(subset_data.T)
        var = [var_cov_matrix[x][x] for x in range(0,len(var_cov_matrix))]
        final_return_df.loc[final_return_df.index[row],:] = var
    return(final_return_df.dropna())

### ROLLING KURTOSIS ANALYSIS ###
def rolling_kurtosis(data,
                     rolling_window):
    '''
    data = usd_yields_variance_chg
    rolling_window = 63
    '''
    final_return_df = pd.DataFrame(columns = data.columns,
                                   index = data.index)
    for row in range(rolling_window+1,len(data)):
        subset_data = data.iloc[(row-rolling_window):row,]
        kurtosis_val = kurtosis(subset_data)
        final_return_df.loc[final_return_df.index[row],:] = kurtosis_val
    return(final_return_df.dropna())

### ROLLING KURTOSIS ANALYSIS ###
def rolling_skew(data,
                 rolling_window):
    '''
    data = usd_yields_variance_chg
    rolling_window = 63
    '''
    final_return_df = pd.DataFrame(columns = data.columns,
                                   index = data.index)
    for row in range(rolling_window+1,len(data)):
        subset_data = data.iloc[(row-rolling_window):row,]
        skew_val = skew(subset_data)
        final_return_df.loc[final_return_df.index[row],:] = skew_val
    return(final_return_df.dropna())

def drawdown(df, title='Drawdown Graph', ylabel='Drawdown'):
    plt.figure(figsize=(10, 6))
    for column in df.columns:
        cumulative_returns = (df[column] + 1).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        plt.plot(drawdown, label=column)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def get_refinitiv_historical_data(tickers,flds,end_date_str):
    '''
    :param tickers: list
    :return df: pandas dataframe
    '''
    df = rd.get_history(
        universe=tickers,
        fields=flds,
        interval='1d',
        start='1999-12-31',
        end=end_date_str
    )
    df.index = df.index.values
    df.index = pd.to_datetime(df.index)
    df.columns = tickers
    return(df)


