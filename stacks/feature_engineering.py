import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
import stacks.config as config
import warnings
warnings.filterwarnings("ignore")


def fe(df_merge, df_sales_grouped, stream, key, test):
    test=int(test)
    print("Running for {b_stream} and partition {part}".format(
        b_stream=stream, part=key))
    stl = STL(df_sales_grouped[df_sales_grouped["Key"]
              == key][["Net_Sales"]], period=12)
    res = stl.fit()
    # res.plot()
    # Oil_Prices
    df_fe = df_merge
    # Negative Treatment
    df_fe["Net_Sales"] = np.where(
        df_fe["Net_Sales"] < 0, 0, df_fe["Net_Sales"])
    # 1.Introduce lags
    df_fe["Lag-12"] = df_fe["Net_Sales"].shift(24).bfill()
    # df_fe["Lag-12"]=np.where(df_fe["Month_Adjusted"]>pd.to_datetime("1-01-2021"),df_fe["Lag-12"],df_fe["Lag-12"])
    # 2. Introduce Seasonality and Trend
    seas_trend = pd.DataFrame()
    seas_trend["Seasonality"] = res.seasonal
    seas_trend["Trend"] = res.trend
    seas_trend.reset_index(drop=True, inplace=True)
    df_fe["Seasonality"] = seas_trend["Seasonality"]
    df_fe["Trend"] = seas_trend["Trend"]
    # 3. Introduce LY sales
    #df_fe["Rolling_mean"] = df_fe["Net_Sales"].rolling(
    #    24).mean().shift(12).bfill()
    # df_fe["Rolling_mean"]=df_fe["Rolling_mean"]*1.5
    # Month Number and Year No
    # df_fe=df_fe.fillna(method="ffill")
    # df_fe=df_fe.fillna(method="bfill")
    df_fe["Month_no"] = df_fe["Month_Adjusted"].apply(lambda x: x.month)
    df_fe["Year"] = df_fe["Month_Adjusted"].apply(lambda x: x.year)
    # Data Point Number
    df_fe["Number"] = df_fe.index
    # Covid Treatment Variable
    df_fe["Covid"] = 0
    df_fe["Covid"] = np.where((df_fe["Month_Adjusted"] > pd.to_datetime("2020-03-01")) & (
        df_fe["Month_Adjusted"] < pd.to_datetime("2020-06-01")),1, df_fe["Covid"])
    df_fe["Covid"] = np.where((df_fe["Month_Adjusted"] > pd.to_datetime("2020-12-01")) & (
        df_fe["Month_Adjusted"] < pd.to_datetime("2021-06-01")),1, df_fe["Covid"])
    df_fe["Covid"] = np.where((df_fe["Month_Adjusted"] > pd.to_datetime("2021-08-01")) & (
        df_fe["Month_Adjusted"] < pd.to_datetime("2022-03-01")),0.5, df_fe["Covid"])
    df_fe["Covid"].plot()

    # Adding Govt variables to the list
    # Adding Govt variables to the list
    # Create  various lags
    # df_fe["Nat_Gas_Prices_3"]=df_fe["Nat_Gas_Prices"].shift(3)
    # df_fe["Nat_Gas_Prices_6"]=df_fe["Nat_Gas_Prices"].shift(6)
    df_fe["Net_Migration_3"] = df_fe["Net_Migration"].shift(3)
    df_fe["Net_Migration_6"] = df_fe["Net_Migration"].shift(6)
    df_fe["WCS_3"] = df_fe["WCS"].shift(3)
    df_fe["WCS_6"] = df_fe["WCS"].shift(6)
    # df_fe["WTI_3"]=df_fe["WTI"].shift(3)
    # df_fe["WTI_6"]=df_fe["WTI"].shift(6)
    #df_fe["CPI_3"] = df_fe["CPI"].shift(3)
    #df_fe["CPI_6"] = df_fe["CPI"].shift(6)
    df_fe["Unemployment_Rate_3"] = df_fe["Unemployment_Rate"].shift(3)
    df_fe["Unemployment_Rate_6"] = df_fe["Unemployment_Rate"].shift(6)
    # Treating Nan Values
    df_fe = df_fe.fillna(method="ffill")
    df_fe = df_fe.fillna(method="bfill")
    # Train_Test Split

    Data_size = int(df_merge.shape[0])
    train_cutoff = Data_size-int(test)

    x_train = df_fe.iloc[:int(train_cutoff),].drop(
        ["Net_Sales", "Month_Adjusted"], axis=1)
    x_test = df_fe.iloc[train_cutoff:,].drop(
        ["Net_Sales", "Month_Adjusted"], axis=1)
    y_train = df_fe.iloc[:train_cutoff,]["Net_Sales"]
    y_test = df_fe.iloc[train_cutoff:,]["Net_Sales"]
    # train_dates
    train_dates = df_fe.iloc[:train_cutoff,]["Month_Adjusted"]
    # Seasonality
    seasonality_df = df_fe[["Month_no", "Year", "Number", "Seasonality"]]
    seasonality_df_filtered = seasonality_df.iloc[:train_cutoff,]
    max_train_month = seasonality_df_filtered["Month_no"].max()
    max_train_year = seasonality_df_filtered["Year"].max()
    max_number = seasonality_df_filtered["Number"].max()
    seasonality_df_filtered.tail()

    # Taking seasonality as the Mean of last two years
    seas_test = pd.DataFrame()
    seas_test["seas"] = np.nan*test
    seas_test["actual_seas"] = seasonality_df.iloc[train_cutoff:,
                                                   ]["Seasonality"].reset_index(drop=True)

    for i in range(0, test):
        seas_test.loc[i, "seas"] = (
            seasonality_df_filtered.iloc[-24+i, 3]+seasonality_df_filtered.iloc[-12+i, 3])/2

    # Fixing the trend variable
    trend_df = df_fe[["Number", "Trend"]]
    trend_df_train_x = trend_df.iloc[:train_cutoff,]["Number"]
    trend_df_train_y = trend_df.iloc[:train_cutoff,][["Trend"]]
    trend_df_test_x = trend_df.iloc[train_cutoff:,]["Number"]
    trend_df_test_y = trend_df.iloc[train_cutoff:,][["Trend"]]
    # holt winters
    # single exponential smoothing
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    # double and triple exponential smoothing
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    # Set the value of Alpha and define m (Time Period)
    m = 48
    alpha = 1/(2*m)
    SES = SimpleExpSmoothing(trend_df_train_y).fit(
        smoothing_level=alpha, optimized=False, use_brute=True)
    trend_df_test_y["HWES1"] = SES.predict(
        start=max_number+1, end=df_fe.shape[0])
    # trend_df_test_y[["Trend","HWES1"]].plot(title="Holt Winters Single Exponential Smoothing")
    HWES_ADD = ExponentialSmoothing(
        trend_df_train_y["Trend"], trend="add").fit()
    trend_df_test_y["HWES2_ADD"] = HWES_ADD.predict(
        start=max_number+1, end=df_fe.shape[0])
    # HWES_MUL=ExponentialSmoothing(trend_df_train_y["Trend"],trend="mul").fit()
    # trend_df_test_y["HWES2_MUL"] = HWES_MUL.predict(start=max_number+1,end=df_fe.shape[0])
    trend_df_test_y[["Trend", "HWES1", "HWES2_ADD"]].plot(
        title=stream+" "+key+" "+"Holt Winters Double Exponential Smoothing: Additive and Multiplicative Trend")

    # Replace Trend variable in the test
    trend_type_dic = {"flat": "HWES1", "aggressive": "HWES2_ADD"}
    trend_to_used = trend_type_dic["flat"]
    x_test["Trend"] = trend_df_test_y[trend_to_used]
    x_test["Seasonality"] = seas_test["seas"].values
    seas_test.plot(title="Seasonality Plot")
    return x_train, x_test, y_train, train_dates
