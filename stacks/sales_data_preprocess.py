import pandas as pd
import datetime as datetime
import warnings
import numpy as np
warnings.filterwarnings("ignore")
import stacks.config as config


def sales_preprocessing(df, stream):
    print("Preprocessing started for {stream_name}".format(stream_name=stream))
    if stream=="liquor":
        df["Net_Sales"]=df["Net Sales"]
        df["Date"]=df["Month_Adjusted"]

    df_sales_sorted = df.sort_values(
        by="Date", ascending=True).reset_index(drop=True)
    # Rerun from Here to Change Dates
    if stream == "vlt" or stream == "cannabis":
        df_sales_sorted["Key"] = stream
    elif stream == "slots":
        
        df_sales_sorted=df_sales_sorted[df_sales_sorted["Ownership"]!="HYBRID"]
        df_sales_sorted["Ownership"] = np.where(df_sales_sorted["Ownership"] == "OWNED", "owned", df_sales_sorted["Ownership"])
        df_sales_sorted["Ownership"] = np.where(df_sales_sorted["Ownership"] == "LEASED", "leased", df_sales_sorted["Ownership"])

        df_sales_sorted["Key"] = df_sales_sorted["CasinoType"] + \
            "_"+df_sales_sorted["Ownership"]
    elif stream == "online":
        df_sales_sorted["Key"] = df_sales_sorted["GameVertical"]
    elif stream=="liquor":
        df_sales_sorted["Key"] = stream +"_"+df_sales_sorted["Key"]
        print(df_sales_sorted)
        #df_sales_sorted["Net_Sales"]=df_sales_sorted["Net_Sales"].astype("float")

    
        

    df_sales_grouped = df_sales_sorted.groupby(by=["Date", "Key"])["Net_Sales"].sum().reset_index()
    
    df_sales_grouped["Date"] = pd.to_datetime(df_sales_grouped["Date"])
    df_sales_grouped["Month_Adjusted"] = df_sales_grouped["Date"].apply(
        lambda x: datetime.datetime(x.year, x.month, 1))
    df_sales_grouped["Month_Adjusted"] = pd.to_datetime(
        df_sales_grouped["Month_Adjusted"])
    df_sales_grouped.drop("Date", inplace=True, axis=1)
    df_sales_grouped.set_index("Month_Adjusted", inplace=True, drop=True)
    df_sales_grouped = df_sales_grouped.groupby(
        by=["Month_Adjusted", "Key"]).sum().reset_index()
    # Remove latest month
    df_sales_grouped = df_sales_grouped.iloc[:-1, :]
    # Plot Net_Sales
    df_sales_grouped.plot()
    # Date filter
    df_sales_grouped = df_sales_grouped[(
        df_sales_grouped["Month_Adjusted"] >= pd.to_datetime("2006-01-01"))]
    # Set dates as index
    df_sales_grouped.set_index("Month_Adjusted", inplace=True, drop=True)
    # Generating valid keys
    valid_keys = []
    for key in list(df_sales_grouped["Key"].unique()):
        if df_sales_grouped[df_sales_grouped["Key"] == key][["Net_Sales"]].index[-1].year > 2023:
            if df_sales_grouped[df_sales_grouped["Key"] == key][["Net_Sales"]].shape[0]>config.forecast_horizon:

                print(df_sales_grouped[df_sales_grouped["Key"]
                                   == key][["Net_Sales"]].shape)
                valid_keys.append(key)
   # if stream=="cannabis":
        #df_sales_grouped=pd.read_csv("processed_data/monthly_data_cannabis.csv")
    #    df_sales_grouped["Month_Adjusted"]=pd.to_datetime(df_sales_grouped["Month_Adjusted"])
     #   df_sales_grouped.set_index("Month_Adjusted", inplace=True, drop=True)
    # To save to Data for sales
    df_sales_grouped.to_csv(
        "processed_data/monthly_data_{b_stream}.csv".format(b_stream=stream))
    print("Preprocessing Done")

    return df_sales_grouped, valid_keys
