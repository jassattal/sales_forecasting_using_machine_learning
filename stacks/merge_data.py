import pandas as pd
import stacks.config as config
import stacks.features as features
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings("ignore")


def merge(df_sales_grouped, max_training_date, forecast_horizon):
    df_sales_f = df_sales_grouped.reset_index()
    # Remove latest month
    df_sales_f = df_sales_f.iloc[:-1, :]
    # df_online_grouped_f=df_online_grouped.reset_index()
    training_dates = pd.date_range(
        start=df_sales_f["Month_Adjusted"].min(), end=config.max_training_date, freq="MS")
    test_dates = pd.date_range(start=df_sales_f["Month_Adjusted"].min(
    ), end=max_training_date+relativedelta(months=forecast_horizon), freq="MS")
    dates = pd.DataFrame(test_dates)
    dates.columns = ["Month_Adjusted"]
    # Merge with dates
    df_merge = dates.merge(df_sales_f, how="left", on="Month_Adjusted")
    # Merge files with Vlt Sales
    df_merge = df_merge.merge(features.unemployment,
                              how="left", on="Month_Adjusted")
    # df_merge=df_merge.merge(retail_grouped,how="left",on="Month_Adjusted")
    df_merge = df_merge.merge(
        features.Migration, how="left", on="Month_Adjusted")
    df_merge = df_merge.merge(features.Oil_pivot_2,
                              how="left", on="Month_Adjusted")
    # df_merge=df_merge.merge(df_online_grouped_f,how="left",on="Month_Adjusted")
    df_merge = df_merge.merge(features.cpi, how="left", on="Month_Adjusted")
    # df_merge=df_merge.merge(Nat_Gas_Price,how="left",on="Month_Adjusted")
    df_merge = df_merge.merge(features.gdp, how="left", on="Month_Adjusted")
    df_merge = df_merge.merge(
        features.population, how="left", on="Month_Adjusted")
    return df_merge, test_dates
