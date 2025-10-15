import pandas as pd
import warnings
warnings.filterwarnings("ignore")
# SQL query for Liquor

input_df=pd.read_csv("../inputs/input_liquor.xlsX")
available_streams = ["liquor"]
available_models = ["prophet", "random_forest", "catboost","linear_regression","xgboost"]

run_date = pd.to_datetime(input_df.loc[:, "date"].tail(1).values[0])
hptuning=input_df.loc[:, "tuning"].tail(1).values[0]
max_training_date = pd.to_datetime(
    input_df.loc[:, "train_till"].tail(1).values[0])

# Streams to be run
streams=[]
for stream in available_streams:
    if input_df.loc[:, stream].tail(1).values[0] == 1:
        streams.append(stream)
# Models to be run
models = []
for model in available_models:
    if input_df.loc[:, model].tail(1).values[0] == 1:
        models.append(model)

# forecast_horizon
forecast_horizon = input_df.loc[:, "forecast_horizon"].tail(1).values[0]
# decription
description = input_df.loc[:, "description"].tail(1).values[0]

