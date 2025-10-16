from sklearn.metrics import mean_absolute_error
from quantile_forest import RandomForestQuantileRegressor
import catboost as cb
from xgboost import XGBRegressor  

from prophet import Prophet
import warnings
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import numpy as np

from prophet import Prophet
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

def univariate(y_train, model, forecast_horizon, train_dates, n_splits=5):
    if model != "prophet":
        raise ValueError("Only 'prophet' model is supported in univariate.")

    df = pd.DataFrame({"ds": train_dates, "y": y_train})
    total_points = len(df)
    split_size = int((total_points - forecast_horizon) // n_splits)

    rmse_yhat = []
    rmse_lower = []
    rmse_upper = []

    for i in range(n_splits):
        train_end = (i + 1) * split_size
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:train_end + forecast_horizon]

        if len(val_df) < forecast_horizon:
            break

        model_prophet = Prophet(
            growth="linear",
            changepoint_prior_scale=0.5,
            seasonality_prior_scale=1,
            changepoint_range=0.95,
            seasonality_mode='multiplicative'
        )
        model_prophet.fit(train_df)

        future = model_prophet.make_future_dataframe(periods=forecast_horizon, freq='MS')
        forecast = model_prophet.predict(future)
        forecast = forecast.set_index("ds").loc[val_df["ds"]]

        rmse_yhat.append(np.sqrt(mean_squared_error(val_df["y"], forecast["yhat"])))
        rmse_lower.append(np.sqrt(mean_squared_error(val_df["y"], forecast["yhat_lower"])))
        rmse_upper.append(np.sqrt(mean_squared_error(val_df["y"], forecast["yhat_upper"])))

    return {
        "prophet_mean": np.mean(rmse_yhat),
        "prophet_lower": np.mean(rmse_lower),
        "prophet_upper": np.mean(rmse_upper)
    }



def multivariate(x_train, y_train, sel_feature, model, forecast_horizon, n_splits,hyper_params):

    params=hyper_params
    sel_feature = list(set(sel_feature))
    x_train_final = x_train[sel_feature]

    total_points = len(x_train_final)
    min_train_size = 12
    split_size = (total_points - forecast_horizon - min_train_size) // n_splits
    if split_size < 1:
        raise ValueError("Not enough data to perform cross-validation with the given settings.")

    rmse_scores = []
    rmse_lower = []
    rmse_upper = []

    for i in range(n_splits):
        train_end = min_train_size + i * split_size
        val_start = train_end
        val_end = val_start + forecast_horizon

        if val_end > total_points:
            break

        X_tr = x_train_final.iloc[:train_end]
        y_tr = y_train.iloc[:train_end]
        X_val = x_train_final.iloc[val_start:val_end]
        y_val = y_train.iloc[val_start:val_end]

        if model == "random_forest":
            reg = RandomForestQuantileRegressor(**params)
            reg.fit(X_tr, y_tr)
            y_pred = reg.predict(X_val, quantiles=[0.50])
            y_pred_5 = reg.predict(X_val, quantiles=[0.05])
            y_pred_95 = reg.predict(X_val, quantiles=[0.95])

            rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
            rmse_lower.append(np.sqrt(mean_squared_error(y_val, y_pred_5)))
            rmse_upper.append(np.sqrt(mean_squared_error(y_val, y_pred_95)))

        elif model == "catboost":
            reg = cb.CatBoostRegressor(**params)
            reg.fit(X_tr, y_tr)
            y_pred = reg.predict(X_val)
            rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))

        elif model == "linear_regression":
            reg = sm.OLS(y_tr, X_tr).fit()
            y_pred = reg.predict(X_val)
            rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))

        elif model == "xgboost":
            reg = XGBRegressor(**params)
            reg.fit(X_tr, y_tr)
            y_pred = reg.predict(X_val)
            rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))

        else:
            raise ValueError(f"Unsupported model: {model}")

    if model == "random_forest":
        return {
            "random_forest_50_percentile": np.mean(rmse_scores),
            "random_forest_5_percentile": np.mean(rmse_lower),
            "random_forest_95_percentile": np.mean(rmse_upper)
        }
    else:
        return {
            model
            : np.mean(rmse_scores)
        }
