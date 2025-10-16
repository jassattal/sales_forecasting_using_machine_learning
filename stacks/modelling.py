from quantile_forest import RandomForestQuantileRegressor
import catboost as cb
from xgboost import XGBRegressor  

from prophet import Prophet
import warnings
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings("ignore")

def univariate(y_train, model, forecast_horizon, train_dates):
    output_dict = {}
    if model == "prophet":
        y_train = pd.DataFrame(y_train)
        prophet_df = y_train.copy(deep=True)
        prophet_df["Month_Adjusted"] = train_dates
        prophet_df.rename(columns={"Month_Adjusted": "ds",
                                   "Net_Sales": "y"}, inplace=True)

        model_prophet = Prophet("linear", changepoint_prior_scale=0.5, seasonality_prior_scale=1,
                                changepoint_range=0.95, seasonality_mode='multiplicative')
        model_prophet.fit(prophet_df)
        df_future = model_prophet.make_future_dataframe(
            periods=forecast_horizon, freq='MS')
        forecast_prophet = model_prophet.predict(df_future)
        forecast_prophet.set_index("ds", inplace=True)
        forecast_prophet = forecast_prophet.tail(forecast_horizon)
        output_dict = {"prophet_mean": forecast_prophet["yhat"].values,
                       "prophet_lower": forecast_prophet["yhat_lower"].values,
                       "prophet_higher": forecast_prophet["yhat_upper"].values}

    return output_dict


def multivariate(x_train, y_train, x_test, sel_feature, model,hyper_params):
    sel_feature = list(set(sel_feature))
    x_train_final = x_train[sel_feature]
    x_test_final = x_test[sel_feature]

    if model == "random_forest":

        params = hyper_params if hyper_params else {"n_estimators": 1000, "learning_rate": 0.05, "random_state": 0}

        reg_rf = RandomForestQuantileRegressor(**params)
        reg_rf.fit(x_train_final, y_train)
        # Get predictions at 95% prediction intervals and median.
        y_pred_95_rf = reg_rf.predict(x_test_final, quantiles=[0.95])
        y_pred_05_rf = reg_rf.predict(x_test_final, quantiles=[0.05])
        y_pred_50_rf = reg_rf.predict(x_test_final, quantiles=[0.50])
        output_dict = {"random_forest_95_percentile": y_pred_95_rf,
                       "random_forest_5_percentile": y_pred_05_rf,
                       "random_forest_50_percentile": y_pred_50_rf}
    elif model == "catboost":
        params = hyper_params if hyper_params else {"itertions": 1000, "learning_rate": 0.05, "random_state": 0}
        
        reg_cb = cb.CatBoostRegressor(**params)
        reg_cb.fit(x_train_final, y_train)
        y_pred_cb = reg_cb.predict(x_test_final)
        output_dict = {"catboost": y_pred_cb}

    elif model == "linear_regression":
        #x_train_final = sm.add_constant(x_train_final)
        #x_test_final = sm.add_constant(x_test_final)
        model = sm.OLS(y_train, x_train_final).fit()
        y_pred_lr = model.predict(x_test_final)
        x_test.to_csv("x_test.csv")
        x_train_final.to_csv("x_train.csv")
        output_dict = {"linear_regression": y_pred_lr.values}
    
    elif model == "xgboost":
        
        params = hyper_params if hyper_params else {"n_estimators": 1000, "learning_rate": 0.05, "random_state": 0}

        reg_xgb = XGBRegressor(**params)
        reg_xgb.fit(x_train_final, y_train)
        y_pred_xgb = reg_xgb.predict(x_test_final)
        output_dict = {"xgboost": y_pred_xgb}


    return output_dict

