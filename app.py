import stacks.load_data as load_data
import stacks.sales_data_preprocess as sales_data_preprocess
import stacks.feature_engineering as feature_engineering
import stacks.feature_selection as feature_selection
import stacks.merge_data as merge_data
import stacks.post_processing as post_processing
import stacks.config as config
import stacks.modelling as modelling

from stacks.cross_validations import univariate as univariate_cv
from stacks.cross_validations import multivariate as multivariate_cv

import stacks.best_fit as best_fit

import ast


import stacks.hptuning as hptuning


import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Initialize result containers
overall_results_df = pd.DataFrame()
hyperparams=pd.DataFrame(columns=["stream", "key", "model", "Params"])
scores_df = pd.DataFrame(columns=["stream", "key", "model", "cv_rmse_mean"])

# Loop through each stream
for stream in config.streams:
    stream_result_df = pd.DataFrame()
    df = load_data.load(stream)
    df_grouped, valid_keys = sales_data_preprocess.sales_preprocessing(df, stream)

    for key in valid_keys:
        df_sales_f = df_grouped[df_grouped["Key"] == key][["Net_Sales"]].reset_index()
        df_merged, test_dates = merge_data.merge(
            df_sales_f, config.max_training_date, config.forecast_horizon
        )
        x_train, x_test, y_train, train_dates = feature_engineering.fe(
            df_merged, df_grouped, stream, key, config.forecast_horizon
        )
        selected_features = feature_selection.selection(x_train, y_train, stream, key)

        model_result_df = pd.DataFrame()
        counter = 0

        for model in config.models:
            print(f"Running for {model}")


            # Step 1: Hyperparameter tuning (if applicable)
            if (model in ["random_forest", "catboost", "xgboost"]) and (config.hptuning==1):

                print("Hyper params tuning is on",(config.hptuning))


                best_params = hptuning.tune_model(
                    x_train=x_train,
                    y_train=y_train,
                    sel_feature=selected_features,
                    model=model
                )
                print(f"Best parameters for {model}: {best_params}")
                hyperparams=pd.concat([hyperparams,pd.DataFrame([{"stream": stream,"key": key,"model": model,"Params":best_params}])], ignore_index=True)
            elif (model in ["random_forest", "catboost", "xgboost"]) and (config.hptuning!=1):
                print("Hyper params tuning is off",model,key)
                best_params_data=pd.read_csv("Hyperparams\params.csv")
                print(best_params_data[(best_params_data["model"]==model)&(best_params_data["key"]==key)]["Params"].values[0])
                best_params=ast.literal_eval(best_params_data[(best_params_data["model"]==model)&(best_params_data["key"]==key)]["Params"].values[0])

                print(best_params)

            else:
                
                best_params = {"model": model, "params": None,"cv_rmse":None}  # Prophet or models without tuning
            
            
            


            # Step 1: Cross-validation
            if model == "prophet":
                cv_result = univariate_cv(
                    y_train=y_train,
                    model=model,
                    forecast_horizon=config.forecast_horizon,
                    train_dates=train_dates,
                    n_splits=5
                )
            else:
                cv_result = multivariate_cv(
                    x_train=x_train,
                    y_train=y_train,
                    sel_feature=selected_features,
                    model=model,
                    n_splits=5,
                    forecast_horizon=config.forecast_horizon,
                    hyper_params=best_params["params"]

                )

            # Step 2: Save CV scores for each variant
            for model_variant, rmse in cv_result.items():


                scores_df = pd.concat([
                    scores_df,
                    pd.DataFrame([{
                        "stream": stream,
                        "key": key,
                        "model": model_variant,
                        "cv_rmse_mean": rmse
                    }])
                ], ignore_index=True)


            # Step 3: Forecasting (use base model name only)
            if model == "prophet":
                result_dict = modelling.univariate(
                    y_train=y_train,
                    model=model,
                    forecast_horizon=config.forecast_horizon,
                    train_dates=train_dates
                )
            else:
                result_dict = modelling.multivariate(
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    sel_feature=selected_features,
                    model=model,
                    hyper_params=best_params["params"]
                )

            # Step 4: Post-processing
            key_result_df = post_processing.post_processing(
                result_dict, test_dates, df_sales_f, model, stream, key,
                config.run_date, config.description, config.forecast_horizon, config.max_training_date
            )

            if counter == 0:
                key_result_df = post_processing.put_actual_data(
                    key_result_df, df_sales_f, stream, key,
                    config.run_date, config.description, config.forecast_horizon, config.max_training_date
                )
                counter += 1
            

            model_result_df = pd.concat([model_result_df, key_result_df])
        
        # Compute and append best fit forecasts
        model_result_df = best_fit.compute_best_fit(scores_df,stream,key, config.run_date,  config.description, config.forecast_horizon, config.max_training_date, model_result_df)
            


        stream_result_df = pd.concat([stream_result_df, model_result_df])

    overall_results_df = pd.concat([overall_results_df, stream_result_df])
    # Compute and append best fit forecasts
    #overall_results_df = best_fit.compute_best_fit(scores_df,stream,key, config.run_date,  config.description, config.forecast_horizon, config.max_training_date, overall_results_df)
    print(overall_results_df.Model_Type.unique())



# Save results
print("Modeling and Post Processing Completed")
post_processing.saving_results(overall_results_df, stream, config.run_date, config.description,scores_df,hyperparams)
print("Results saved successfully")
print("Forecasting Done Successfully")

#Testing out checkout

