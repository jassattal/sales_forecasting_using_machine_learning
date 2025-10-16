import pandas as pd
import warnings
import stacks.config as config
warnings.filterwarnings("ignore")


def post_processing(result_dict, test_dates, df_sales_f, model, stream, key, run_date,  description, forecast_horizon, trained_till):
    result_pred_overall = pd.DataFrame()
    for model_type in result_dict.keys():
        results_pred = pd.DataFrame()
        results_pred["Month_Adjusted"] = test_dates[-forecast_horizon:]
        results_pred["run_date"] = run_date
        results_pred["run_description"] = description
        results_pred["forecast_horizon"] = forecast_horizon
        results_pred["train_till_date"] = trained_till

        results_pred["Business_Stream"] = stream
        results_pred["Business_Parition"] = key
        results_pred["Model"] = model
        results_pred["Model_Type"] = model_type
        results_pred["Forecast"] = result_dict[model_type]
        result_pred_overall = pd.concat([result_pred_overall, results_pred])
        print(result_dict[model_type])

    return result_pred_overall


def put_actual_data(key_result_df, df_sales_f, stream, key, run_date,  description, forecast_horizon, trained_till):
    # Put Acutals Data
    actuals_df = pd.DataFrame()
    actuals_df["Month_Adjusted"] = df_sales_f["Month_Adjusted"]
    actuals_df["run_date"] = run_date
    actuals_df["run_description"] = description
    actuals_df["forecast_horizon"] = forecast_horizon
    actuals_df["train_till_date"] = trained_till

    actuals_df["Business_Stream"] = stream
    actuals_df["Business_Parition"] = key
    actuals_df["Model"] = "actuals"
    actuals_df["Model_Type"] = "actuals"
    actuals_df["Forecast"] = df_sales_f["Net_Sales"]
    key_result_df = pd.concat([actuals_df, key_result_df])
    return key_result_df


def saving_results(overall_results_df, stream, run_date, description,scores_df,hyperparams):

    path = "results\model_results_{current_date}_{running_date}_{desc}_.csv".format(current_date=str(
        pd.to_datetime("today").date()), running_date=str(config.run_date.date()), desc=config.description)
    """    path="results\model_results_" +
        str(run_date)+"_"+str(description) +
        str(pd.to_datetime("today").date())+".csv"""
    overall_results_df.to_csv(path, index=False)
    print("Sucessfully Saved results for {b_stream}".format(b_stream=stream))
    path_scores_df="scores\model_results_scores{current_date}_{running_date}_{desc}.csv".format(current_date=str(
        pd.to_datetime("today").date()), running_date=str(config.run_date.date()), desc=config.description)
    scores_df["run_date"] = run_date
    scores_df["run_description"]=description
    scores_df.to_csv(path_scores_df, index=False)
    path_hyperparams="Hyperparams\params.csv"
    if config.hptuning==1:
        hyperparams.to_csv(path_hyperparams,index=False)
    print("Sucessfully Saved scores for {b_stream}".format(b_stream=stream))
    
    return
