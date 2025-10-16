import pandas as pd


def compute_best_fit(scores_df,stream,key, run_date,  description, forecast_horizon, trained_till, model_result_df):
    """
    Computes the best fit forecast by averaging the top 3 models with the lowest CV RMSE scores for each key.
    
    Parameters:
    - scores_df: DataFrame containing CV RMSE scores with columns ['stream', 'key', 'model', 'cv_rmse_mean']
    - overall_results_df: DataFrame containing all model forecasts
    
    Returns:
    - DataFrame with best fit forecasts appended to overall_results_df
    """
    best_fit_results = []

    # Group by stream and key
    for (stream, key), group in scores_df.groupby(['stream', 'key']):
        # Sort models by CV RMSE
        top_models = group.sort_values(by='cv_rmse_mean').head(3)
        top_model_names = top_models['model'].tolist()
        # Filter forecasts for top models
        forecasts = model_result_df[
        (model_result_df['Model_Type'].isin(top_model_names)) &
        (model_result_df['Business_Stream'] == stream) &
        (model_result_df['Business_Parition'] == key)]


        # Average the forecasts
        avg_forecast = forecasts.groupby(['Month_Adjusted']).agg({
            'Forecast': 'mean'
        }).reset_index()
        avg_forecast["run_date"] = run_date
        avg_forecast["run_description"] = description
        avg_forecast["forecast_horizon"] = forecast_horizon
        avg_forecast["train_till_date"] = trained_till
        avg_forecast["Business_Stream"] = stream
        avg_forecast["train_till_date"] = trained_till

        avg_forecast["Business_Stream"] = stream
        avg_forecast["Business_Parition"] = key

        # Add metadata
        model_name = f"best_fit_{'_'.join(top_model_names)}"
        avg_forecast["Model"]="Best-Fit"
        avg_forecast['Model_Type'] = model_name

        best_fit_results.append(avg_forecast)

    # Combine with original results
    best_fit_df = pd.concat(best_fit_results, ignore_index=True)

    final_df = pd.concat([model_result_df, best_fit_df], ignore_index=True)

    return final_df
