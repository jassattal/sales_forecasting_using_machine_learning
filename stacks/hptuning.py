import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from quantile_forest import RandomForestQuantileRegressor
import numpy as np
import stacks.config as config

def tune_model(x_train, y_train, sel_feature, model, hp_tuning=config.hptuning, n_trials=20, n_splits=5):
    print(hp_tuning)
    if not hp_tuning:
        return {"model": model, "params": "default", "cv_rmse": None}

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scorer = make_scorer(mean_squared_error, greater_is_better=False)

    def objective(trial):
        if model == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state": 0
            }
            model_instance = XGBRegressor(**params)

        elif model == "catboost":
            params = {
                "iterations": trial.suggest_int("iterations", 100, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "depth": trial.suggest_int("depth", 3, 10),
                "random_state": 0,
                "verbose": 0
            }
            model_instance = CatBoostRegressor(**params)

        elif model == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
                "random_state": 0
            }
            model_instance = RandomForestQuantileRegressor(**params)

        else:
            raise ValueError(f"Unsupported model: {model}")

        scores = cross_val_score(model_instance, x_train[sel_feature], y_train, cv=tscv, scoring=scorer)
        return -1 * np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return {
        "model": model,
        "params": study.best_params,
        "cv_rmse": study.best_value
    }
