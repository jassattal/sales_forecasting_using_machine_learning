import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import warnings
import stacks.config as config
warnings.filterwarnings("ignore")


def selection(x_train, y_train, stream, key):
    #Removing the features which are not required
    features = x_train.columns
    remove_features = ["index","Month_no", "Year", "Number", "Covid"]
    for i in remove_features:
        if i in features:
            features = features.drop(i)
    random_forest = RandomForestRegressor()
    rf = random_forest.fit(x_train[features], y_train)
    Feature_imp = pd.DataFrame()
    Feature_imp["Feature"] = x_train[features].columns
    Feature_imp["Importance"] = (rf.feature_importances_)*100
    Feature_imp = Feature_imp[Feature_imp["Importance"] > 0.1]
    Feature_imp["stream"] = stream
    Feature_imp["key"] = key
    Feature_imp.sort_values(by="Importance", ascending=False, inplace=True)
    sel_feature = Feature_imp.Feature.to_list()
    """    if df_fe.shape[0]>202:
        hard_coded_features=["Covid"]
        for i in hard_coded_features:

            sel_features=sel_feature.append(i)"""
    # sel_features=sel_feature.append("OnlineNetSales")
    sel_feature = list(set(sel_feature))
    feat_sel_path="feat_imp/model_results_feat_imp{current_date}_{running_date}_{desc}_{stream}.csv".format(current_date=str(
        pd.to_datetime("today").date()), running_date=str(config.run_date.date()), desc=config.description,stream=stream)
    Feature_imp.to_csv(feat_sel_path, index=False)

    

    return sel_feature
