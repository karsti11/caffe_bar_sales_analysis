import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from src.features.build_features import add_calendar_features, add_holidays_features, fill_time_series
from src.data.data_loader import load_data
from src.evaluation.metrics import wmape
from src.data.splitting import split_dataset, time_series_cv
from src.visualization.visualize import plot_fit_and_residuals
from src.utils import get_project_root

def transform_and_fit(
    raw_data_df: pd.DataFrame, 
    independent_vars: list, 
    dependent_var: str, 
    validation_split_date: str,
    visualize: bool):
    """ 
    Fills missing time series data in between raw_data_df.index.min() and raw_data_df.index.max(),
    adds calendar features, splits training dataset into training and validation dataset,
    fits estimator on training data and predicts on validation data.
    
    Parameters
    -----------
    raw_data_df: Time series dataframe
    independent_vars: list of column names to be used as independent variables for scikit-learn estimator
    dependent_var: column name to be used as dependent variable (target) for scikit-learn estimator
    validation_split_date: first date of validation set
    visualize: to plot or not
    
    Returns
    --------
    metrics_dict: metadata and metrics for individual item
    
    """
    item_name = raw_data_df.item_name.unique()[0]
    print(f"Transform and fit for {item_name}")
    data_filled = fill_time_series(raw_data_df.copy())
    data_w_feats = add_features_to_raw_data(data_filled)
    X_train, X_val, y_train, y_val = split_dataset(data_w_feats, validation_split_date, independent_vars, dependent_var)
    lin_reg = LinearRegression().fit(X_train, y_train)
    y_pred_train = lin_reg.predict(X_train)
    y_pred_train = pd.Series(y_pred_train, index=y_train.index)
    train_wmape = wmape(y_train, y_pred_train)
    print(f"Train WMAPE: {train_wmape}")
    y_pred_val = lin_reg.predict(X_val)
    y_pred_val = pd.Series(y_pred_val, index=y_val.index)
    val_wmape = wmape(y_val, y_pred_val)
    print(f"Validation WMAPE: {val_wmape}")
    
    features_coefs = {var:lin_reg.coef_[idx] for idx,var in enumerate(independent_vars)}
    features_coefs_df = pd.Series(features_coefs)
    train_residuals = y_train - y_pred_train
    test_residuals = y_val - y_pred_val
    
    if visualize:
        plot_fit_and_residuals(y_train, 
                               y_pred_train, 
                               y_val, 
                               y_pred_val, 
                               features_coefs_df,
                               train_residuals, 
                               test_residuals)
    return {
        "item_name": item_name,
        "estimator": lin_reg.__class__.__name__,
        "train_from": X_train.index.min().strftime('%Y-%m-%d'),
        "train_to": X_train.index.max().strftime('%Y-%m-%d'),
        "val_from": X_val.index.min().strftime('%Y-%m-%d'),
        "val_to": X_val.index.max().strftime('%Y-%m-%d'),
        "train_wmape": train_wmape,
        #"train_smape": train_smape,
        "val_wmape": val_wmape,
        #"val_smape": val_smape,
        "features_coefs": features_coefs
    }
    
def transform_and_fit_gridsearch(
    raw_data_df: pd.DataFrame, 
    independent_vars: list, 
    dependent_var: str, 
    validation_split_date: str,
    visualize: bool):
    """ 
    Fills missing time series data in between raw_data_df.index.min() and raw_data_df.index.max(),
    adds calendar features, splits training dataset into training and validation dataset,
    fits estimator on training data and predicts on validation data.
    
    Parameters
    -----------
    raw_data_df: Time series dataframe
    independent_vars: list of column names to be used as independent variables for scikit-learn estimator
    dependent_var: column name to be used as dependent variable (target) for scikit-learn estimator
    validation_split_date: first date of validation set
    visualize: to plot or not
    
    Returns
    --------
    metrics_dict: metadata and metrics for individual item
    
    """
    item_name = raw_data_df.item_name.unique()[0]
    print(f"Executing transform and gridsearch fit for {item_name} ...")
    data_filled = fill_time_series(raw_data_df.copy())
    dataset_df = add_calendar_features(data_filled)
    dataset_df = add_holidays_features(dataset_df)
    X_train, X_test, y_train, y_test = split_dataset(dataset_df, validation_split_date, independent_vars, dependent_var)
    cv_split_idxs = time_series_cv(X_train, num_train_years=3)
    wmape_scorer = make_scorer(wmape, greater_is_better=False)
    grid_search_params = {'alpha': [0.01, 0.1, 1.0, 10, 100]}
    lin_reg_gscv = GridSearchCV(
        Ridge(), 
        grid_search_params,
        scoring=wmape_scorer,
        cv=cv_split_idxs).fit(X_train, y_train)
    
    y_pred_train = lin_reg_gscv.predict(X_train)
    y_pred_train = pd.Series(y_pred_train, index=y_train.index)
    train_wmape = wmape(y_train, y_pred_train)
    print(f"Train WMAPE: {train_wmape}")
    y_pred_test = lin_reg_gscv.predict(X_test)
    y_pred_test = pd.Series(y_pred_test, index=y_test.index)
    test_wmape = wmape(y_test, y_pred_test)
    print(f"Validation WMAPE: {test_wmape}")
    
    features_coefs = {var:lin_reg_gscv.best_estimator_.coef_[idx] for idx,var in enumerate(independent_vars)}
    features_coefs_df = pd.Series(features_coefs)
    train_residuals = y_train - y_pred_train
    test_residuals = y_test - y_pred_test
    
    if visualize:
        plot_fit_and_residuals(y_train, 
                               y_pred_train, 
                               y_test, 
                               y_pred_test, 
                               features_coefs_df,
                               train_residuals, 
                               test_residuals)
    return {
        "item_name": item_name,
        "estimator": lin_reg_gscv.best_estimator_.__class__.__name__,
        "train_from": X_train.index.min().strftime('%Y-%m-%d'),
        "train_to": X_train.index.max().strftime('%Y-%m-%d'),
        "test_from": X_test.index.min().strftime('%Y-%m-%d'),
        "test_to": X_test.index.max().strftime('%Y-%m-%d'),
        "train_wmape": train_wmape,
        "test_wmape": test_wmape,
        "features_coefs": features_coefs
    }    



if __name__ == '__main__':
    
    PROJECT_ROOT_PATH = get_project_root()
    TRAIN_FILENAME = os.path.join(PROJECT_ROOT_PATH, 'data/interim/train_data_90_perc_value_v1_3.csv')
    TEST_FILENAME = os.path.join(PROJECT_ROOT_PATH, 'data/interim/test_data_90_perc_value_v1_3.csv')
    SCORES_DIR = os.path.join(PROJECT_ROOT_PATH,'reports/scores/')
    SCORE_FILENAME = pd.to_datetime('today').strftime('%Y_%m_%d') + 'scores.csv'

    INDEPENDENT_VARS = ['item_price','day_of_week_0',
       'day_of_week_1', 'day_of_week_2', 'day_of_week_3', 'day_of_week_4',
       'day_of_week_5', 'day_of_week_6', 'month_of_year_1', 'month_of_year_2',
       'month_of_year_3', 'month_of_year_4', 'month_of_year_5',
       'month_of_year_6', 'month_of_year_7', 'month_of_year_8',
       'month_of_year_9', 'month_of_year_10', 'month_of_year_11',
       'month_of_year_12', 'year', 'first_third_of_month',
       'second_third_of_month', 'last_third_of_month']
    DEPENDENT_VAR = 'sales_qty'
    VALIDATION_SPLIT_DATE = '2018-01-01'
    train_data = load_data(TRAIN_FILENAME)

    cols = ['estimator', 'train_from', 'train_to', 'train_wmape', 'features_coefs']
    metrics_df = pd.DataFrame(columns=cols)
    train_data.item_name.unique().tolist()

    for item in train_data.item_name.unique().tolist():
        raw_data_df = train_data[train_data.item_name == item].copy()
        if raw_data_df.index.max() > pd.to_datetime(VALIDATION_SPLIT_DATE, utc=True):
            metrics_dict = transform_and_fit_gridsearch(raw_data_df, INDEPENDENT_VARS, DEPENDENT_VAR, VALIDATION_SPLIT_DATE, visualize=False)
            metrics_df = metrics_df.append(metrics_dict, ignore_index=True)

    print(metrics_df)