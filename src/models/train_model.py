import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import make_scorer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
#from src.features.build_features import add_calendar_features, add_holidays_features, fill_time_series
from src.features.build_features import MetadataTransformer, CalendarTransformer, HolidaysTransformer, fill_time_series
from src.evaluation.scoring import wmape, wbias, calculate_errors
from src.data.splitting import split_dataset, time_series_cv, check_validation_splits_sparsity
from src.visualization.visualize import plot_fit_and_residuals
from src.utils import get_project_root
from src.data.make_dataset import load_dataset


class PositiveOnlyRidge(Ridge):

    def predict(self, X):
        predictions = super().predict(X)
        predictions[predictions < 0] = 0
        return predictions


def get_predictions(fitted_gscv, X_train, X_test):
    y_pred_train = fitted_gscv.predict(X_train)
    y_pred_train = pd.Series(y_pred_train, index=X_train.index)
    y_pred_test = fitted_gscv.predict(X_test)
    y_pred_test = pd.Series(y_pred_test, index=X_test.index)
    return y_pred_train, y_pred_test

def get_features_coefs(fitted_gscv):
    feats_names = fitted_gscv.best_estimator_.named_steps['holidays_tf'].get_feature_names()
    regressor_coefs = fitted_gscv.best_estimator_.named_steps['regressor'].coef_
    features_coefs = {var: regressor_coefs[idx] for idx,var in enumerate(feats_names)}
    return features_coefs



def transform_and_fit_gridsearch(
    raw_data_df: pd.DataFrame, 
    #independent_vars: list, 
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
    #dataset_df = add_calendar_features(data_filled)
    #dataset_df = add_holidays_features(dataset_df)
    X_train, X_test, y_train, y_test = split_dataset(data_filled, validation_split_date, dependent_var)
    cv_split_idxs = time_series_cv(X_train, num_train_years=3, percentage_cut=0.8)
    check_validation_splits_sparsity(cv_split_idxs, y_train)
    wmape_scorer = make_scorer(wmape, greater_is_better=False)
    grid_search_params = {'regressor__alpha': [0.01, 0.1, 1.0, 10, 100]}
    pipeline = Pipeline(steps=[
                       ('metadata_tf', MetadataTransformer()),
                       ('calendar_tf', CalendarTransformer()),
                       ('holidays_tf', HolidaysTransformer()),
                       ('regressor', PositiveOnlyRidge())
                        ])
    lin_reg_gscv = GridSearchCV(
        pipeline, 
        grid_search_params,
        scoring=wmape_scorer,
        cv=cv_split_idxs,
        refit=True).fit(X_train, y_train)

    features_coefs = get_features_coefs(lin_reg_gscv)
    y_pred_train, y_pred_test = get_predictions(lin_reg_gscv, X_train, X_test)
    errors_dict = calculate_errors(y_train, y_test, y_pred_train, y_pred_test)
    train_residuals = y_train - y_pred_train
    test_residuals = y_test - y_pred_test
    
    if visualize:
        plot_fit_and_residuals(y_train, 
                               y_pred_train, 
                               y_test, 
                               y_pred_test, 
                               pd.Series(features_coefs),
                               train_residuals, 
                               test_residuals)
    return {
        "item_name": item_name,
        "best_estimator": lin_reg_gscv.best_estimator_,#lin_reg_gscv.best_estimator_.named_steps['regressor'].__class__.__name__,
        "train_from": X_train.index.min().strftime('%Y-%m-%d'),
        "train_to": X_train.index.max().strftime('%Y-%m-%d'),
        "test_from": X_test.index.min().strftime('%Y-%m-%d'),
        "test_to": X_test.index.max().strftime('%Y-%m-%d'),
        "train_wmape": errors_dict['train_wmape'],
        "test_wmape": errors_dict['test_wmape'],
        "validation_wmape": abs(lin_reg_gscv.best_score_),
        "train_wbias": errors_dict['train_wbias'],
        "test_wbias": errors_dict['test_wbias'],
        "features_coefs": features_coefs,
        "train_data_sparsity": round((len(y_train[y_train==0])/len(y_train))*100, 2),
        "test_data_sparsity": round((len(y_test[y_test==0])/len(y_test))*100, 2)
    }    



if __name__ == '__main__':
    #TODO:
    #- filter ako je bolje samo monhtly feature
    #- možda i temperaturu
    #- stavit slike koje pokazuju poboljšanje preciznosti
    # Stavit info o praznom validation setu
    
    PROJECT_ROOT_PATH = get_project_root()
    #TRAIN_FILENAME = os.path.join(PROJECT_ROOT_PATH, 'data/interim/train_data_90_perc_value_v1_3.csv')
    #TEST_FILENAME = os.path.join(PROJECT_ROOT_PATH, 'data/interim/test_data_90_perc_value_v1_3.csv')
    SCORES_DIR = os.path.join(PROJECT_ROOT_PATH,'reports/scores/')
    SCORE_FILENAME = pd.to_datetime('today').strftime('%Y_%m_%d') + '_scores.csv'
    DEPENDENT_VAR = 'sales_qty'
    VALIDATION_SPLIT_DATE = pd.to_datetime('2019-01-01', utc=True)
    print("Loading and preprocessing raw data ...")
    dataset = load_dataset()
    train_data = dataset.set_index('sales_date')

    cols = ['item_name', 'best_estimator', 'train_from', 'train_to', 
            'test_from', 'test_to', 'train_wmape', 'test_wmape', 
            'validation_wmape', 'train_wbias', 'test_wbias', 'features_coefs',
            'train_data_sparsity', 'test_data_sparsity']
    metrics_df = pd.DataFrame(columns=cols)

    for item in tqdm(train_data.item_name.unique().tolist()):
        print(f"\nProcessing item: {item}")
        raw_data_df = train_data[train_data.item_name == item].copy()
        if (raw_data_df.index.max() > VALIDATION_SPLIT_DATE) & (raw_data_df.index.min() < VALIDATION_SPLIT_DATE):
            print(f"Fitting a model for {item}")
            metrics_dict = transform_and_fit_gridsearch(raw_data_df, DEPENDENT_VAR, VALIDATION_SPLIT_DATE, visualize=False)
            metrics_df = metrics_df.append(metrics_dict, ignore_index=True)
        else:
            print(f"Condition (raw_data_df.index.max() > VALIDATION_SPLIT_DATE) & (raw_data_df.index.min() < VALIDATION_SPLIT_DATE) not fulfilled for {item}")
    metrics_df.to_csv(SCORES_DIR+SCORE_FILENAME, index=False)
