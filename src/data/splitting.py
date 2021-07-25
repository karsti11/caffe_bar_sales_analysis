import pandas as pd
import itertools

def split_dataset(all_data_df: pd.DataFrame, 
                  test_split_date: str, 
                  dependent_var: str):
    """Split dataset by date. 
    First date of test is test_split_date.
    """
    X_train = all_data_df[all_data_df.index < pd.to_datetime(test_split_date, utc=True)].drop(dependent_var, axis=1).copy()
    X_test = all_data_df[all_data_df.index >= pd.to_datetime(test_split_date, utc=True)].drop(dependent_var, axis=1).copy()
    y_train = all_data_df[all_data_df.index < pd.to_datetime(test_split_date, utc=True)][dependent_var].copy()
    y_test = all_data_df[all_data_df.index >= pd.to_datetime(test_split_date, utc=True)][dependent_var].copy()
    print(f"Train dataset is from {X_train.index.min().strftime('%Y-%m-%d')} to {X_train.index.max().strftime('%Y-%m-%d')}")
    print(f"Test dataset is from {X_test.index.min().strftime('%Y-%m-%d')} to {X_test.index.max().strftime('%Y-%m-%d')}")
    return X_train, X_test, y_train, y_test

def time_series_cv(raw_data_filled_df, num_train_years):
    """Custom time-series split in train-validation sets per year.
    Dataset is split depending on 'num_train_years', which is maximum
    number of years in training dataset.
    """
    groups = raw_data_filled_df.reset_index().groupby(raw_data_filled_df.index.year).groups
    sorted_groups = [value.tolist() for (key, value) in sorted(groups.items())]#list of indices per year
    if len(groups.keys()) < 2:
        raise ValueError("Not enough groups for validation set.")
    elif len(groups.keys()) <= num_train_years+1:
        return [(list(itertools.chain(*sorted_groups[:-1])), sorted_groups[-1])]
    else:
        return [(list(itertools.chain(*sorted_groups[i:num_train_years+i])), sorted_groups[i+num_train_years])
          for i in range(len(sorted_groups) - num_train_years)]