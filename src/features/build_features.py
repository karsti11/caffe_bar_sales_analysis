import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.features.calendar import easter_dates, easter_monday_dates

def fill_time_series(
    raw_data_df: pd.DataFrame
    ) -> pd.DataFrame:
    """Fills data for missing dates in raw dataframe per item. 
    Dataframe must have daily DatetimeIndex.
    """
    data_df = raw_data_df.resample('D').sum()
    data_df.item_price = data_df.item_price.ffill().bfill()
    return data_df


class CalendarTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy() # creating a copy to avoid changes to original dataset
        X_.loc[:,'day_of_week'] = X_.index.day_of_week
        X_.loc[:,'month_of_year'] = X_.index.month
        day_of_week_dummies = pd.get_dummies(X_.day_of_week, prefix='day_of_week', drop_first=True)
        month_of_year_dummies = pd.get_dummies(X_.month_of_year, prefix='month_of_year', drop_first=True)
        X_ = X_.merge(day_of_week_dummies, how='left', left_index=True, right_index=True)
        X_ = X_.merge(month_of_year_dummies, how='left', left_index=True, right_index=True)
        X_.loc[:,'year'] = X_.index.year - X_.index.year.min()
        X_.loc[:,'first_third_of_month'] = (X_.index.day <= 10).astype('int8')
        X_.loc[:,'second_third_of_month'] = ((X_.index.day > 10) & (X_.index.day <= 20)).astype('int8')
        X_.loc[:,'last_third_of_month'] = (X_.index.day > 20).astype('int8')
        X_.drop(columns=['day_of_week', 'month_of_year'], inplace=True)

        return X_


class HolidaysTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, feature_names=None):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy() # creating a copy to avoid changes to original dataset
        X_.loc[:, 'easter'] = X_.index.isin(easter_dates).astype('int8')
        X_.loc[:, 'easter_monday'] = X_.index.isin(easter_monday_dates).astype('int8')
        X_.loc[:, 'christmas'] = ((X_.index.month==12) & (X_.index.day==25)).astype('int8')
        X_.loc[:, 'new_years_day'] = ((X_.index.month==1) & (X_.index.day==1)).astype('int8')
        X_.loc[:, 'new_years_eve'] = ((X_.index.month==12) & (X_.index.day==31)).astype('int8')
        self.feature_names = X_.columns.tolist()

        return X_

    def get_feature_names(self):
        return self.feature_names



def add_calendar_features(
    raw_data_df: pd.DataFrame
    ) -> pd.DataFrame:

    """Adds calendar features to raw dataframe(days of week, month of year, 
    year, thirds of month). Dataframe must have daily DatetimeIndex.
    """
    transformed_data_df = raw_data_df.copy()
    transformed_data_df.loc[:,'day_of_week'] = transformed_data_df.index.day_of_week
    transformed_data_df.loc[:,'month_of_year'] = transformed_data_df.index.month
    day_of_week_dummies = pd.get_dummies(transformed_data_df.day_of_week, prefix='day_of_week', drop_first=True)
    month_of_year_dummies = pd.get_dummies(transformed_data_df.month_of_year, prefix='month_of_year', drop_first=True)
    transformed_data_df = transformed_data_df.merge(day_of_week_dummies, how='left', left_index=True, right_index=True)
    transformed_data_df = transformed_data_df.merge(month_of_year_dummies, how='left', left_index=True, right_index=True)
    transformed_data_df.loc[:,'year'] = transformed_data_df.index.year - transformed_data_df.index.year.min()
    transformed_data_df.loc[:,'first_third_of_month'] = (transformed_data_df.index.day <= 10).astype('int8')
    transformed_data_df.loc[:,'second_third_of_month'] = ((transformed_data_df.index.day > 10) & (transformed_data_df.index.day <= 20)).astype('int8')
    transformed_data_df.loc[:,'last_third_of_month'] = (transformed_data_df.index.day > 20).astype('int8')
    transformed_data_df.drop(columns=['day_of_week', 'month_of_year'], inplace=True)

    return transformed_data_df

def add_holidays_features(
    data_df: pd.DataFrame
    ) -> pd.DataFrame:
    """Add easter and easter monday dummy variables. 
    DataFrame must have DatetimeIndex.
    """

    data_df.loc[:, 'easter'] = data_df.index.isin(easter_dates).astype('int8')
    data_df.loc[:, 'easter_monday'] = data_df.index.isin(easter_monday_dates).astype('int8')
    data_df.loc[:, 'christmas'] = ((data_df.index.month==12) & (data_df.index.day==25)).astype('int8')
    data_df.loc[:, 'new_years_day'] = ((data_df.index.month==1) & (data_df.index.day==1)).astype('int8')
    data_df.loc[:, 'new_years_eve'] = ((data_df.index.month==12) & (data_df.index.day==31)).astype('int8')
    
    return data_df

#def add_special_events(data_df: pd.DataFrame) -> pd.DataFrame:
