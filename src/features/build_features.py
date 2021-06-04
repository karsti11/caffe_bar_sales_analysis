import pandas as pd

def fill_time_series(raw_data_df: pd.DataFrame) -> pd.DataFrame:
    """Fills data for missing dates in raw dataframe per item. 
    Dataframe must have daily DatetimeIndex.
    """
    data_df = raw_data_df.resample('D').sum()
    data_df.item_price = data_df.item_price.ffill().bfill()
    return data_df

def add_features_to_raw_data(raw_data_df: pd.DataFrame) -> pd.DataFrame:

    """Adds calendar features to raw dataframe(days of week, month of year, 
    year, thirds of month). Dataframe must have daily DatetimeIndex.
    """
    transformed_data_df = raw_data_df.copy()
    transformed_data_df.loc[:,'day_of_week'] = transformed_data_df.index.day_of_week
    transformed_data_df.loc[:,'month_of_year'] = transformed_data_df.index.month
    day_of_week_dummies = pd.get_dummies(transformed_data_df.day_of_week, prefix='day_of_week')
    month_of_year_dummies = pd.get_dummies(transformed_data_df.month_of_year, prefix='month_of_year')
    transformed_data_df = transformed_data_df.merge(day_of_week_dummies, how='left', left_index=True, right_index=True)
    transformed_data_df = transformed_data_df.merge(month_of_year_dummies, how='left', left_index=True, right_index=True)
    transformed_data_df.loc[:,'year'] = transformed_data_df.index.year - transformed_data_df.index.year.min()
    transformed_data_df.loc[:,'first_third_of_month'] = (transformed_data_df.index.day <= 10).astype('int8')
    transformed_data_df.loc[:,'second_third_of_month'] = ((transformed_data_df.index.day > 10) & (transformed_data_df.index.day <= 20)).astype('int8')
    transformed_data_df.loc[:,'last_third_of_month'] = (transformed_data_df.index.day > 20).astype('int8')
    transformed_data_df.drop(columns=['day_of_week', 'month_of_year'], inplace=True)

    return transformed_data_df