import os
import pandas as pd

def load_data(data_abs_path: str) -> pd.DataFrame:
    """Load raw data
    
    Parameters:
    -----------
    data_abs_path: absolute path of csv data

    Returns:
    --------
    data_df: raw data dataframe

    """
    data_df = pd.read_csv(data_abs_path)
    data_df.sales_datetime = pd.to_datetime(data_df.sales_datetime, format='%Y-%m-%d', utc=True)
    data_df.set_index('sales_datetime', inplace=True)
    return data_df