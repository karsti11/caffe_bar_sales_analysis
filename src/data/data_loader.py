import os
import pandas as pd

def load_data(root_path: str, filename: str) -> pd.DataFrame:
    """Load raw data
    
    Parameters:
    -----------
    root_path: project root path
    filename: ž

    Returns:
    --------
    data_df: raw data dataframe

    """
    data_df = pd.read_csv(os.path.join(root_path, filename))
    data_df.sales_datetime = pd.to_datetime(data_df.sales_datetime, format='%Y-%m-%d', utc=True)
    data_df.set_index('sales_datetime', inplace=True)
    return data_df