import numpy as np
import pandas as pd

def wmape(actual: pd.Series, forecast: pd.Series):
    """Weighted mean absolute percentage error:
        "...variant of MAPE in which errors are weighted by values of actuals..."
        from https://en.wikipedia.org/wiki/WMAPE
    - Disadvantage: when forecast is larger tha

    Parameters:
    -----------
    actual: array of 
    """
    score = (np.abs(actual - forecast).sum() / np.abs(actual).sum())*100
    return round(score, 2)