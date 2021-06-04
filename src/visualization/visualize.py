import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt


def visualize_seasonality(item_daily_data_df):
    """Groups dataset and calculates means of target variable to make plots of 
    trend and seasonalities.
    """
    year_month_mean = item_daily_data_df.groupby(pd.Grouper(freq='MS')).agg({'sales_qty':'mean'})
    month_mean = item_daily_data_df.groupby(item_daily_data_df.index.month).agg({'sales_qty':'mean'})
    day_of_week_mean = item_daily_data_df.groupby(item_daily_data_df.index.day_of_week).agg({'sales_qty':'mean'})
    day_of_month_mean = item_daily_data_df.groupby(item_daily_data_df.index.day).agg({'sales_qty':'mean'})

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20,10))
    ax1.plot(year_month_mean)
    ax1.set_title("Overall trend (averaged monthly)")
    ax2.plot(month_mean)
    ax2.set_title("Monthly seasonality")
    ax3.plot(day_of_week_mean)
    ax3.set_title("Day of week average pattern")
    ax4.plot(day_of_month_mean)
    ax4.set_title("Day of month average pattern")

def plot_fit_and_residuals(train_sales: pd.Series, 
                           train_predictions: pd.Series, 
                           val_sales: pd.Series, 
                           val_predictions: pd.Series, 
                           features_coefs: pd.Series,
                           train_residuals: pd.Series, 
                           val_residuals: pd.Series):
    """Renders two figures: 
        1. Line plots of sales and predictions and residual plots for both training 
        and validation set and coefficients plot 
        2. Interactive plot of predictions for train set.

    Parmeters:
    ----------
    train_sales: target variable of train set
    train_predictions: target variable predictions for train set
    val_sales: target variable of validation set 
    val_predictions: target variable predictions for validation set
    features_coefs: series of features coefficients obtained from estimator (Lasso)
    train_residuals: residuals of train set predictions
    val_residuals: residuals of validation set predictions

    """
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(30,20))

    train_sales.plot(kind='line', ax=ax1)
    train_predictions.plot(kind='line', ax=ax1)
    
    val_sales.plot(kind='line', ax=ax2)
    val_predictions.plot(kind='line', ax=ax2)
    
    ax3.scatter(train_residuals, train_predictions)
    ax4.scatter(val_residuals, val_predictions)
    
    features_coefs_df.plot(kind='barh', ax=ax5)
    plt.show()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_sales.index, y=train_sales,
                        mode='lines',
                        name='Sales'))
    fig.add_trace(go.Scatter(x=train_predictions.index, y=train_predictions,
                        mode='lines',
                        name='Predictions'))
    fig.show()