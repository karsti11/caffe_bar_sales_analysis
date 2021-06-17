import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf


def visualize_seasonality(
    item_daily_data_df: pd.DataFrame
    ) -> None:
    """Groups dataset and calculates means of target variable to make plots of 
    trend and seasonalities.
    """
    year_month_mean = item_daily_data_df.groupby(
        pd.Grouper(freq='MS')).agg(
        {'sales_qty':'mean'})
    month_mean = item_daily_data_df.groupby(
        item_daily_data_df.index.month).agg(
        {'sales_qty':'mean'})
    day_of_week_mean = item_daily_data_df.groupby(
        item_daily_data_df.index.day_of_week).agg(
        {'sales_qty':'mean'})
    day_of_month_mean = item_daily_data_df.groupby(
        item_daily_data_df.index.day).agg(
        {'sales_qty':'mean'})

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        nrows=2, 
        ncols=2, 
        figsize=(20,10))
    ax1.plot(year_month_mean)
    ax1.set_title("Overall trend (averaged monthly)")
    ax2.plot(month_mean)
    ax2.set_title("Monthly seasonality")
    ax3.plot(day_of_week_mean)
    ax3.set_title("Day of week average pattern")
    ax4.plot(day_of_month_mean)
    ax4.set_title("Day of month average pattern")

def plot_fit_and_residuals(
    train_sales: pd.Series, 
    train_predictions: pd.Series, 
    test_sales: pd.Series, 
    test_predictions: pd.Series, 
    features_coefs: pd.Series,
    train_residuals: pd.Series, 
    test_residuals: pd.Series
    ) -> None:
    """Renders two figures: 
        1. Line plots of sales and predictions and residual plots for both training 
        and test set and coefficients plot 
        2. Interactive plot of predictions for train set.

    Parmeters:
    ----------
    train_sales: target variable of train set
    train_predictions: target variable predictions for train set
    test_sales: target variable of test set 
    test_predictions: target variable predictions for test set
    features_coefs: series of features coefficients obtained from estimator (Lasso)
    train_residuals: residuals of train set predictions
    test_residuals: residuals of test set predictions

    """
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
        nrows=3, 
        ncols=2, 
        gridspec_kw= {'hspace': 0.6}, 
        figsize=(30,20))

    train_sales.plot(
        kind='line', 
        title='Actual sales (train set)', 
        xlabel='Date', 
        ylabel='Sales (pcs)',
        fontsize=16,
        ax=ax1)
    train_predictions.plot(
        kind='line',
        fontsize=16, 
        ax=ax1)
    test_sales.plot(
        kind='line', 
        title='Actual sales (test set)', 
        xlabel='Date', 
        ylabel='Sales (pcs)',
        fontsize=16, 
        ax=ax2)
    test_predictions.plot(
        kind='line',
        fontsize=16, 
        ax=ax2)
    ax3.scatter(
        train_residuals, 
        train_predictions)
    ax3.set_title("Train set residuals plot", size=16)
    ax3.set_xlabel("Predictions", fontsize = 16)
    ax3.set_ylabel("Actual - Predictions", fontsize = 16)
    ax4.scatter(
        test_residuals, 
        test_predictions)
    ax4.set_title("Test set residuals plot", size=16)
    ax4.set_xlabel("Predictions", fontsize = 16)
    ax4.set_ylabel("Actual - Predictions", fontsize = 16)
    
    features_coefs.plot(
        kind='barh', 
        title='Features coefficients',
        fontsize=16, 
        ax=ax5)
    plot_acf(
        train_sales, 
        ax=ax6)
    plt.show()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_sales.index, 
        y=train_sales,
        mode='lines',
        name='Sales'))
    fig.add_trace(go.Scatter(
        x=train_predictions.index, 
        y=train_predictions,
        mode='lines',
        name='Predictions'))
    fig.show()