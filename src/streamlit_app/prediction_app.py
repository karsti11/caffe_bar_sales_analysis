import shap
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from catboost import CatBoost, Pool
import matplotlib.pyplot as plt
import plotly.express as px
import datetime
from src.features.build_features import (MetadataTransformer,
                                         CalendarTransformer, 
                                         HolidaysTransformer)
from sklearn.pipeline import Pipeline
from src.data.make_dataset import load_dataset

LAST_PRICES_FN = "last_prices.pkl"
MODEL_FN = "model.cbm"


class PredictionsMaker:

    def __init__(self):
        self.last_prices_df = pd.read_pickle(LAST_PRICES_FN)
        self.model = CatBoost()
        self.model.load_model(MODEL_FN)

    def create_predict_data(self, from_date, to_date, items_list):
        X_df = pd.DataFrame(pd.date_range(from_date, to_date, freq='D'))
        X_df.columns = ['sales_date']
        prices_df = self.last_prices_df[self.last_prices_df.item_name.isin(items_list)]
        X_df = X_df.merge(prices_df, how='cross')
        X_df = X_df.set_index('sales_date')
        return X_df

    def make_predictions(self, X_df):
        
        pipeline = Pipeline(steps=[
                           ('metadata_tf', MetadataTransformer()),
                           ('calendar_tf', CalendarTransformer()),
                           ('holidays_tf', HolidaysTransformer())
                            ])
        self.dataset_w_feats = pipeline.fit_transform(X_df)
        preds = pd.Series(self.model.predict(Pool(self.dataset_w_feats, cat_features=['item_name'])), index=X_df.index)
        preds.name = 'prediction'
        preds = pd.concat([X_df['item_name'], preds], axis=1)
        return preds#.groupby('item_name')['prediction'].sum().reset_index()

    """def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)"""

class PredictionsExplainer:

    def __init__(self, model, X_df):
        self.model = model
        self.X_df = X_df

    def explain_preds(self):
        explainercat = shap.TreeExplainer(self.model)
        shap_values = explainercat(self.X_df)
        fig = plt.figure(figsize=(8,16))
        shap.plots.beeswarm(shap_values, max_display=20)
        st.pyplot(fig)

    def plot_shap_waterfall(self, item_name, selected_date):
        X_exp = self.X_df[(self.X_df.item_name == item_name) & (self.X_df.index == str(selected_date))]
        print(selected_date)
        print(X_exp)
        explainercat = shap.TreeExplainer(self.model)
        shap_values = explainercat(X_exp)
        fig = plt.figure(figsize=(8,16))
        shap.plots.waterfall(shap_values[0])
        st.pyplot(fig)

    def plotly_visualize_preds(self, predictions_df, item_name):
        preds_visualize = predictions_df[predictions_df.item_name == item_name]
        fig = px.line(preds_visualize, y=['prediction'])
        st.plotly_chart(fig, use_container_width=True)


# Instantiate model and last valid prices within PredictionsMaker
predictions_maker = PredictionsMaker()
#Title
st.title('Caffe bar sales forecast')
# Select dates header
st.subheader('Select dates for sales forecast.')
# Input date range for predictions
date_from = st.date_input("From date:", datetime.date(2020, 12, 1))
date_to = st.date_input("To date:", datetime.date(2021, 3, 1))
#st.write(f"Generation of predictions from {date_from} to {date_to}")
# Select items header
st.subheader('Select items for sales forecast.')
all_items = predictions_maker.last_prices_df.item_name.values.tolist()
default_items = ['Kava s mlijekom velika', 'Kava', 'Kava s mlijekom', 'Cedevita',
       'Nescaffe', 'Coca Cola', 'Mineralna voda', 'Caj', 'Emotion',
       'Niksicko pivo', 'Ozujsko pivo', 'Beck`s',
       'Jana vitaminski napitak']
items_list = st.multiselect('Which items you want predictions for?', all_items, default=default_items)
if items_list:
    data_load_state = st.text('Getting predictions...')
    X_df = predictions_maker.create_predict_data(date_from, date_to, items_list)
    predictions_df = predictions_maker.make_predictions(X_df)
    st.write(predictions_df.groupby('item_name')['prediction'].sum().reset_index())
    data_load_state.text('Predictions retrieved.')
    predictions_expl = PredictionsExplainer(predictions_maker.model, 
                                        predictions_maker.dataset_w_feats)
    # Predictions visual and explanations header
    st.subheader('Explore your predictions.')
    # Plot predictions signal
    visual_preds = st.checkbox('I want predictions visualized.')
    if visual_preds:
        option = st.selectbox('Select item to visualize predictions.', items_list)
        if option:
            predictions_expl.plotly_visualize_preds(predictions_df, option)
    explain = st.checkbox('I want predictions explained.')
    if explain:
        predictions_expl.explain_preds()
    waterfall = st.checkbox('I want predictions explained daily per item.')
    if waterfall and len(items_list) > 1:
        selected_item = st.select_slider('Select item for prediction waterfall graph',
                    options=items_list)
        selected_date = st.slider(
         "Select date for prediction waterfall graph",
         min_value=date_from,
         max_value=date_to,
         value=date_from,
         format="DD-MM-YYYY")
        #if selected_item and selected_date:
        predictions_expl.plot_shap_waterfall(selected_item, selected_date)
    elif len(items_list) <= 1:
        st.write("At least two items have to be selected to generate slider.")



    # Plot shap waterfall graph with slider for selection of date

