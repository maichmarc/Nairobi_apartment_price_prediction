import streamlit as st
import numpy as np
import pandas as pd

from src.pipelines.predict_pipeline import CustomData, PredictPipeline

st.title('Nairobi Apartment Price Prediction')

st.write("This web app uses data scraped from 'https://www.buyrentkenya.com/' to create a Machine Learning algorithm to predict " \
         "the price of an apartment in Nairobi based on its number of bedrooms, bathrooms and the location." )


bedrooms = st.selectbox(
    "Number of bedrooms",
    (1, 2, 3, 4, 5),
    index=None,
    placeholder="Select number of bedrooms...",
)

bathrooms = st.selectbox(
    "Number of bathrooms",
    (1, 2, 3, 4, 5),
    index=None,
    placeholder="Select number of bathrooms...",
)

location = st.selectbox(
    "Location",
    ('Westlands', 'Kilimani', 'Kileleshwa', 'Lavington', 'Riverside', 'Parklands', 'Mombasa Road', 'Lower Kabete', 'Thika', 'Ngong Road', 
     'Dagoretti', 'Spring Valley', 'Kitisuru', 'Embakasi', 'Kiambu', 'Riara'),
    index=None,
    placeholder="Select location...",
)

button = st.button("Predict")
if bedrooms and bathrooms and location and button:
    data = CustomData(
                bedrooms= bedrooms,
                bathrooms= bathrooms,
                location= location
            )
    pred_df = data.get_data_as_dataframe()
    # print(pred_df)

    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    st.write(f"The predicted price for a {pred_df['Bedrooms'][0]} bedroom, {int(pred_df['Bathrooms'][0])} bathroom apartment in {pred_df['Location'][0]} is Kshs. {results[0]:,.2f}")