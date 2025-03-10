from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import sys

from sklearn.preprocessing import StandardScaler

from src.pipelines.predict_pipeline import CustomData, PredictPipeline
from src.utils import load_object
from src.exception import CustomException
import pickle

application = Flask(__name__)

app = application

## Route for homepage

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])


def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            bedrooms= request.form.get('bedrooms'),
            bathrooms= float(request.form.get('bathrooms')),
            location= request.form.get('location')
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', bedrooms = str(pred_df['Bedrooms']), results = results[0])
    

if __name__ == '__main__':
    app.run(host='0.0.0.0')
        