from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import sys

from sklearn.preprocessing import StandardScaler

from src.pipelines.predict_pipeline import CustomData, PredictPipeline
from src.utils import load_object
from src.exception import CustomException
import pickle

# class CustomData:
#     def __init__(self, bedrooms: int, bathrooms: float, location: str):
#         self.bedrooms = bedrooms
#         self.bathrooms = bathrooms
#         self.location = location

#     def get_data_as_dataframe(self):
#         try:
#             custom_data_input_dict = {
#                 'Bedrooms':[self.bedrooms],
#                 'Bathrooms': [self.bathrooms],
#                 'Location': [self.location]
#             }

#             return pd.DataFrame(custom_data_input_dict)
        
#         except Exception as e:
#             raise CustomException(e,sys)

# class PredictPipeline:
#     def __init__(self):
#         pass

#     def predict(self, features):
#         try:
#             # model_path = 'artifact\model.pkl'
#             # preprocessor_path = 'artifact\preprocessor.pkl'
#             with open('artifact\preprocessor.pkl', 'rb') as preprocessor_obj:
#                 preprocessor = pickle.load(preprocessor_obj)
#             with open('artifact\model.pkl', 'rb') as model_obj:
#                 model = pickle.load(model_obj)
#             # preprocessor = load_object(file_path = preprocessor_path)
#             data_scaled = preprocessor.transform(features)
#             prediction = model.predict(data_scaled)
#             return prediction
        
#         except Exception as e:
#             raise CustomException(e,sys)

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
    app.run(host='0.0.0.0', debug=True)
        