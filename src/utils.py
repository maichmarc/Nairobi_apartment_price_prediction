import os
import sys

import numpy as np 
import pandas as pd
import pickle
import math

from src.exception import CustomException

from sklearn.metrics import r2_score

def location_fill(df1, df2, col_a, col_b):
    location_list = list(df2[col_b])
    list_location = [
        next((item1 for item1 in location_list if item1 in item2), item2) 
        for item2 in list(df1[col_a])
    ]
    df3 = df1.copy()
    df3[col_b] = list_location
    return df3

def remove_duplicates_null(df):
    df1 = df.drop_duplicates()
    df2 = df1.dropna()
    return df2

def remove_outliers(df):
    q1_bath = np.percentile(df['Bathrooms'], 25)
    q2_bath = np.percentile(df['Bathrooms'], 50)
    q3_bath = np.percentile(df['Bathrooms'], 75)
    iqr_bath = q3_bath - q1_bath
    lower_bath_limit = q1_bath - (1.5*iqr_bath)
    upper_bath_limit = q3_bath + (1.5*iqr_bath)
    
    q1_price = np.percentile(df['Price'], 25)
    q2_price = np.percentile(df['Price'], 50)
    q3_price = np.percentile(df['Price'], 75)
    iqr_price = q3_price - q1_price
    lower_price_limit = q1_price - (3*iqr_price)
    upper_price_limit = q3_price + (3*iqr_price)
    
    q1_bed = np.percentile(df['Bedrooms'], 25)
    q2_bed = np.percentile(df['Bedrooms'], 50)
    q3_bed = np.percentile(df['Bedrooms'], 75)
    iqr_bed = q3_bed - q1_bed
    lower_bed_limit = q1_bed - (1.5*iqr_bed)
    upper_bed_limit = q3_bed + (1.5*iqr_bed)
    
    df1 = df.drop(df[df['Price']>upper_price_limit].index)
    df2 = df1.drop(df1[df1['Bedrooms']>math.ceil(upper_bed_limit)].index)
    df3 = df2.drop(df2[df2['Bathrooms']>math.ceil(upper_bath_limit)].index)
    
    return df3

def keep_locations(df, col, n ):
    location_counts = df[col].value_counts()
    locations_to_keep = location_counts[location_counts >= n].index
    data = df[df[col].isin(locations_to_keep)]
    return data

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)