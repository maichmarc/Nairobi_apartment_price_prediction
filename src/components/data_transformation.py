import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from src.utils import remove_outliers
from src.utils import save_object
from scipy.sparse import csr_matrix, hstack

@dataclass
class DataTransformationConfig():
    train_data_path: str = os.path.join('artifact', 'train.csv')
    test_data_path: str = os.path.join('artifact', 'test.csv')
    preprocessor_obj_file_path = os.path.join('artifact', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            # numerical_features = ['Bedrooms','Bathrooms','Price']
            numerical_features = ['Bedrooms','Bathrooms']
            categorical_features = ['Location']

            data=pd.read_csv('artifact\prep.csv')
            data = remove_outliers(data)
            print(data.size)
            logging.info('Train test split initiated')
            train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
            train_set.to_csv(self.data_transformation_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.data_transformation_config.test_data_path, index=False, header=True)

            numerical_pipeline = Pipeline(
                steps=[('Scaler',StandardScaler())]
            )

            categorical_pipeline = Pipeline(
                steps=[('OneHotEncoder', OneHotEncoder())]
            )

            # logging.info('Numerical columns standard scaling completed')
            # logging.info('Categorical columns standard encoding completed')

            preprocessor = ColumnTransformer(
                [
                ('Categorical Pipeline', categorical_pipeline, categorical_features),
                ('Numerical Pipeline', numerical_pipeline, numerical_features)
                ]
            )
            return (
                self.data_transformation_config.train_data_path,
                self.data_transformation_config.test_data_path,
                preprocessor
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading train and test data completed.')
            logging.info('Obtaining preprocessing object.')

            _,_,preprocessor_obj = self.get_data_transformer_obj()

            target_column_name = 'Price'
            # numerical_features = ['Bedrooms','Bathrooms','Price']
            # categorical_features = ['Location']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(target_column_name, axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info('Applying preprocessor object on training and testing dataframe')

            # print(input_feature_train_df.columns)
            # print(input_feature_test_df.columns)
            # print(target_feature_train_df.name)
            # print(target_feature_test_df.name)

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            # print(input_feature_train_arr.shape)
            # print(input_feature_test_arr.shape)
            # print(input_feature_test_arr)
            input_feature_train_arr_sparce = csr_matrix(input_feature_train_arr)
            input_feature_test_arr_sparce = csr_matrix(input_feature_test_arr)

            # print(np.array(target_feature_train_df).shape)
            # print(np.array(target_feature_test_df).shape)
            

            target_feature_train_df_array = np.array(target_feature_train_df).reshape(-1,1)
            target_feature_test_df_array = np.array(target_feature_test_df).reshape(-1,1)


            # print(target_feature_train_df_array.shape)
            # print(target_feature_test_df_array.shape)
            # print(target_feature_test_df_array)

            train_arr = hstack([
                input_feature_train_arr_sparce,target_feature_train_df_array 
            ])

            test_arr = hstack([
                input_feature_test_arr_sparce, target_feature_test_df_array
            ])

            # print(type(train_arr))
            # print(type(test_arr))

            train_arr = train_arr.toarray()
            test_arr = test_arr.toarray()


            logging.info('Saved preprocessing object.')

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj

            )

            # print(type(train_arr))
            # print(type(test_arr))

            # print(test_arr)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)



    
