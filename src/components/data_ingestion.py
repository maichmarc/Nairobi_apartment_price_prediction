import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.utils import location_fill
from src.utils import remove_duplicates_null
from src.utils import keep_locations
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig


@dataclass
class DataIngestionConfig:
    # train_data_path: str = os.path.join('artifact', 'train.csv')
    # test_data_path: str = os.path.join('artifact', 'test.csv')
    raw_data_path: str = os.path.join('artifact', 'raw.csv')
    prep_data_path: str = os.path.join('artifact', 'prep.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered data ingestion component')
        try:
            df=pd.read_csv('notebooks\data\house_prices.csv')
            locations = pd.read_csv('notebooks\data\locations.csv')
            logging.info('Read the dataset as dataframe')
            data = remove_duplicates_null(df)
            data = location_fill(data,locations,'Location','Location')
            data = keep_locations(data, 'Location', 4)
            # print(data.size)
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            logging.info('Cleaning the data initiated i.e. remove duplicates and missing data')
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            data.to_csv(self.ingestion_config.prep_data_path, index=False, header=True)
            # logging.info('Train test split initiated')
            # train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
            # train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            # test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Ingestion of the data is complete')
            return(
                self.ingestion_config.raw_data_path,
                self.ingestion_config.prep_data_path,
                
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_data, test_data,_= data_transformation.get_data_transformer_obj()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))

    


