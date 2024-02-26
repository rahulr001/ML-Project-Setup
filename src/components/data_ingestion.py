import os
import sys
import pandas as pd
from src.logger import logging
from dataclasses import dataclass
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train_data.csv')
    test_data_path: str = os.path.join('artifacts', 'test_data.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Initiated Data Ingestion')
        try:
            df = pd.read_csv('notebook/data/stud.csv')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(os.path.join(self.ingestion_config.raw_data_path), index=False)

            train_set, test_set = train_test_split(df, random_state=40, test_size=.2)

            train_set.to_csv(os.path.join(self.ingestion_config.train_data_path), index=False)
            test_set.to_csv(os.path.join(self.ingestion_config.test_data_path), index=False)

            logging.info('Data Ingestion completed successfully')
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info('Exception occured during Data Ingestion')
            raise CustomException(e, sys)


if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    train_arr, test_arr, _ = DataTransformation().initiate_data_transformation(train_data, test_data, 'math_score')
    ModelTrainer().initiate_model_trainer(train_arr, test_arr)

