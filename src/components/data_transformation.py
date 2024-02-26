import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataclasses import dataclass
from src.utils import Helpers


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation(Helpers):

    def __init__(self):
        self.preprocessor_config = DataTransformationConfig()

    @staticmethod
    def get_preprocessor_obj(target_column):
        try:
            raw_df = pd.read_csv(os.path.join('artifacts', 'raw_data.csv'))

            num_features = [feature for feature in raw_df.columns if raw_df[feature].dtype != 'O']
            cat_features = [feature for feature in raw_df.columns if raw_df[feature].dtype == 'O']

            num_features.remove(target_column)

            num_pipline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])

            cat_pipline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ])

            preprocessor = ColumnTransformer(
                [
                    ('num_pipline', num_pipline, num_features),
                    ('cat_pipline', cat_pipline, cat_features)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path, target_column):
        try:
            logging.info('Data Transformation Started')
            preprocessor = self.get_preprocessor_obj(target_column)

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            self.save_object(self.preprocessor_config.preprocessor_obj_file_path, preprocessor)

            logging.info('Data Transformation Started')
            return train_arr, test_arr, self.preprocessor_config.preprocessor_obj_file_path,
        except Exception as e:
            raise CustomException(e, sys)
