import os
import sys
from src.utils import Helpers
from src.logger import logging
from xgboost import XGBRegressor
from dataclasses import dataclass
from sklearn.metrics import r2_score
from src.configs import model_params
from catboost import CatBoostRegressor
from src.exception import CustomException
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor


@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer(Helpers):

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            x_train, y_train, x_test, y_test = (
                train_arr[:, : -1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                'Random Forest Regressor': RandomForestRegressor(),
                'K Neighbors Regressor': KNeighborsRegressor(),
                'Decision Tree Regressor': DecisionTreeRegressor(),
                'Ada Boost Regressor': AdaBoostRegressor(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'XGBRegressor': XGBRegressor(),
                'Cat Boost Regressor': CatBoostRegressor(verbose=False)
            }

            report = self.model_evaluation(x_train, y_train, x_test, y_test, models, model_params)

            best_score = max(report.values())
            if best_score < 0.6:
                raise CustomException("No best model found", sys)

            best_model_name = list(report.keys())[list(report.values()).index(best_score)]

            best_model = models[best_model_name]

            self.save_object(self.model_trainer_config.trained_model_path, best_model)

            predicted = best_model.predict(x_test)

            return r2_score(y_test, predicted)
        except Exception as e:
            raise CustomException(e, sys)
