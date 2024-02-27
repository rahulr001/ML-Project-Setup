import os
import sys
import pickle
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


class Helpers:

    @staticmethod
    def save_object(file_path, obj):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as file_obj:
                pickle.dump(obj, file_obj)
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def load_object(file_path):
        try:
            with open(file_path, 'rb') as file_obj:
                return pickle.load(file_obj)
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def model_evaluation(x_train, y_train, x_test, y_test, models, params):
        try:
            report = dict()
            for model_name, model in models.items():
                gs = GridSearchCV(model, params.get(model_name), cv=5)
                gs.fit(x_train, y_train)

                model.set_params(**gs.best_params_)
                model.fit(x_train, y_train)

                test_prediction = model.predict(x_test)
                score = r2_score(y_test, test_prediction)

                report[model_name] = score

            return report
        except Exception as e:
            raise CustomException(e, sys)
