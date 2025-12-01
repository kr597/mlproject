import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models


@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            # FIXED SPLITTING
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "KNN Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            # ADD PARAMS
            param = {
                "Random Forest": {
                    "n_estimators": [50, 100],
                    "max_depth": [5, 10]
                },
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse"],
                    "max_depth": [5, 10]
                },
                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [50, 100]
                },
                "Linear Regression": {},
                "KNN Regressor": {
                    "n_neighbors": [3, 5, 7]
                },
                "XGB Regressor": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [50, 100]
                },
                "CatBoost Regressor": {
                    "depth": [6, 8],
                    "learning_rate": [0.01, 0.1]
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [50, 100]
                }
            }

            # FIXED FUNCTION CALL
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=param
            )

            # GET BEST MODEL
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info("Best model found and saved")

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
