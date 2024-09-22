import os,sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    model_obj_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Splitting dependedant and independent variables from train and test data")
            X_train,X_test,y_train,y_test = (
                train_array[:,:-1],
                test_array[:,:-1],
                train_array[:,-1],
                test_array[:,-1]
            )

            models = {
                "Linear Regression":LinearRegression(),
                "Ridge":Ridge(),
                "Lasso":Lasso(),
                "Elastic Net":ElasticNet()
            }
            
            model_report:dict = evaluate_model(X_train,X_test,y_train,y_test,models)
            print(model_report)
            print("\n----------------------------------------------------------------------------------------\n")
            logging.info(f"Model Report: {model_report}")

            # To get the best score form the dictionary

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best model found: Model name -> {best_model_name} and ,model_score is -> {best_model_score}")
            print("\n----------------------------------------------------------------------------------------\n")
            logging.info(f"Best model found: Model name -> {best_model_name} and ,model_score is -> {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.model_obj_file_path,
                obj = best_model
            )

        except Exception as e:
            logging.info("Exception occured while initiating model trainer")
            raise CustomException(e,sys)
