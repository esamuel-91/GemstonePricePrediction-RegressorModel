import os,sys
from src.logger import logging
from src.exception import CustomException
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_object:
            pickle.dump(obj=obj,file=file_object)

    except Exception as e:
        logging.info("Exception occured while saving the object")
        raise CustomException(e,sys)
    
def evaluate_model(X_train,X_test,y_train,y_test,models):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X_train,y_train)

            # making predictions
            y_test_pred = model.predict(X_test)

            test_model_score = r2_score(y_test,y_test_pred)
            report[list(models.keys())[i]] = test_model_score

            return report

    except Exception as e:
        logging.info("Exception occured while evaluating models")
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_object:
            return pickle.load(file_object)
    except Exception as e:
        logging.info("Exception occured while loading the object")
        raise CustomException(e,sys)