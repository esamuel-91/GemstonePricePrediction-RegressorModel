import sys,os
from src.exception import CustomException
from src.logger import logging
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            logging.info("Data Transformation started.")
            # Defining which column should be ordinal encoded and which should be scaled
            categorical_columns = ['cut', 'color', 'clarity']
            numerical_columns = ['carat', 'depth', 'table', 'x', 'y', 'z']

            # Defining custom ranking to ordinal values
            cut_categories = ["Fair","Good","Very Good","Premium","Ideal"]
            color_categories = ["D","E","F","G","H","I","J"]
            clarity_categories = ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]

            # Setting Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("impute",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("impute",SimpleImputer(strategy="most_frequent")),
                    ("ordinal_encoder",OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ("scaler",StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)
            ])
            
            return preprocessor
        
        except Exception as e:
            logging.info("Exception occured in Data Transformation")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and Test data read as pandas dataframe")
            logging.info(f"Train data head: {train_df.head().to_string()}")
            logging.info(f"Test data head: {test_df.head().to_string()}")

            logging.info("Obtaining Preprocessing object")

            preprocessing_obj = self.get_data_transformation_obj()

            #Seperating independent and dependent columns
            target_column = "price"
            drop_column = [target_column,"price"]

            input_feature_train_df = train_df.drop(drop_column,axis=1)
            input_target_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(drop_column,axis=1)
            input_target_test_df = test_df[target_column]

            #Transforming using preprocessing object
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("Applying preprocessing obj to train and test dataset")

            train_arr = np.c_[input_feature_train_arr,np.array(input_target_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(input_target_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info("Preprocess pickle file saved")

            return(
                test_arr,
                train_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Exception occured during Data Transformation")
            raise CustomException(e,sys)