import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import save_objects
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from dataclasses import dataclass
from sklearn.pipeline import Pipeline


@dataclass
class DataTransformationconfig():
    processor_file_path =os.path.join('artifacts','processor.pkl')

class DataTransformation():
    def __init__ (self):
        self.processor_path=DataTransformationconfig()

    def data_transformation_obj(self,train):
        try:

            train=pd.read_csv(train)
            train=train.drop(columns=['Churn'])

            logging.info("Data read successfully")

            num_columns=train.select_dtypes(exclude='object').columns
            logging.info(f"Numerical columns: {num_columns}")


            cat_columns=train.select_dtypes(include='object').columns
            logging.info(f"Categorical columns: {cat_columns}")
            
            num_pipeline = Pipeline(
                steps=[
                    ('imputers',SimpleImputer(strategy='median')),
                    ('scalar',StandardScaler())
                ] 
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehotencoder',OneHotEncoder(handle_unknown='ignore',sparse_output=False)),
                    # ('scalar',StandardScaler(with_mean=False))
                ]
            )

            logging.info("Pipelines created successfully")

            processor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,num_columns),
                    ('cat_pipeline',cat_pipeline,cat_columns)
                ]
            )

            logging.info("ColumnTransformer created successfully")
            return processor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            
            logging.info("starting initiate data transformation")

            processor=self.data_transformation_obj(train_path)

            logging.info("data_transformation_obj successfully called")

            train = pd.read_csv(train_path)
            train_target = train['Churn']
            train_ = train.drop(columns=['Churn'])


            test = pd.read_csv(test_path)
            test_target = test['Churn']
            test_ = test.drop(columns=['Churn'])
        
            logging.info("Train and test data are read successfully")

            processed_train_data = processor.fit_transform(train_)

            processed_test_data =processor.transform (test_)

            logging.info("Data transformation completed successfully")





            save_objects(
                obj=processor,
                file_path=self.processor_path.processor_file_path 
            )

            logging.info("Processor object saved successfully")

            logging.info (f"shape{processed_train_data.shape}")
            logging.info (f"shape{train_target.shape}")




            train_arr=np.c_[processed_train_data,np.array(train_target)]   ##  np.c_ is a function in NumPy that concatenates arrays along the second axis (columns). In this case, it combines the processed_train_data (which contains the transformed features) with the train_target (which contains the target variable) to create a new array called train_arr. The resulting train_arr will have all the transformed features along with the corresponding target values in a single array.
            test_arr=np.c_[processed_test_data,np.array(test_target)]


            logging.info("Train and test arrays created successfully")

            return (
                train_arr,
                test_arr,
            )
        
        except Exception as e:
            raise CustomException(e,sys)







