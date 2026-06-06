import os  
import sys 
import pandas as pd 
import numpy as np 



from src.logger import logging 
from src.exception import CustomException 
from dataclasses import dataclass 
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionCConfig():
    train_data_path:str= os.path.join('artifacts','train.csv')
    test_data_path:str= os.path.join('artifacts','test.csv')
    raw_data_path:str= os.path.join('artifacts','data.csv') 
    

class DataIngestion():
    def __init__(self):
        self.ingestion_config=DataIngestionCConfig()

    def initiate_data_ingestion(self):
        try:
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            csv_path = os.path.join(BASE_DIR, 'notebook', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')


            df=pd.read_csv(csv_path)


            df['Churn'] = df['Churn'].map({
            'No': 0,
            'Yes': 1
                })
            
            
            logging.info("Dataset read as pandas dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)


            df=df.iloc[:,1:]
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train test split initiated")
            Train,Test = train_test_split(df,random_state=42,test_size=0.2) 
            logging.info("Train test split completed")
            Train.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            Test.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    obj1 = DataIngestion()
    train,test=obj1.initiate_data_ingestion()