import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import modelTrainer
from src.components.model_evaluate import modelEvaluate
from src.logger import logging
from src.exception import CustomException

class model_pipeline:
    def start_model_tranning(self):
        try:
            Ingestion=DataIngestion()
            train,test = Ingestion.initiate_data_ingestion()

            transformation = DataTransformation()
            logging.info("trian_pipeline")
            train_arr,test_arr = transformation.initiate_data_transformation(train,test)  

            # modelTrain = modelTrainer()
            # model_path = modelTrain.initiate_model_trainer(train_arr,test_arr)      

            model_eval= modelEvaluate()
            accuracy,precision,f1,recall= model_eval.initiate_model_evaluate(train_arr,test_arr,_) 

            


        except Exception as e :
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj = model_pipeline()
    obj.start_model_tranning()

