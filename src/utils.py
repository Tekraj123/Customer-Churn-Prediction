import os
import pandas as pd
import numpy as np
import sys
import pickle
from src.exception import CustomException
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV

from src.logger import logging

# os.path.dirname(file_path) → extracts only the directory part of that path, removing the actual file name.

def save_objects(obj,file_path):  
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)


def model_evaluates(x_train,y_train,x_test,y_test,models,params):
    try:
        report={}
        matrices={}
        for name,model in models.items():
            parameters = params[name]
            grid = GridSearchCV(
                estimator = model,
                param_grid = parameters,
                cv = 5,
                # Notice the '_macro' or '_micro' suffixes added below
                scoring = 'accuracy',

                # refit = 'recall_macro'
                n_jobs=-1 ,  # 👈 uses ALL CPU cores

            )

            grid.fit(x_train,y_train)

            report[name] = {
                'best_params': grid.best_estimator_,   #best model with its parameters
                'best_score': grid.best_score_,
            }
            
            name=classification_report(y_test, grid.predict(x_test))
            logging.info(f"...............Classification report   \n\n\n for {model}\n{name} \n\n\n")


        
        return report



            



        # for i in range(len(models)):
        #     model=list(models.values())[i]  
        #     model.fit(x_train,y_train)       
        #     prd_value = model.predict (x_test)   
        #     accuracy=accuracy_score(y_test,prd_value)        
        #     report[list(models.keys())[i]]=accuracy

     

        # return report

    except Exception as e :
        raise CustomException(e,sys)
    



def load_objects(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e :
        raise CustomException (e,sys)
        
    