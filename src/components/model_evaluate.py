#  logging.info("Best model saved successfully")

#             predicted=best_model.prdict(x_test)
#             classification_rep=classification_report(y_test,predicted)

#             logging.info("Classification report generated successfully")

#             return classification_report

from sklearn.metrics import classification_report,accuracy_score,f1_score,precision_score,recall_score

from src.logger import logging
from src.utils import load_objects


class modelEvaluate():
    def initiate_model_evaluate(self,train_arr,test_arr,model_path):
        x_train,y_train,x_test,y_test = (
            train_arr[:,:-1],
            train_arr[:,-1],
            test_arr[:,:-1],
            test_arr[:,-1]
        )
        
        

        model = load_objects(model_path)

        pred_value = model.predict(x_test)

        accuracy= accuracy_score(y_test,pred_value)
        precision= precision_score(y_test,pred_value)
        f1= f1_score(y_test,pred_value)
        recall= recall_score(y_test,pred_value)
        logging.info(f"THE BEST MODEL IS    {model}   accuacy={accuracy}\nprecision={precision}\nf1={f1}\nrecall={recall}"   )   



        logging.info ("all done ")



        return (
            accuracy,
            precision,
            f1,
            recall,
        )

if __name__ == "__main__":
    logging.info("Logging has started") 





