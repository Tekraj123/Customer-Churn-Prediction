import os 
import sys
from src.pipeline.predict_pipeline import CustomData
from src.pipeline.predict_pipeline import predictionPipeline
from src.exception import CustomException
from flask import Flask,render_template,request


app = Flask(__name__) 
@app.route("/")
def index():
    return render_template('home.html')

@app.route('/predictor',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('index.html')
    else:
          data=CustomData( 
                request.form.get('gender'),
                request.form.get('SeniorCitizen'),
                request.form.get("Partner"),
                request.form.get("Dependents"),
                request.form.get("tenure"),
                request.form.get("PhoneService"),
                request.form.get("MultipleLines"),
                request.form.get("InternetService"),
                request.form.get("OnlineSecurity"),
                request.form.get("OnlineBackup"),
                request.form.get("DeviceProtection"),
                request.form.get("TechSupport"),
                request.form.get("StreamingTV"),
                request.form.get("StreamingMovies"),
                request.form.get("Contract"),
                request.form.get("PaperlessBilling"),
                request.form.get("PaymentMethod"),
                request.form.get("MonthlyCharges"),
                request.form.get("TotalCharges"),
                )
          




    data_frame=data.get_dataframe()
    pred_pipeline = predictionPipeline()
    pred_value =  pred_pipeline.predict_(data_frame)
    

    return render_template ("predict.html",result=pred_value)
        


        
if __name__ == "__main__":
    app.run(host="0.0.0.0")  

        








