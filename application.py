from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for a home page
@application.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Fetching data from the HTML form and matching your Telco Churn features
        data = CustomData(
    gender=request.form.get('gender', ''),
    SeniorCitizen=int(request.form.get('SeniorCitizen') or 0),
    Partner=request.form.get('Partner', ''),
    Dependents=request.form.get('Dependents', ''),
    tenure=int(request.form.get('tenure') or 0),
    PhoneService=request.form.get('PhoneService', ''),
    MultipleLines=request.form.get('MultipleLines', ''),
    InternetService=request.form.get('InternetService', ''),
    OnlineSecurity=request.form.get('OnlineSecurity', ''),
    OnlineBackup=request.form.get('OnlineBackup', ''),
    DeviceProtection=request.form.get('DeviceProtection', ''),
    TechSupport=request.form.get('TechSupport', ''),
    StreamingTV=request.form.get('StreamingTV', ''),
    StreamingMovies=request.form.get('StreamingMovies', ''),
    Contract=request.form.get('Contract', ''),
    PaperlessBilling=request.form.get('PaperlessBilling', ''),
    PaymentMethod=request.form.get('PaymentMethod', ''),
    MonthlyCharges=float(request.form.get('MonthlyCharges') or 0.0),
    TotalCharges=float(request.form.get('TotalCharges') or 0.0)
)
        
        # Converting the gathered form data into a DataFrame format
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")
        
        # Executing the prediction pipeline
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        
        # Sending the prediction result back to your home.html template
        # results[0] will typically yield a binary value (e.g., 1 for Churn, 0 for Retain)
        status = "Likely to Churn" if results[0] == 1 else "Likely to Stay"
        
        return render_template('home.html', results=status)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8000)