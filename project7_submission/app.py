from flask import Flask
from flask import request
from flask import jsonify
from modules.insurance_predict import InsurancePredict
import pandas as pd

app = Flask(__name__)

@app.route("/")
def home():
    return "API Modeling"

@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.Dataframe(data, index=[0])
    predict_code = InsurancePredict().runModel(df, typed='single')
    
    result_predict = 'interested' if predict_code == 1 else 'Not interested'
    
    return jsonify({
        "status" : "Predicted",
        "predicted_code" : predict_code,
        "result" : result_predict,
        "data" : data
    })