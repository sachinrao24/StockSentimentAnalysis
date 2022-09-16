# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 17:43:09 2022

@author: Sachin
"""
from flask import Flask, request, render_template
import sklearn
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('model/stock_senti.pkl')
vectorizer = joblib.load('model/tfidf_vector.pkl')

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if(request.method=='POST'):
        company = request.form['company_name']
        headline = request.form['headlines']
        # Data Cleaning
        headline = headline.replace('[^a-zA-Z]',' ')
        headline = headline.lower()
        data = vectorizer.transform([headline])
        
        #Prediction
        output = model.predict(data)

        #Output
        if(output==0):
            return render_template('result.html',prediction_text="Stock price of {} may fall".format(company),result = 0)
        else:
            return render_template('result.html',prediction_text="Stock price of {} may rise".format(company),result = 1)
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)
