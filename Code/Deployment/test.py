from flask import Flask,request, render_template
import pandas as pd
import pickle
import numpy as np
from math import exp
import keras


app = Flask(__name__)
path = "C:/Users/kinja/OneDrive/Desktop/All/4.SEM4/Data Science Practicuum/UI/test"
model = keras.models.load_model('C:/Users/kinja/OneDrive/Desktop/All/4.SEM4/Data Science Practicuum/UI/test/ecovid_model_new.h5')

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict', methods=["GET","POST"])
def predict():
    if request.method=="POST":
        image_ip = request.form["img"]
        result=model.predict(image_ip)
        print(result)
        return render_template('index2.html', prediction_text='The patient is {}'.format(result))
    	
if __name__ == "__main__":
    app.run(debug=True)