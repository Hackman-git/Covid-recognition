from __future__ import print_function
from flask import Flask,request, render_template
import pandas as pd
import pickle
import numpy as np
import keras
import os
import cv2
import keras
import tensorflow as tf
import io
from PIL import Image
import datetime
import sys
import csv
import postprocess


app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict', methods=["GET","POST"])
def predict():
    if request.method=="POST":
        image_ip = request.files.getlist("file")

        #Pre-processing steps
        for img in image_ip:
            temp_img = img.filename
            img = Image.open(img)
            img = img.convert('L')  # greyscale
            img.save('Saved_test_data/{}'.format(temp_img))   

        postprocess.write_csv()
        
        results = "Predictions are ready for Download file"
        return render_template('index2.html', prediction_text=results)
    
if __name__ == "__main__":
    app.run(debug=True)
