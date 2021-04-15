from __future__ import print_function
from flask import Flask,request, render_template, send_file, send_from_directory, Response
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
import datetime
from postprocess import write_csv

global folder 
global folder_name

batchid = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
folder_name = "Save_data_{}".format(batchid)
folder = os.mkdir(folder_name)

app = Flask(__name__)
response = Response()

@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    response.headers["Expires"] = '0'
    response.headers["Pragma"] = "no-cache"
    return response

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
            img.save('{}/{}'.format(folder_name,temp_img))  
        
        results = "Files have been uploaded."
        return render_template('index2.html', prediction_text=results) 

@app.route('/results', methods=["GET","POST"])
def results():
        postprocess.write_csv(folder_name,batchid)
        results = "Predictions are ready for Download."
        return render_template('index2.html', result_text=results)
    
@app.route('/return_files/')
def return_files_tut():
    try:

        # The absolute path of the directory containing CSV files for users to download
        path = "C:/Users/kinja/OneDrive/Desktop/All/4.SEM4/DSP/UI/test/static/"
        report_name = "Report{}.csv".format(batchid)
        return send_from_directory(path, report_name, attachment_filename= "Report.csv", as_attachment=True)
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(threaded=True)
