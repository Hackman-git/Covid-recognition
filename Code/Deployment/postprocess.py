from flask import Flask,request, render_template
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import cv2
import keras
import tensorflow as tf
import datetime
import csv
import glob

def write_csv():
    
    IMAGE_SIZE = (224,224)
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    def get_targets(file):
        split = tf.strings.split(file, os.path.sep)
        if split[-2] == 'COVID':
            return 0
        elif split[-2] == 'Viral Pneumonia':
            return 2
        else: return 1

    def preprocess_img(file):
        target = get_targets(file)
        img = tf.io.read_file(file)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, IMAGE_SIZE)
        return img, target, file

    sample_new = tf.data.Dataset.list_files("Saved_test_data/*") 
    BATCH_SIZE = 32
    new_data = sample_new.map(preprocess_img)
    new_data = new_data.batch(BATCH_SIZE)
    sharp_model = keras.models.load_model('covid_model_deep.h5')
    predictions = sharp_model.predict(new_data)

    #Class for each prediction.
    prediction_class = np.argmax(predictions, axis=1)
    prediction_class = prediction_class.tolist()  

    
    #Write results in csv
    write_to_csv = [['Filename', 'Classifier', 'Accuracy level of class detected']]
    for files in glob.glob("Saved_test_data/*"):
        sample_new = tf.data.Dataset.list_files(files)
        BATCH_SIZE = 32
        new_data = sample_new.map(preprocess_img)
        new_data = new_data.batch(BATCH_SIZE)
        sharp_model = keras.models.load_model('covid_model_deep.h5')
        predictions = sharp_model.predict(new_data)
        prediction_class = np.argmax(predictions, axis=1)
        prediction_class = prediction_class.tolist()[0]
        file_name = files.strip("'").split("Saved_test_data\\")[1]

        classifier = None
        if prediction_class == 1:
            classifier = "Normal"
        elif prediction_class == 2:
            classifier = "Viral Pneumonia"
        else:
            classifier = "COVID"

        accuracy_level = predictions[0][int(prediction_class)]
        write_to_csv.append([file_name, classifier, accuracy_level])
        #print(file_name, classifier, predictions, prediction_class, accuracy_level)

    filename = 'static/Report.csv'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(write_to_csv)

