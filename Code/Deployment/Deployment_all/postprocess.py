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

def write_csv():
    
    IMAGE_SIZE = (224,224)
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    def get_targets(file):
        split = tf.strings.split(file, os.path.sep)
        if split[-2] == 'COVID 19':
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
    sharp_model = keras.models.load_model('ecovid_model_new.h5')
    predictions = sharp_model.predict(new_data)

    #Write output:
    for l in list(new_data):
        ls_numpy = str(l[2].numpy())

    ls_new = ls_numpy.strip(" [] ").strip("\n").split("b'Saved_test_data\\\\")
    file_ls = [l.strip("' \n") for l in ls_new][1:]

    pred_class = np.argmax(predictions, axis=1)
    pred_class = pred_class.tolist()

    #batchid = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    #filename = '{}.csv'.format(batchid)

    filename = 'static/Report.csv'
    write_to_csv = []

    for file_name,pred,pclass in zip(file_ls, predictions,pred_class):
        if pclass == 1:
            classifier = "Normal"
        elif pclass == 2:
            classifier = "Viral Pneumonia"
        else:
            classifier = "COVID"
        write_to_csv.append([file_name, classifier, pred[pclass]])

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(write_to_csv)


