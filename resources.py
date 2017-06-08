#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from image import Image

data = ['lbp_default_train.txt',
        'lbp_ror_train.txt',
        'lbp_uniform_train.txt',
        'lbp_nri_uniform_train.txt',
        'glcm_1_train.txt',
        'glcm_2_train.txt',
        'glcm_3_train.txt',
        'glcm_4_train.txt']

def printResult(labels, result, output_file):
    output_file.write("Confusion Matrix: ")
    cm = confusion_matrix(labels,result)
    for l in cm:
        np.savetxt(output_file, l, fmt='%-7.2f')
    output_file.write("Accuracy: ",accuracy_score(labels,result))

def readData(file_name):
    data = open(file_name, "r")
    data_features = []
    data_labels = []
    while True:
        aux = data.readline.split()
        if len(aux) == 0:
            data.close()
            return data_features, data_labels
        else:
            data_labels.append(aux[len(aux)-1])
            data_features.append(aux[0:len(aux)-1])

def readImages(file_name):
    images_file = open(file_name, "r")
    images = []
    while True:
        aux = images_file.readline().split()
        if len(aux) == 0:
            images_file.close()
            return images
        else:
            images.append(Image(aux[0], aux[1]))

def writeFeatures(label, features, output_file):
    s = ""
    for f in features:
        s += str(f) + " "
    s += label+"\n"
    output_file.write(s)
