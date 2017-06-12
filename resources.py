#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from image import Image
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve

data = ['lbp_files/lbp_default_training.txt',
        'lbp_files/lbp_ror_training.txt',
        'lbp_files/lbp_uniform_training.txt',
        'lbp_files/lbp_nri_uniform_training.txt',
        'glcm_files/glcm_1_training.txt',
        'glcm_files/glcm_2_training.txt',
        'glcm_files/glcm_3_training.txt',
        'glcm_files/glcm_4_training.txt']

def print_list(my_list,output_file):
    for l in my_list:
        output_file.write(str(l)+" ")

def printGridSearchResult(clf,output_file):
    output_file.write("Grid scores on development set:\n\n")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        output_file.write("%0.3f (+/-%0.03f) for %r\n"%(mean, std * 2, params))

    output_file.write("\n\nBest parameters set found on development set:\n")
    output_file.write(str(clf.best_params_))

    output_file.write("\n\nBest score set found on development set:\n")
    output_file.write(str(clf.best_score_))

def printResult(result,labels,output_file):
    output_file.write("Confusion Matrix: \n")
    cm = confusion_matrix(labels,result)
    for cols in cm:
        r = ""
        for rows in cols:
            if rows < 10:
                r += "   " + str(rows)
            elif rows < 100:
                r += "  " + str(rows)
            elif rows < 1000:
                r += " " + str(rows)
        r += "\n"
        output_file.write(r)

    accuracy = accuracy_score(labels,result)
    output_file.write("\nAccuracy: " + str(accuracy) + "\n")
    output_file.write("\n-------------------------------------------------\n\n")
    return accuracy

def readData(file_name):
    data = open(file_name, "r")
    data_features = []
    data_labels = []
    while True:
        aux = data.readline().split()
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

def clear_name(s):
    if s[0] == 'l':
        return s[10:len(s)]
    elif s[0] == 'g':
        return s[11:len(s)]
    else:
        print("Unknown string!")
        exit(0)
