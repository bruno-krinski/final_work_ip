#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import random
import numpy as np
from resources import *
from image import Image
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

def svm_gridSearch(data_features,data_labels,out_file):

    out_file = clear_name(out_file)
    output_file_name = "svm_results/svm_" + out_file
    output_file = open(output_file_name,"w+")

    print("Grid Searching and Validating of",out_file)
    print("Results in ",output_file_name)

    min_max_scaler = preprocessing.MinMaxScaler()
    data_features = min_max_scaler.fit_transform(data_features)

    train_f,val_f,train_l,val_l = train_test_split(data_features,
                                                   data_labels,
                                                   test_size=0.5,
                                                   random_state=0)
    C_range = 2. ** np.arange(-8, 9, 2)
    kernels = ['linear', 'poly', 'rbf']
    degrees = [2,3,4]
    gamma_range= 2. ** np.arange(3, -15, -2)
    decision_type = ['ovo', 'ovr']

    params = {'C': C_range,
              'kernel':kernels,
              'degree':degrees,
              'gamma':gamma_range,
              'decision_function_shape':decision_type}

    print("Making Grid Search...")
    svm = GridSearchCV(SVC(),params,n_jobs=4,cv=5)
    svm.fit(train_f,train_l)
    printGridSearchResult(svm,output_file)
    output_file.close()

def svm_validation(train_f,val_f,train_l,val_l,output_file):

    output_file.write("Validation:\n")

    svm = SVC(C=4.0,
              decision_function_shape='ovo',
              degree=2,
              gamma=8.0,
              kernel='poly')
    svm.fit(train_f,train_l)
    r = svm.predict(val_f)
    return printResult(r,val_l,output_file)

def svm_test(train_features,train_labels,test_features,test_labels):
    start_time = time.time()

    print("Testing...")

    min_max_scaler = preprocessing.MinMaxScaler()
    train_features = min_max_scaler.fit_transform(train_features)
    test_features = min_max_scaler.transform(test_features)

    output_file_name = "svm_results/svm_test.txt"
    output_file = open(output_file_name,"w+")

    svm = SVC(C=4.0,
              decision_function_shape='ovo',
              degree=2,
              gamma=8.0,
              kernel='poly')
    svm.fit(train_features,train_labels)
    r = svm.predict(test_features)

    printResult(r,test_labels,output_file)

    print("Results in ",output_file_name)

    output_file.close()

    print("--- %s seconds ---" % (time.time() - start_time))
