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
from sklearn.ensemble import BaggingClassifier

def bagging_svm_validation(train_f,val_f,train_l,val_l,output_file):

    bagging = BaggingClassifier(SVC(C=4.0,
                                    decision_function_shape='ovo',
                                    degree=2,
                                    gamma=8.0,
                                    kernel='poly'),
                                max_samples=1.0,
                                max_features=1.0)
    bagging.fit(train_f,train_l)
    r = bagging.predict(val_f)
    return printResult(r,val_l,output_file)

def bagging_svm_test(train_features,train_labels,test_features,test_labels):
    start_time = time.time()

    print("Testing...")

    min_max_scaler = preprocessing.MinMaxScaler()
    train_features = min_max_scaler.fit_transform(train_features)
    test_features = min_max_scaler.transform(test_features)

    output_file_name = "bagging_svm_results/bagging_svm_test.txt"
    output_file = open(output_file_name,"w+")

    bagging = BaggingClassifier(SVC(C=4.0,
                                    decision_function_shape='ovo',
                                    degree=2,
                                    gamma=8.0,
                                    kernel='poly'),
                                max_samples=1.0,
                                max_features=1.0)
    bagging.fit(train_features,train_labels)
    r = bagging.predict(test_features)

    printResult(r,test_labels,output_file)

    print("Results in ",output_file_name)

    output_file.close()

    print("--- %s seconds ---" % (time.time() - start_time))
