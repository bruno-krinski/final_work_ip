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
from sklearn.ensemble import AdaBoostClassifier

def boosting_validation(train_f,val_f,train_l,val_l,output_file):

        boosting = AdaBoostClassifier(base_estimator=SVC(C=4.0,
                                        decision_function_shape='ovo',
                                        degree=2,
                                        gamma=8.0,
                                        kernel='poly'),
                                     algorithm='SAMME',
                                     n_estimators=100)

        boosting.fit(train_f,train_l)
        r = boosting.predict(val_f)
        return printResult(r,val_l,output_file)

def boosting_test(train_features,train_labels,test_features,test_labels):
    start_time = time.time()

    print("Testing...")

    min_max_scaler = preprocessing.MinMaxScaler()
    train_features = min_max_scaler.fit_transform(train_features)
    test_features = min_max_scaler.transform(test_features)

    output_file_name = "boosting_results/boosting_test.txt"
    output_file = open(output_file_name,"w+")

    boosting = AdaBoostClassifier(base_estimator=SVC(C=4.0,
                                    decision_function_shape='ovo',
                                    degree=2,
                                    gamma=8.0,
                                    kernel='poly'),
                                  algorithm='SAMME',
                                  n_estimators=100)

    boosting.fit(train_features,train_labels)
    r = boosting.predict(test_features)

    printResult(r,test_labels,output_file)

    print("Results in ",output_file_name)

    output_file.close()

    print("--- %s seconds ---" % (time.time() - start_time))
