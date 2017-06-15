#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import random
import numpy as np
from resources import *
from image import Image
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

def knn_gridSearch(data_features,data_labels,out_file):

    out_file = clear_name(out_file)
    output_file_name = "knn_results/knn_gridsearch_" + out_file
    output_file = open(output_file_name,"w+")

    print("Grid Searching and Validating of",out_file)
    print("Results in ",output_file_name)

    min_max_scaler = preprocessing.MinMaxScaler()
    data_features = min_max_scaler.fit_transform(data_features)

    train_f,val_f,train_l,val_l = train_test_split(data_features,
                                                   data_labels,
                                                   test_size=0.5,
                                                   random_state=0)
    k = [1,3,5]
    params = {'n_neighbors':k}
    print("Making Grid Search...")
    knn = GridSearchCV(KNeighborsClassifier(),params,n_jobs=-1,cv=5)
    knn.fit(train_f,train_l)
    printGridSearchResult(knn,output_file)
    output_file.close()

def knn_validation(train_f,val_f,train_l,val_l,output_file):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_f,train_l)
    r = knn.predict(val_f)
    return printResult(r,val_l,output_file)

def knn_test(train_features,train_labels,test_features,test_labels):
    start_time = time.time()

    print("Testing...")

    min_max_scaler = preprocessing.MinMaxScaler()
    train_features = min_max_scaler.fit_transform(train_features)
    test_features = min_max_scaler.transform(test_features)

    output_file_name = "knn_results/knn_test.txt"
    output_file = open(output_file_name,"w+")

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_features,train_labels)
    r = knn.predict(test_features)

    printResult(r,test_labels,output_file)

    print("Results in ",output_file_name)

    output_file.close()

    print("--- %s seconds ---" % (time.time() - start_time))
