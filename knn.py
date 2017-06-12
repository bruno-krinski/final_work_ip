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
from sklearn.model_selection import train_test_split

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

def knn_validation(data_features,data_labels,out_file):
    out_file = clear_name(out_file)
    output_file_name = "knn_results/knn_validation_" + out_file
    output_file = open(output_file_name,"w+")

    print("Validating...")
    print("Results in ",output_file_name)
    output_file.write("Validation:\n")

    min_max_scaler = preprocessing.MinMaxScaler()
    data_features = min_max_scaler.fit_transform(data_features)

    accuracy_scores = []
    for i in range(10):
        start_time = time.time()
        print("Progress:[",i,"/10]")
        train_f,val_f,train_l,val_l = train_test_split(data_features,
                                      data_labels,test_size=0.4,
                                      random_state=random.randint(1,1000))

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(train_f,train_l)
        r = knn.predict(val_f)
        accuracy_scores.append(printResult(r,val_l,output_file))
        print("--- %s seconds ---" % (time.time() - start_time))
    print("Progress:[10/10]")
    output_file.write("\n\nValidation Results:\n")
    print_list(accuracy_scores,output_file)
    m = sum(accuracy_scores)/10.0
    output_file.write("\n\nMean:"+str(m)+"\n\n")
    output_file.close()

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

def main(argv):
    if len(argv) != 2:
        print("Use mode: python knn.py <mode>")
        print("mode = train,val or test")
        return

    mode = argv[1]

    if mode == "train":
        for d in data:
            data_features, data_labels = readData(d)
            knn_gridSearch(data_features, data_labels,d)
    elif mode == "val":
        for d in data:
            data_features, data_labels = readData(d)
            knn_validation(data_features,data_labels,d)
    elif mode == "test":
        train_file = input("Enter the train file path: ")
        test_file = input("Enter the test file path: ")
        train_features, train_labels = readData(train_file)
        test_features, test_labels = readData(test_file)
        knn_test(train_features,train_labels,test_features,test_labels)
    else:
        print("Unknown mode!")

if __name__ == "__main__":
    main(sys.argv)
