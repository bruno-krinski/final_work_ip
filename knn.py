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

    print("Grid Searching and Validating of",out_file)

    output_file_name = "knn_" + out_file
    output_file = open(output_file_name,"w+")

    print("Results in ",output_file_name)

    min_max_scaler = preprocessing.MinMaxScaler()
    data_features = min_max_scaler.fit_transform(data_features)

    train_f,val_f,train_l,val_l = train_test_split(data_features,
                                                   data_labels,
                                                   test_size=0.5,
                                                   random_state=0)
    k = [1,3,5,7,9]
    params = {'n_neighbors':k}
    print("Making Grid Search...")
    knn = GridSearchCV(KNeighborsClassifier(),params,n_jobs=-1,cv=5)
    knn.fit(train_f,train_l)
    printGridSearchResult(knn,output_file)
    knn_validation(data_features,data_labels,knn,output_file)

def knn_validation(data_features,data_labels,clf,output_file):
    print("Validating...")
    output_file.write("\n\nValidation:\n")
    accuracy_scores = []
    for i in range(10):
        start_time = time.time()
        print("Progress:[",i,"/10]")
        train_f,val_f,train_l,val_l = train_test_split(data_features,
                                                       data_labels,
                                                       test_size=0.4,
                                                       random_state=random.randint(1, 100))
        knn = KNeighborsClassifier(**clf.best_params_).fit(train_f,train_l)
        r = knn.predict(val_f)
        accuracy_scores.append(printResult(knn,r,val_l,output_file))
        print("--- %s seconds ---" % (time.time() - start_time))
    print("Progress:[10/10]")
    output_file.write("\n\nValidation Results:\n")
    print_list(accuracy_scores,output_file)
    m = sum(accuracy_scores)/10.0
    output_file.write("\n\nMean:"+str(m)+"\n\n")
    output_file.close()

def knn_test():
    print("test")

def main(argv):
    if len(argv) != 2:
        print("Use mode: python knn.py <mode>")
        print("mode = train or mode = test")
        return

    mode = argv[1]

    if mode == "train":
        for d in data:
            data_features, data_labels = readData(d)
            knn_gridSearch(data_features, data_labels,d)
    else:
        knn_test()

if __name__ == "__main__":
    main(sys.argv)
