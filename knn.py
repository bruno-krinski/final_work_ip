#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import random
import numpy as np
from resources import *
from image import Image
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

def knn_train(train_features, train_labels):

    k = [1,3,5,7,9]

    params = {'n_neighbors':k}

    rand = random.randint(1, 100)
    cv = StratifiedShuffleSplit(n_splits=10,test_size=0.4,random_state=rand)

    knn = GridSearchCV(KNeighborsClassifier(),params,n_jobs=-1,cv=cv).fit(train_features, train_labels)

    print("The best parameters are %s with a score of %0.2f"%(knn.best_params_, knn.best_score_))

    #output_file.write(str(clf.best_params_)+"\n")
    #result = knn.predict(test_features)
    #printResult(test_labels, result, output_file)

#def knn_test(train_features, train_labels, test_features, test_labels,output_file):
def knn_test():
    print("test")

def main(argv):
    if len(argv) != 2:
        print("Use mode: python knn.py <mode>")
        print("mode = train or mode = test")
        return

    mode = argv[1]
    for d in data:
        data_features, data_labels = readData(d)
        min_max_scaler = preprocessing.MinMaxScaler()
        data_features = min_max_scaler.fit_transform(data_features)
        #output_file_name = "knn_" + d
        #output_file = open(output_file_name,"w+")
        if mode == "train":
            #for i in range(0, 10):
            #    rand = random.randint(1, 100)
            #    train_f ,test_f,train_l,test_l = train_test_split(data_features,
            #                                                      data_labels,
            #                                                      test_size=0.4,
            #                                                      random_state=rand)
            knn_train(data_features, data_labels)
            #    output_file.write("=========================================\n")
        else:
            knn_test()
            #knn_test(train_f, train_l, test_f, test_l,output_file)
        #output_file.close()

if __name__ == "__main__":
    main(sys.argv)
