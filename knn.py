#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from resources import *
from image import Image
from skimage import feature
from sklearn.neighbors import KNeighborsClassifier as knc

def knn_function(train_features, train_labels, test_features, test_labels,data_file):
    ks = [1, 3, 5]
    for k in ks:
        output_file_name = "knn_" + str(k) + "_" + data_file
        output_file = open(output_file_name,"w+")
        knn = knc(n_neighbors=k)
        knn.fit(train_features, train_labels)
        result = knn.predict(test_features)
        output_file.write("k = ",k)
        printResult(test_labels, result, output_file)
        output_file.close()

def main(argv):
    for d in data:
        data_features, data_labels = readData(d)
        min_max_scaler = preprocessing.MinMaxScaler()
        data_features = min_max_scaler.fit_transform(data_features)
        for i in range(0, 10):
            rand = random.randint(1, 100)
            train_f ,test_f,train_l,test_l = train_test_split(data_features,
                                                              data_labels,
                                                              test_size=0.4,
                                                              random_state=rand)
            knn(train_f, train_l, test_f, test_l,d)

if __name__ == "__main__":
    main(sys.argv)
