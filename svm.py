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
from sklearn.model_selection import train_test_split

def svm_train(train_features,train_labels,out_file):

    print("Training",out_file)

    output_file_name = "svm_" + out_file
    output_file = open(output_file_name,"w+")

    min_max_scaler = preprocessing.MinMaxScaler()
    train_features = min_max_scaler.fit_transform(train_features)

    C_range = 2. ** np.arange(-8, 9, 2)
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    degrees = [2,3,4]
    gamma_range= 2. ** np.arange(3, -15, -2)
    decision_type = ['ovo', 'ovr', 'None']

    params = {'C': C_range,
              'kernel':kernels,
              'degree':degrees,
              'gamma':gamma_range,
              'decision_function_shape':decision_type}

    svm = GridSearchCV(SVC(),params,n_jobs=-1)

    print("Validating...")
    accuracy_scores = []
    for i in range(10):
        start_time = time.time()
        print("Progress:[",i,"/10]")
        train_f ,val_f,train_l,val_l = train_test_split(train_features,
                                                        train_labels,
                                                        test_size=0.4,
                                                        random_state=random.randint(1, 100))

        svm.fit(train_f, train_l)
        r = svm.predict(val_f)
        accuracy_scores.append(printResult(svm,r,val_l,output_file))
        print("--- %s seconds ---" % (time.time() - start_time))
    print("Results in ",output_file_name)
    output_file.close()

def svm_test():
    print("test")

def main(argv):
    if len(argv) != 2:
        print("Use mode: python svm.py <mode>")
        print("mode = train or mode = test")
        return

    mode = argv[1]

    if mode == "train":
        for d in data:
            data_features, data_labels = readData(d)
            svm_train(data_features, data_labels,d)
    else:
        svm_test()

if __name__ == "__main__":
    main(sys.argv)
