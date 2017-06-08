#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import random
import numpy as np
from resources import *
from sklearn import svm
from image import Image
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def svm_train(train_features,train_labels,test_features,test_labels,output_file):

    C_range = 2. ** np.arange(-8, 15, 2)
    gamma_range = 2. ** np.arange(3, -15, -2)

    parameters = {'kernel': ('linear', 'rbf'),'gamma':gamma_range,'C': C_range}
    svr = svm.SVC()
    clf = GridSearchCV(svr, parameters,n_jobs=4).fit(train_features, train_labels)

    output_file.write(str(clf.best_params_)+"\n")
    result = clf.predict(test_features)
    printResult(test_labels, result, output_file)

#def svm_test(train_features,train_labels,test_features,test_labels,output_file):
def svm_test():
    print("test")

def main(argv):
    if len(argv) != 2:
        print("Use mode: python svm.py <mode>")
        print("mode = train or mode = test")
        return

    mode = argv[1]
    for d in data:
        data_features, data_labels = readData(d)
        min_max_scaler = preprocessing.MinMaxScaler()
        data_features = min_max_scaler.fit_transform(data_features)
        output_file_name = "svm_" + d
        output_file = open(output_file_name,"w+")
        if mode == "train":
            for i in range(0, 10):
                rand = random.randint(1, 100)
                train_f ,test_f,train_l,test_l = train_test_split(data_features,
                                                                  data_labels,
                                                                  test_size=0.4,
                                                                  random_state=rand)
                svm_train(train_f, train_l, test_f, test_l,output_file)
                output_file.write("=========================================\n")
        else:
            #svm_test(train_f, train_l, test_f, test_l,output_file)
            svm_test()
        output_file.close()

if __name__ == "__main__":
    main(sys.argv)
