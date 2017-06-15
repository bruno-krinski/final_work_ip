#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import random
import numpy as np
from resources import *
from image import Image
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

def mlp_gridSearch(data_features,data_labels,out_file):

    out_file = clear_name(out_file)
    output_file_name = "mlp_results/mlp_" + out_file
    output_file = open(output_file_name,"w+")

    print("Grid Searching and Validating of",out_file)
    print("Results in ",output_file_name)

    min_max_scaler = preprocessing.MinMaxScaler()
    data_features = min_max_scaler.fit_transform(data_features)

    train_f,val_f,train_l,val_l = train_test_split(data_features,
                                                   data_labels,
                                                   test_size=0.5,
                                                   random_state=0)

    activation = ['identity', 'logistic', 'tanh', 'relu']
    solver = ['lbfgs', 'sgd', 'adam']
    learning_rate = ['constant', 'invscaling', 'adaptive']

    params = {'activation':activation,
              'solver':solver,
              'learning_rate':learning_rate}

    print("Making Grid Search...")
    mlp = GridSearchCV(MLPClassifier(max_iter=10000),params,n_jobs=16,cv=5)
    mlp.fit(train_f,train_l)
    printGridSearchResult(mlp,output_file)
    output_file.close()

def mlp_validation(train_f,val_f,train_l,val_l,output_file):
    mlp = MLPClassifier(max_iter=10000,
                        activation='relu',
                        solver='lbfgs',
                        learning_rate='adaptive')
    mlp.fit(train_f, train_l)
    r = mlp.predict(val_f)
    return printResult(r,val_l,output_file)

def mlp_test(train_features,train_labels,test_features,test_labels):
    start_time = time.time()

    print("Testing...")

    min_max_scaler = preprocessing.MinMaxScaler()
    train_features = min_max_scaler.fit_transform(train_features)
    test_features = min_max_scaler.transform(test_features)

    output_file_name = "mlp_results/mlp_test.txt"
    output_file = open(output_file_name,"w+")

    mlp = MLPClassifier(max_iter=10000,
                        activation='relu',
                        solver='lbfgs',
                        learning_rate='adaptive')
    mlp.fit(train_features,train_labels)
    r = mlp.predict(test_features)

    printResult(r,test_labels,output_file)

    print("Results in ",output_file_name)

    output_file.close()

    print("--- %s seconds ---" % (time.time() - start_time))
