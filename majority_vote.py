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
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

def vote_gridSearch(data_features,data_labels,out_file):

    out_file = clear_name(out_file)
    output_file_name = "majority_vote_results/majority_vote_" + out_file
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
    activation = ['identity', 'logistic', 'tanh', 'relu']
    solver = ['lbfgs', 'sgd', 'adam']
    learning_rate = ['constant', 'invscaling', 'adaptive']
    C_range = 2. ** np.arange(-8, 9, 2)
    kernels = ['linear', 'poly', 'rbf']
    degrees = [2,3,4]
    gamma_range= 2. ** np.arange(3, -15, -2)
    decision_type = ['ovo', 'ovr']

    params = {'knc__n_neighbors':k,
              'mlp__activation':activation,
              'mlp__solver':solver,
              'mlp__learning_rate':learning_rate,
              'svc__C':C_range,
              'svc__kernel':kernels,
              'svc__degree':degrees,
              'svc__gamma':gamma_range,
              'svc__decision_function_shape':decision_type}

    knn = KNeighborsClassifier()
    mlp = MLPClassifier(max_iter=10000)
    svm = SVC()

    vote = VotingClassifier(estimators=[('knc', knn),
                                        ('mlp', mlp),
                                        ('svc', svm)],voting='hard')
    print("Making Grid Search...")
    grid = GridSearchCV(estimator=vote,param_grid=params,n_jobs=16,cv=5)
    grid.fit(train_f,train_l)
    printGridSearchResult(grid,output_file)
    output_file.close()

def vote_validation(train_f,val_f,train_l,val_l,output_file):
    knn = KNeighborsClassifier(n_neighbors=1)
    mlp = MLPClassifier(max_iter=10000,
                        activation='relu',
                        solver='lbfgs',
                        learning_rate='adaptive')
    svm = SVC(C=4.0,
              decision_function_shape='ovo',
              degree=2,
              gamma=8.0,
              kernel='poly')

    vote = VotingClassifier(estimators=[('knc', knn),
                                        ('mlp', mlp),
                                        ('svc', svm)],
                            n_jobs=16,
                            voting='hard')
    vote.fit(train_f,train_l)
    r = vote.predict(val_f)
    return printResult(r,val_l,output_file)

def vote_test(train_features,train_labels,test_features,test_labels):
    start_time = time.time()

    print("Testing...")

    min_max_scaler = preprocessing.MinMaxScaler()
    train_features = min_max_scaler.fit_transform(train_features)
    test_features = min_max_scaler.transform(test_features)

    output_file_name = "majority_vote_results/majority_vote_test.txt"
    output_file = open(output_file_name,"w+")

    knn = KNeighborsClassifier(n_neighbors=1)
    mlp = MLPClassifier(max_iter=10000)
    svm = SVC(C=4.0,
              decision_function_shape='ovo',
              degree=2,
              gamma=8.0,
              kernel='poly')

    vote = VotingClassifier(estimators=[('knc', knn),
                                        ('mlp', mlp),
                                        ('svc', svm)],
                            n_jobs=16,
                            voting='hard')

    vote.fit(train_features,train_labels)
    r = vote.predict(test_features)

    printResult(r,test_labels,output_file)

    print("Results in ",output_file_name)

    output_file.close()

    print("--- %s seconds ---" % (time.time() - start_time))
