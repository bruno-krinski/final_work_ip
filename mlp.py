#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import random
import numpy as np
import progressbar
from resources import *
from image import Image
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def mlp_train(train_features,train_labels,out_file):

    print("Training",out_file)

    output_file_name = "mlp_" + out_file
    output_file = open(output_file_name,"w+")

    min_max_scaler = preprocessing.MinMaxScaler()
    train_features = min_max_scaler.fit_transform(train_features)

    activation = ['identity', 'logistic', 'tanh', 'relu']
    solver = ['lbfgs', 'sgd', 'adam']
    learning_rate = ['constant', 'invscaling', 'adaptive']

    params = {'activation':activation,
              'solver':solver,
              'learning_rate':learning_rate}

    mlp = GridSearchCV(MLPClassifier(max_iter=10000),params,n_jobs=-1)

    print("Validating...")
    accuracy_scores = []
    for i in range(10):
        start_time = time.time()
        print("Progress:[",i,"/10]")
        train_f ,val_f,train_l,val_l = train_test_split(train_features,
                                                        train_labels,
                                                        test_size=0.4,
                                                        random_state=random.randint(1, 100))

        mlp.fit(train_f, train_l)
        r = mlp.predict(val_f)
        accuracy_scores.append(printResult(mlp,r,val_l,output_file))
        print("--- %s seconds ---" % (time.time() - start_time))
    print("Results in ",output_file_name)
    output_file.close()

def mlp_test():
    print("test")

def main(argv):
    if len(argv) != 2:
        print("Use mode: python mlp.py <mode>")
        print("mode = train or mode = test")
        return

    mode = argv[1]

    if mode == "train":
        #bar = progressbar.ProgressBar()
        for d in data:
            data_features, data_labels = readData(d)
            mlp_train(data_features,data_labels,d)
    else:
        mlp_test()

if __name__ == "__main__":
    main(sys.argv)
