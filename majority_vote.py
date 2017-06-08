#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import random
import numpy as np
from resources import *
from sklearn import svm
from image import Image
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def voting_train(train_features, train_labels, test_features, test_labels,output_file):
    C_range = 2. ** np.arange(-8, 15, 2)
    gamma_range = 2. ** np.arange(3, -15, -2)

    parameters = {'kernel': }

    clf1 = KNeighborsClassifier()
    clf2 = svm.SVC()
    clf3 = MLPClassifier()
    eclf = VotingClassifier(estimators=[('knc', clf1), ('svm', clf2), ('mlp', clf3)], voting='soft')
    params = {'knc__n_neighbors':[1,3,5,7,9],
              'svm__kernel':('linear','rbf'),
              'svm__gamma':gamma_range,
              'svm__C':C_range,
              'mlp__activation': ('identity', 'logistic', 'tanh', 'relu'),
              'mlp__solver': ('lbfgs', 'sgd', 'adam'),
              'mlp__learning_rate': ('constant', 'invscaling', 'adaptive')}
    grid = GridSearchCV(estimator=eclf, param_grid=params).fit(train_features, train_labels)
    output_file.write(str(eclf.best_params_)+"\n")
    result = eclf.predict(test_features)
    printResult(test_labels, result, output_file)

#def knn_test(train_features, train_labels, test_features, test_labels,output_file):
def voting_test():
    print("test")

def main(argv):
    if len(argv) != 2:
        print("Use mode: python majority_vote.py <mode>")
        print("mode = train or mode = test")
        return

    mode = argv[1]
    for d in data:
        data_features, data_labels = readData(d)
        min_max_scaler = preprocessing.MinMaxScaler()
        data_features = min_max_scaler.fit_transform(data_features)
        output_file_name = "voting_" + d
        output_file = open(output_file_name,"w+")
        if mode == "train":
            for i in range(0, 10):
                rand = random.randint(1, 100)
                train_f ,test_f,train_l,test_l = train_test_split(data_features,
                                                                  data_labels,
                                                                  test_size=0.4,
                                                                  random_state=rand)
                voting_train(train_f, train_l, test_f, test_l,output_file)
                output_file.write("=========================================\n")
        else:
            voting_test()
            #knn_test(train_f, train_l, test_f, test_l,output_file)
        output_file.close()

if __name__ == "__main__":
    main(sys.argv)
