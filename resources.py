#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from image import Image
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve

data = ['lbp_default_train.txt',
        'lbp_ror_train.txt',
        'lbp_uniform_train.txt',
        'lbp_nri_uniform_train.txt',
        'glcm_1_train.txt',
        'glcm_2_train.txt',
        'glcm_3_train.txt',
        'glcm_4_train.txt']

#data = ['lbp_default_train.txt']

def print_list(my_list,output_file):
    for l in my_list:
        output_file.write(str(l)+" ")

def printGridSearchResult(clf,output_file):
    output_file.write("Grid scores on development set:\n\n")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        output_file.write("%0.3f (+/-%0.03f) for %r\n"%(mean, std * 2, params))

    output_file.write("\n\nBest parameters set found on development set:\n")
    output_file.write(str(clf.best_params_))

    output_file.write("\n\nBest score set found on development set:\n")
    output_file.write(str(clf.best_score_))

def printResult(clf,result,labels,output_file):

    output_file.write("\nConfusion Matrix: \n")
    cm = confusion_matrix(labels,result)
    for cols in cm:
        r = ""
        for rows in cols:
            if rows < 10:
                r += "   " + str(rows)
            elif rows < 100:
                r += "  " + str(rows)
            elif rows < 1000:
                r += " " + str(rows)
        r += "\n"
        output_file.write(r)

    accuracy = accuracy_score(labels,result)
    output_file.write("\nAccuracy: " + str(accuracy) + "\n")
    output_file.write("\n-----------------------------------------------------")
    return accuracy

def readData(file_name):
    data = open(file_name, "r")
    data_features = []
    data_labels = []
    while True:
        aux = data.readline().split()
        if len(aux) == 0:
            data.close()
            return data_features, data_labels
        else:
            data_labels.append(aux[len(aux)-1])
            data_features.append(aux[0:len(aux)-1])

def readImages(file_name):
    images_file = open(file_name, "r")
    images = []
    while True:
        aux = images_file.readline().split()
        if len(aux) == 0:
            images_file.close()
            return images
        else:
            images.append(Image(aux[0], aux[1]))

def writeFeatures(label, features, output_file):
    s = ""
    for f in features:
        s += str(f) + " "
    s += label+"\n"
    output_file.write(s)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
