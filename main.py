#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import random
from knn import *
from svm import *
from mlp import *
import numpy as np
from boosting import *
from resources import *
from image import Image
from bagging_knn import *
from bagging_mlp import *
from bagging_svm import *
from majority_vote import *
from sklearn.model_selection import train_test_split

def main(argv):
    if len(argv) != 2:
        print("Use mode: python knn.py <mode>")
        print("mode = train,val or test")
        return

    mode = argv[1]

    if mode == "train":
        for d in data:
            data_features, data_labels = readData(d)
            knn_gridSearch(data_features, data_labels,d)
            svm_gridSearch(data_features, data_labels,d)
            mlp_gridSearch(data_features, data_labels,d)
    elif mode == "val":
        end_file = []
        for d in data:
            data_features, data_labels = readData(d)
            min_max_scaler = preprocessing.MinMaxScaler()
            data_features = min_max_scaler.fit_transform(data_features)
            accuracy_scores_knn = []
            accuracy_scores_mlp = []
            accuracy_scores_svm = []
            accuracy_scores_vote = []
            accuracy_scores_bagging_knn = []
            accuracy_scores_bagging_mlp = []
            accuracy_scores_bagging_svm = []
            accuracy_scores_boosting = []
            accuracy_scores = []
            out_file = clear_name(d)
            print("Validating",out_file)
            output_file_name = "knn_results/knn_validation_" + out_file
            output_file_knn = open(output_file_name,"w+")

            output_file_name = "mlp_results/mlp_validation_" + out_file
            output_file_mlp = open(output_file_name,"w+")

            output_file_name = "svm_results/svm_validation_" + out_file
            output_file_svm = open(output_file_name,"w+")

            output_file_name = "majority_vote_results/majority_vote_validation_" + out_file
            output_file_vote = open(output_file_name,"w+")

            output_file_name = "bagging_knn_results/bagging_knn_validation_" + out_file
            output_file_bagging_knn = open(output_file_name,"w+")

            output_file_name = "bagging_mlp_results/bagging_mlp_validation_" + out_file
            output_file_bagging_mlp = open(output_file_name,"w+")

            output_file_name = "bagging_svm_results/bagging_svm_validation_" + out_file
            output_file_bagging_svm = open(output_file_name,"w+")

            output_file_name = "boosting_results/boosting_validation_" + out_file
            output_file_boosting = open(output_file_name,"w+")
            #------------------------------------------------------------------------------------------
            output_file_dat_name = "knn_results/knn_validation_" + out_file + ".dat"
            output_file_dat_knn = open(output_file_dat_name,"w+")

            output_file_dat_name = "mlp_results/mlp_validation_" + out_file + ".dat"
            output_file_dat_mlp = open(output_file_dat_name,"w+")

            output_file_dat_name = "svm_results/svm_validation_" + out_file + ".dat"
            output_file_dat_svm = open(output_file_dat_name,"w+")

            output_file_dat_name = "majority_vote_results/majority_vote_validation_" + out_file + ".dat"
            output_file_dat_vote = open(output_file_dat_name,"w+")

            output_file_dat_name = "bagging_knn_results/bagging_knn_validation_" + out_file + ".dat"
            output_file_dat_bagging_knn = open(output_file_dat_name,"w+")

            output_file_dat_name = "bagging_mlp_results/bagging_mlp_validation_" + out_file + ".dat"
            output_file_dat_bagging_mlp = open(output_file_dat_name,"w+")

            output_file_dat_name = "bagging_svm_results/bagging_svm_validation_" + out_file + ".dat"
            output_file_dat_bagging_svm = open(output_file_dat_name,"w+")

            output_file_dat_name = "boosting_results/boosting_validation_" + out_file + ".dat"
            output_file_dat_boosting = open(output_file_dat_name,"w+")

            out_files = []
            out_files.append(output_file_knn)
            out_files.append(output_file_mlp)
            out_files.append(output_file_svm)
            out_files.append(output_file_vote)
            out_files.append(output_file_bagging_knn)
            out_files.append(output_file_bagging_mlp)
            out_files.append(output_file_bagging_svm)
            out_files.append(output_file_boosting)

            out_dat_files = []
            out_dat_files.append(output_file_dat_knn)
            out_dat_files.append(output_file_dat_mlp)
            out_dat_files.append(output_file_dat_svm)
            out_dat_files.append(output_file_dat_vote)
            out_dat_files.append(output_file_dat_bagging_knn)
            out_dat_files.append(output_file_dat_bagging_mlp)
            out_dat_files.append(output_file_dat_bagging_svm)
            out_dat_files.append(output_file_dat_boosting)

            for i in range(10):
                print("Progress:[",i,"/10]")
                train_f,val_f,train_l,val_l = train_test_split(data_features,
                                              data_labels,test_size=0.4,
                                              random_state=random.randint(1,10000))
                accuracy_scores_knn.append(knn_validation(train_f,val_f,train_l,val_l,output_file_knn))
                accuracy_scores_mlp.append(mlp_validation(train_f,val_f,train_l,val_l,output_file_mlp))
                accuracy_scores_svm.append(svm_validation(train_f,val_f,train_l,val_l,output_file_svm))
                accuracy_scores_vote.append(vote_validation(train_f,val_f,train_l,val_l,output_file_vote))
                accuracy_scores_bagging_knn.append(bagging_knn_validation(train_f,val_f,train_l,val_l,output_file_vote))
                accuracy_scores_bagging_mlp.append(bagging_mlp_validation(train_f,val_f,train_l,val_l,output_file_vote))
                accuracy_scores_bagging_svm.append(bagging_svm_validation(train_f,val_f,train_l,val_l,output_file_vote))
                accuracy_scores_boosting.append(boosting_validation(train_f,val_f,train_l,val_l,output_file_vote))
            print("Progress:[10/10]")
            accuracy_scores.append(accuracy_scores_knn)
            accuracy_scores.append(accuracy_scores_mlp)
            accuracy_scores.append(accuracy_scores_svm)
            accuracy_scores.append(accuracy_scores_vote)
            accuracy_scores.append(accuracy_scores_bagging_knn)
            accuracy_scores.append(accuracy_scores_bagging_mlp)
            accuracy_scores.append(accuracy_scores_bagging_svm)
            accuracy_scores.append(accuracy_scores_boosting)

            end_file.append(accuracy_scores_knn)
            end_file.append(accuracy_scores_mlp)
            end_file.append(accuracy_scores_svm)
            end_file.append(accuracy_scores_vote)
            end_file.append(accuracy_scores_bagging_knn)
            end_file.append(accuracy_scores_bagging_mlp)
            end_file.append(accuracy_scores_bagging_svm)
            end_file.append(accuracy_scores_boosting)

            for i in range(8):
                write_mean(accuracy_scores[i],out_files[i])
                write_dat_files(accuracy_scores[i],out_dat_files[i])
                out_dat_files[i].close()
                out_files[i].close()
            #end_file.append(accuracy_scores)
        print_wilcoxon(end_file)
    elif mode == "test":
        train_file = input("Enter the train file path: ")
        test_file = input("Enter the test file path: ")
        train_features, train_labels = readData(train_file)
        test_features, test_labels = readData(test_file)
        knn_test(train_features,train_labels,test_features,test_labels)
        mlp_test(train_features,train_labels,test_features,test_labels)
        svm_test(train_features,train_labels,test_features,test_labels)
        vote_test(train_features,train_labels,test_features,test_labels)
    else:
        print("Unknown mode!")

if __name__ == "__main__":
    main(sys.argv)
