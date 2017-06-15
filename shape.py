#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
from resources import *
from image import Image
from skimage import feature

def shape(image, radius, num_points, method):
    my_shape = feature.local_binary_pattern(image,num_points,radius,method=method)
    n_bins = int(my_shape.max() + 1)
    hist,_ = np.histogram(my_shape.ravel(),normed=True,bins = n_bins,range=(0,n_bins))
    return hist

def main(argv):
    if len(argv) != 4:
        print("Use mode: python shape.py <file with images name> <images path> <out file>")
        return
    in_file = argv[1]
    out_file = argv[3]
    images_path = argv[2]
    method = ['constant', 'reflect', 'wrap', 'nearest', 'mirror']
    images = readImages(in_file)
    for m in method:
        output_file_name = 'shape_files/shape_' + m + '_' + out_file
        output_file = open(output_file_name, "w+")
        for image in images:
            img = cv2.imread(images_path + "/" + image.name, 0)
            image.features = shape(img,1,8,m)
            writeFeatures(image.label, image.features, output_file)
        output_file.close()

if __name__ == "__main__":
    main(sys.argv)
