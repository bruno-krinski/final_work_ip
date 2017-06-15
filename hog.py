#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
from resources import *
from image import Image
from skimage import feature

def hog(image):
    my_hog = feature.hog(image,orientations=8,cells_per_block=(1,1),pixels_per_cell=(8,8),block_norm='L2-Hys')
    return my_hog

def main(argv):
    if len(argv) != 4:
        print("Use mode: python hog.py <file with images name> <images path> <out file>")
        return
    in_file = argv[1]
    out_file = argv[3]
    images_path = argv[2]
    images = readImages(in_file)
    output_file_name = 'hog_files/hog_' + out_file
    output_file = open(output_file_name, "w+")
    for image in images:
        img = cv2.imread(images_path + "/" + image.name, 0)
        image.features = hog(img)
        writeFeatures(image.label, image.features, output_file)
    output_file.close()

if __name__ == "__main__":
    main(sys.argv)
