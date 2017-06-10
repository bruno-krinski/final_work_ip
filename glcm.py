#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
from resources import *
from image import Image
from skimage.feature import greycomatrix, greycoprops

def glcm(image, distances, angles, levels, method):
    my_glcm = greycomatrix(image,distances,angles,levels=levels,normed=True,
                                                                symmetric=True)

    return greycoprops(my_glcm, method)

def main(argv):
    if len(argv) != 4:
        print("Use mode: python glcm.py <file with images name> <images path> <out file>")
        return

    in_file = argv[1]
    out_file = argv[3]
    images_path = argv[2]
    method = ['contrast','dissimilarity','homogeneity','energy','correlation',
              'ASM']

    distances = [[1],[1,2],[1,2,3],[1,2,3,4]]

    i = 1
    images = readImages(in_file)
    for dists in distances:
        output_file_name = "glcm_" + str(i) + "_" + out_file
        output_file = open(output_file_name, "w+")
        for image in images:
            img = cv2.imread(images_path + "/" + image.name, 0)
            tot = []
            for m in method:
                image.features = glcm(img,dists,[0,np.pi/4,np.pi/2,3*np.pi/4],256,m)
                aux = []
                for f in image.features:
                    aux = np.concatenate((aux,f),axis=0)
                tot = np.concatenate((tot,aux),axis=0)
            #print(len(tot))
            writeFeatures(image.label, tot, output_file)
        output_file.close()
        i += 1

if __name__ == "__main__":
    main(sys.argv)
