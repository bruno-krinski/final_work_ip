#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import random

labels = ['Apuleia',
          'Aspidosperma',
          'Astronium',
          'Byrsonima',
          'Calophyllum',
          'Cecropia',
          'Cedrelinga',
          'Cochlospermum',
          'Combretum',
          'Copaifera']

def generateResourceFile(n, m, name):
    resource = []
    for label in labels:
        for i in range(0, n):
            for j in range(0, m):
                if j < 10:
                    image_name = label + "_00" + str(i) + "_00" + str(j) + ".tif"
                else:
                    image_name = label + "_00" + str(i) + "_0" + str(j) + ".tif"
                resource.append(image_name + " " + label + "\n")
    random.shuffle(resource)
    resource_file = open(name, "w+")
    for r in resources:
        resource_file.write(str(r)+"\n")

def main(argv):
    if len(argv) != 4:
        print("Use mode: python genResourceFile <param1> <param2> <resource name>")
        return
    p1 = argv[1]
    p2 = argv[2]
    resource_name = argv[3]
    generateResourceFile(p1,p2,resource_name)

if __name__ == "__main__":
    main(sys.argv)
