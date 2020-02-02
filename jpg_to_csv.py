import skimage.measure
import csv
import sys
from PIL import Image
import numpy as np
import os
import os.path
import time
import pandas as pd
import matplotlib.pyplot as plt
format = '.jpg'


def createFileList(myDir, format='.jpg'):

    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList


#myDir: name of directory contataining jpg images
#numToConvert: num of images in the directory to convert to pixels
def convert(myDir, numToConvert):

# load the original image
    myFileList = createFileList(myDir)

    i = 0
    for file in myFileList:
            # print(file)
            img_file = Image.open(file)
            # img_file.show()

            # get original image parameters...
            width, height = img_file.size
            format = img_file.format
            mode = img_file.mode

            # Make image Greyscale
            img_grey = img_file.convert('L')
            # img_grey.save('result.png')
            # img_grey.show()

            # Save Greyscale values
            value = np.asarray(img_grey.getdata(), dtype=np.int).reshape(
                (img_grey.size[1], img_grey.size[0]))
            # value = np.asarray(img_file.getdata(), dtype=np.int).reshape(
            #     (img_file.size[1], img_grey.size[0], 3))

            value = skimage.measure.block_reduce(value, (2, 2), np.max)
            value = value.flatten()
            # print(value)
            with open("img_pixels.csv", 'a') as f:
                writer = csv.writer(f)
                writer.writerow(value)
            i += 1
            if i > numToConvert - 1:
                break


#file_name: name of file containing rows of pixels
#N: number of pictures located in file_name
#size: a size x size image
def read_from_csv(file_name):
    data = np.genfromtxt(file_name, delimiter=',')
    # data = data.reshape(N,size,size)
    return data


# myDir = "faces"
# N = 200
# # convert(myDir, N)

# someLoc = 'img_pixels.csv'
# size = 125

# data = read_from_csv(someLoc, N, size)


# plot = plt.imshow(data[2], cmap='gray')
# plt.show()
# print('done')
