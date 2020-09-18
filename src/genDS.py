
import pathlib
import os
import math
import glob
import random
import warnings

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import Callback

np.set_printoptions(precision=10)

warnings.filterwarnings('ignore')



W_H=240


data_dir = [    os.path.abspath("../dataset/occlusion_2_unconstrained/"),os.path.abspath("../dataset/swinging_4_unconstrained/"),os.path.abspath("/home/victor/GitRepos/dataset/1000")]

def decode_mask(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img)
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.convert_image_dtype(img, tf.float16)
    img = tf.image.resize(img, [W_H,W_H])
    img.set_shape([W_H,W_H,1])
    
    return img

def decode_opticalFlow(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img)
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.convert_image_dtype(img, tf.float16)
    img = tf.image.resize(img, [W_H,W_H])
    img.set_shape([W_H,W_H,1])
    
    return img


def decode_labels(path):
    data = tf.io.read_file(path)
    string_labels = tf.strings.split(data, '\n')
    labels = tf.strings.to_number(string_labels, tf.float32)
    labels = tf.cast(labels, dtype=tf.float16)
    labels.set_shape([100,])
    return labels


def loadData(path):
    path = path + '/'
    
    opticalFlow = decode_opticalFlow(path + '0.png')
    
    masks = []
    for i in range(1,101):
        im = decode_mask(path + str(i) + '.png')
        masks.append(im)
    masks = tf.convert_to_tensor(masks, tf.float16)

    images = tf.concat([[opticalFlow], masks], 0)

    labels = decode_labels(path+"GT")

    return images, labels



def getInterval(list, inicio, fin):
    return list[inicio:fin]

def saveDataset(srcDir, outDir= "../dataset/", datasetName = "dataset", parts=100):
    train_list = glob.glob(srcDir)
    random.shuffle(train_list)


    DATASET_SIZE = len(train_list)
    print(DATASET_SIZE)
    part_size = DATASET_SIZE // parts
	
    
    print("SAVING DATASET, in ", parts, " of " , part_size, " elements.")
    for i in range(parts):

        filesList = getInterval(train_list, i*part_size, ((i+1)*part_size))
        currentDS = tf.data.Dataset.list_files(filesList)
        currentDS = currentDS.map(loadData, num_parallel_calls=4)

        size = tf.data.experimental.cardinality(currentDS).numpy()
        print("saving part " + str(i+1) +  " out of " +str(parts) + "\n\tsize: " + str(size) + "\n\t" + str(i*part_size) + "--" + str((i+1)*part_size))
        route = path + datasetName + "-" + str(i+1) + "_" + str(parts)
        tf.data.experimental.save(dataset=currentDS, path = route, compression='GZIP')
    print("DONE")
    
    exit()



if __name__ == '__main__':
    saveDataset(	srcDir = "../datset/occlusion_2_unconstrained/",
					outDir = "../dataset_23/fullDS/",
					datasetName = "dataset",
					parts = 100
				)


    