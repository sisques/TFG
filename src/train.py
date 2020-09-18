
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


weigths_path = os.path.abspath("../weights/DYnamicCNN/")



def plotAcc(history,name):
    plt.figure()
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("../plots/"+name+"_accuracy.png")

def plotLoss(history,name):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("../plots/"+name+"_loss.png")

def getModel(regularized = False):
    if regularized:
      inputA = tf.keras.Input(shape=(101,W_H,W_H,1))
      #CNN

      x = tf.keras.layers.Conv3D(
              filters = 32, 
              kernel_size = (1,3,3), 
              activation = 'relu', 
              strides = (1,2,2), 
              padding = 'same',
              data_format = 'channels_last',
              use_bias=True,
              kernel_regularizer=tf.keras.regularizers.l2(l=0.01)
          )(inputA)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.Activation('relu')(x)
      x = tf.keras.layers.Conv3D(
              filters = 64,
              kernel_size = (1,3,3),
              activation = 'relu',
              strides = (1,3,3),
              padding = 'same',
              data_format = 'channels_last',
              use_bias=True,
              kernel_regularizer=tf.keras.regularizers.l2(l=0.01)
          )(x)
      x = tf.keras.layers.Conv3D(
              filters = 128,
              kernel_size = (1,3,3),
              activation = 'relu',
              strides = (1,4,4),
              padding = 'same',
              data_format = 'channels_last',
              use_bias=True,
              kernel_regularizer=tf.keras.regularizers.l2(l=0.01)
          )(x)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.Activation('relu')(x)
      x = tf.keras.layers.Conv3D(
              filters = 64,
              kernel_size = (1,3,3),
              activation = 'relu',
              strides = (1,2,2),
              padding = 'same',
              data_format = 'channels_last',
              use_bias=True,
              kernel_regularizer=tf.keras.regularizers.l2(l=0.01)
          )(x)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.Activation('relu')(x)
      x = tf.keras.layers.Conv3D(
              filters = 32,
              kernel_size = (1,3,3),
              activation = 'relu',
              strides = (1,1,1),
              padding = 'same',
              data_format = 'channels_last',
              use_bias=True,
              kernel_regularizer=tf.keras.regularizers.l2(l=0.01)
          )(x)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.Activation('relu')(x)
      x = tf.keras.layers.Dropout(0.2)(x)
      x = tf.keras.layers.Flatten()(x)
      x = tf.keras.layers.Dense(
              512,
              activation='relu',
              use_bias=True,
              kernel_regularizer=tf.keras.regularizers.l2(l=0.01)
          )(x)
      x = tf.keras.layers.Dropout(0.2)(x)
      x = tf.keras.layers.Dense(
              100,
              activation='sigmoid',
              use_bias=True
          )(x)
      x = tf.keras.Model(inputs = inputA, outputs = x)

      return x
    else:
        inputA = tf.keras.Input(shape=(101,W_H,W_H,1))
        #CNN

        x = tf.keras.layers.Conv3D(
                filters = 32, 
                kernel_size = (1,3,3), 
                activation = 'relu', 
                strides = (1,2,2), 
                padding = 'same',
                data_format = 'channels_last',
                use_bias=True
            )(inputA)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv3D(
                filters = 32,
                kernel_size = (1,3,3),
                activation = 'relu',
                strides = (1,3,3),
                padding = 'same',
                data_format = 'channels_last',
                use_bias=True,
            )(x)
        x = tf.keras.layers.Conv3D(
                filters = 32,
                kernel_size = (1,3,3),
                activation = 'relu',
                strides = (1,4,4),
                padding = 'same',
                data_format = 'channels_last',
                use_bias=True,
            )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(
                100,
                activation='sigmoid',
                use_bias=True
            )(x)
        x = tf.keras.Model(inputs = inputA, outputs = x)
        
        return x


def train(train, test, val, batch_size = 2, epochs = 10, lr = 0.005, modelName="model", regularized=False):



  with tf.device('/device:GPU:0'):
    train_size = tf.data.experimental.cardinality(train).numpy()    
    val_size = tf.data.experimental.cardinality(val).numpy()
    test_size = tf.data.experimental.cardinality(test).numpy()

    print("Training info:", 
          "\n\tTrain ds size =", train_size,
          "\n\tVal ds size =", val_size,
          "\n\tTest ds size =", test_size,
          "\n\tNum epochs =", epochs,
          "\n\tBatch size =", batch_size,
          "\n\tLearning rate =", lr)
    

    train = train.batch(batch_size)
    train = train.repeat(epochs)
    train = train.shuffle(batch_size)
    train = train.prefetch(batch_size)

    val = val.batch(batch_size)
    val = val.repeat(epochs)
    val = val.shuffle(batch_size)
    val = val.cache()
    val = val.prefetch(batch_size)

    test = test.batch(batch_size)
    test = test.repeat(epochs)
    test = test.shuffle(batch_size)
    test = test.prefetch(batch_size)
    
    model = getModel(regularized)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=True),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
        )

    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_binary_accuracy', mode='max', patience=5
    )    

    callbacks = [ es_callback ]
    
    history = model.fit(  train,
                batch_size = batch_size,
                steps_per_epoch = train_size // batch_size,
                validation_data = val,
                validation_batch_size = batch_size,
                validation_steps = val_size // batch_size,
                validation_freq = 5,
                epochs = epochs,
                #callbacks = callbacks,
                verbose = 1
            )
    
    plotAcc(history, modelName)
    plotLoss(history, modelName)

    print("saving model")

    model.save(weigths_path + "/"+modelName+".h5")
    model = tf.keras.models.load_model(weigths_path+"/"+modelName+".h5")


    print("Evaluating the trained model")
    loss, acc = model.evaluate(test, verbose=1, steps = test_size // batch_size)
    print("Trained model, accurac: {:5.2f}%".format(100*acc))



def loadDataset(path="../dataset/dataset", parts=5, limit = None):
    if parts <= 0:
        exit("Parts must be >= 1")
    if limit==None:
      limit=parts
    parts_path = []
    for i in range(parts):
        num = i+1
        act = path+ "-" + str(num) + "_" + str(limit)
        parts_path.append(act)
    i = 0

  
  
    ds = load(path = parts_path[i], 
                     element_spec = (tf.TensorSpec(shape=(101, 240, 240, 1), dtype=tf.float32, name=None), tf.TensorSpec(shape=(100,), dtype=tf.float32, name=None)),
                     compression = 'GZIP')
  
    while i < parts - 1:
        i +=1
        aux = load( path = parts_path[i], 
                    element_spec = (tf.TensorSpec(shape=(101, 240, 240, 1), dtype=tf.float32, name=None), tf.TensorSpec(shape=(100,), dtype=tf.float32, name=None)),
                    compression = 'GZIP')
        ds = ds.concatenate(aux)
        
    DATASET_SIZE = tf.data.experimental.cardinality(ds).numpy()

    train_size = int(0.7 * DATASET_SIZE)
    val_size = int(0.2 * DATASET_SIZE)
    test_size = int(0.1 * DATASET_SIZE)

    train_ds = ds.take(train_size)
    test_ds = ds.skip(train_size)
    val_ds = test_ds.skip(test_size)
    test_ds = test_ds.take(test_size)

    print("Se ha cargado un dataset de tamaño " +  str(DATASET_SIZE) + ".")
    
    return train_ds, val_ds, test_ds

if __name__ == '__main__':

	train_ds, val_ds, test_ds = loadDataset("../dataset_23/fullDS/dataset",10,100)



    
    train(
        train_ds,
        test_ds,
        val_ds, 
        batch_size = 2,
        epochs = 100,
        lr = 5e-4,
        modelName="m1000")

    