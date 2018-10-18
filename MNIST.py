"""
The program aims to identify 0-9 digits using either MLP or CNN
"""
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras import utils
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os

import tensorflow as tf
mnist = tf.keras.datasets.mnist

# define some hyper parameters
batch_size = 128
n_inputs   = 784
n_classes  = 10
n_epochs   = 20

# get the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape the two dimensional 28 x 28 pixels
#   sized images into a single vector of 784 pixels
x_train = x_train.reshape(60000, n_inputs)
x_test  = x_test.reshape(10000, n_inputs)

# convert the input values to float32
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# normalize the values of image vectors to fit under 1
x_train /= 255
x_test /= 255

# convert output data into one hot encoded format
y_train = utils.to_categorical(y_train, n_classes)
y_test  = utils.to_categorical(y_test, n_classes)

# 1. build a MLP using a sequential model
model = Sequential()
model.add(Dense(units=256, activation='relu', input_shape=(n_inputs,)))
model.add(Dropout(0.5))
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=n_classes, activation='softmax'))

# print the summary of our model
model.summary()

# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(),
              metrics=['accuracy'])

# train the model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=n_epochs)

# evaluate the model and print the accuracy score
scores = model.evaluate(x_test, y_test, verbose=1)
print('\n loss:', scores[0])
print('\n accuracy:', scores[1])  # 0.9638

#2. build a CNN
# get the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[:,:,:,np.newaxis]
x_test  = x_test[:,:,:,np.newaxis]
# convert the input values to float32
x_train = x_train.astype(np.float32)
x_test  = x_test.astype(np.float32)
# normalize the values of image vectors to fit under 1
x_train /= 255
x_test  /= 255
# convert output data into one hot encoded format
y_train = utils.to_categorical(y_train, n_classes)
y_test  = utils.to_categorical(y_test, n_classes)

model = Sequential()
model.add(Conv2D(20, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(50, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=500, activation='relu'))
model.add(Dense(units=n_classes, activation='softmax'))
model.summary()

# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

# train the model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=n_epochs)

# evaluate the model and print the accuracy score
scores = model.evaluate(x_test, y_test, verbose=1)
print('\n loss:', scores[0])
print('\n accuracy:', scores[1])  # 0.991