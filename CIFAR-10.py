"""
This program uses a convolutional neural network to learn from CIFAR-10 data.
"""
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import adam
import numpy as np
from keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_train /=255
x_test = x_test.astype('float32')
x_test /=255

y_train = to_categorical(y_train,10)
y_test  = to_categorical(y_test,10)

# build a CNN
model = Sequential()
model.add(Conv2D(100, kernel_size=3, activation='relu',input_shape=(32,32,3), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(100, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(100, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
model.summary()
model.compile(optimizer=adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# fit the model
model.fit(x_train, y_train, batch_size=128, epochs=20)

# evaluate the model
results = model.evaluate(x_test, y_test)
print("Test score:", results[0])
print("Test score:", results[1])   # 0.74
