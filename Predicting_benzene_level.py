# This program uses keras APIs to develop a preditive model for the benzene level

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense
from keras.models import Model

data_original = pd.read_csv("../../downloads/AirQualityUCI/AirQualityUCI.csv", sep=';', decimal=',', header=0)
print(data_original.head(10))

del data_original["Date"]
del data_original["Time"]
del data_original["Unnamed: 15"]
del data_original["Unnamed: 16"]

data_original = data_original.fillna(data_original.mean())
Xorig = data_original.values

scaler = StandardScaler()
Xscaled = scaler.fit_transform(Xorig)

y = Xscaled[:,3]
X = np.delete(Xscaled, 3, axis=1)

train_size = int(0.7*X.shape[0])
Xtrain, Xtest, Ytrain, Ytest = X[0:train_size,:], X[train_size:,:], \
                               y[0:train_size], y[train_size:]

Hidden_size = 8
input1 = Input(shape=(12,))
x = Dense(Hidden_size, activation='relu', kernel_initializer="glorot_uniform")(input1)
prediction = Dense(1, kernel_initializer="glorot_uniform")(x)

model = Model( inputs = [input1], outputs = [prediction])
model.compile(optimizer="adam", loss="mse")

model.fit(Xtrain, Ytrain, batch_size = 10, epochs = 40, validation_split=0.2)   # validation msd = 6e-4


