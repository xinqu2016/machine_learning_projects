# This program classifies two types of breast tumors (Benign vs. malignant) using support vector machine.
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

cancer_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
                          header=None)
cancer_data.head(10)
print(len(cancer_data))
print(cancer_data.describe())

cancer_data_features = cancer_data.drop(cancer_data[[0,1]],axis=1)
print(cancer_data_features.info())
min_max_scaler = preprocessing.MinMaxScaler()
cancer_data_features_scaled = pd.DataFrame(min_max_scaler.fit_transform(cancer_data_features))
print(cancer_data_features_scaled.describe())

cancer_data_labels  = cancer_data[1]
cancer_data_labels  = cancer_data_labels.map({'M':0, 'B': 1})
print(cancer_data_labels)

x_train, x_test, y_train, y_test = train_test_split(cancer_data_features_scaled, cancer_data_labels,
                                                    test_size=0.3, random_state=101)

from sklearn.svm import SVC
model = SVC()

model.fit(x_train, y_train)
prediction = model.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, prediction))  # precision is 0.94.
