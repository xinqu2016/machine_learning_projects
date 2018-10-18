"""
This program compares the performance of a decision tree model and a random forest model applied to a loan data
(https://www.lendingclub.com/info/download-data.action). The goal of the task is to predict the loan status (fully paid
vs. charged off) given other features.
"""

import pandas as pd
import matplotlib.pyplot as plt

loan = pd.read_csv('LoanStats3a.csv', skiprows=1)
print(loan.keys())
loan_downsized = loan[['annual_inc','installment','dti','revol_bal','revol_util','inq_last_6mths',\
                       'delinq_2yrs','pub_rec','int_rate','purpose','loan_status']]

# Step 1: Data cleaning
plt.figure(figsize = (10, 6))
loan_downsized['annual_inc'].hist(bins=100)
loan_downsized['installment'].hist(bins=100)
loan_downsized['dti'].hist(bins=100)
loan_downsized['revol_bal'].hist(bins=100)
loan_downsized['inq_last_6mths'].hist(bins=100)
loan_downsized['delinq_2yrs'].hist(bins=100)
loan_downsized['pub_rec'].hist(bins=100)

# For annual_inc and revol_bal, we identified a long tail from 200000.0 up to 6000000.0 which comprises of 1.8% of the data.
# For robustness we set it to None
loan_downsized['annual_inc'][loan_downsized['annual_inc']>200000.0] = None
loan_downsized['revol_bal'][loan_downsized['revol_bal']>200000.0]   = None

# revol_util and int_rate are strings ending with '%'. We delete the '%' and change them to float64.
import re
loan_downsized['revol_util'] = pd.to_numeric(loan_downsized['revol_util'].map(lambda x: re.sub('[%]','',x), na_action='ignore'))
loan_downsized['int_rate']   = pd.to_numeric(loan_downsized['int_rate'].map(lambda x: re.sub('[%]','',x), na_action='ignore'))
loan_downsized['revol_util'].hist(bins=100)
loan_downsized['int_rate'].hist(bins=100)

# loan_status is a string. We change it to binary values.
loan_downsized['loan_status'] = loan_downsized['loan_status'].map(lambda x: 0 if 'Fully Paid' in x else 1, na_action='ignore')
loan_downsized['loan_status'].hist(bins=2)

# purpose is a categorical variable. We change it to multi-dimensional binary values.
loan_final = pd.get_dummies(loan_downsized,columns=['purpose']).dropna()

# Step 2: Train test split
from sklearn.model_selection import train_test_split
X = loan_final.drop('loan_status', axis=1)
y = loan_final['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

# Step 3: Train, predict and evaluate a decision tree model
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_prediction = dtree.predict(X_test)
print(classification_report(y_test,y_prediction ))   # precision is 0.75, but is only 0.20 for class 1

# Step 4: Train, predict and evaluate a random forest model
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=800)
rfc.fit(X_train, y_train)

y_prediction = rfc.predict(X_test)
print(classification_report(y_test,y_prediction))   # precision is 0.80. In particular, precision for class 1 increases to 0.51