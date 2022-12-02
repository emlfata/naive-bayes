# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 20:53:01 2022

@author: muham
"""

import pandas as pd
df_train = pd.read_csv("titanic_train.csv")

#DATA PROCESSING
df_train['Embarked'].value_counts()
df_train['Embarked'].fillna('S', inplace=True)
embarked = {"Embarked": {"S": 0, "C": 1, "Q": 2}}
df_train.replace(embarked, inplace=True)

df_train.dropna(inplace=True, how='any')

df_train['Fare'] = df_train['Fare'].astype(int)
df_train['Age'] = df_train['Age'].astype(int)

sex = {"Sex": {"male": 0, "female": 1}}
df_train.replace(sex, inplace=True)

#drop unused feature
df_train = df_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
#use 10 data
df_train.head(10)



#CREATE MODEL
import sklearn.model_selection as ms
features = df_train[['Pclass','Embarked','Sex','Age','Fare','SibSp','Parch']]
label = df_train['Survived']
X_train, X_test, y_train, y_test = ms.train_test_split(features, label, test_size=0.25, random_state=0)

#Using Gaussian NB
import sklearn.naive_bayes as nb
import sklearn.metrics as met
gnb = nb.GaussianNB()
gnb.fit(X_train, y_train)

y_prediksi = gnb.predict(X_test)
accuracy = met.accuracy_score(y_test, y_prediksi)
precision = met.precision_score(y_test, y_prediksi)
print('Accuracy=', accuracy, 'Precision=', precision)

import matplotlib.pyplot as plt
y_pred_proba = gnb.predict_proba(X_test) [::,1]
fp, tp, _ = met.roc_curve(y_test, y_pred_proba)
auc = met.roc_auc_score(y_test, y_pred_proba)
plt.plot(fp,tp,label="data 1, auc="+ str(auc))
plt.legend(loc=4)
plt.show()