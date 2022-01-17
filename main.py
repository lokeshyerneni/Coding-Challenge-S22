# Sources: 
# Types of Binary Classification: https://www.learndatasci.com/glossary/binary-classification/#APythonExampleforBinaryClassification
# Choose what estimator to use: https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
# Label Encoder: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
# SVC: https://scikit-learn.org/stable/tutorial/basic/tutorial.html#learning-and-predicting
# SVC: https://stackoverflow.com/questions/38584829/svc-support-vector-classification-with-categorical-string-data-as-labels

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('mushrooms.csv')
enc = preprocessing.LabelEncoder()

for col in df.columns:
    df[col] = enc.fit_transform(df[col])


x = df.loc[:, df.columns != 'class']
y = df["class"]

print(x)
print(y)

X_train, X_test, y_train, y_test = train_test_split(x,  y, test_size=0.2)

model = LinearSVC(verbose=0)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

cm = confusion_matrix(y_test, predictions)

TN, FP, FN, TP = confusion_matrix(y_test, predictions).ravel()

print('True Positive  = ', TP)
print('False Positive = ', FP)
print('True Negative  = ', TN)
print('False Negative = ', FN)

accuracy =  (TP+TN) /(TP+FP+TN+FN)

print('Accuracy of the binary classification = {:0.3f}'.format(accuracy))