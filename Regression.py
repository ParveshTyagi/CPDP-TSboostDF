import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("Data/tomcat.csv")

df = df.drop(['name', 'version', 'name.1'], axis = 1)

import numpy as np

features = np.array(df.drop(['bug'], axis=1), dtype='float32')
labels = np.array(df['bug'], dtype='uint8')

labels[labels>0] = 1


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)

clf = svm.SVC()
clf.fit(X_train, y_train)
    

y_predict = clf.predict(X_test)

result = confusion_matrix(y_test, y_predict)

print(result)
