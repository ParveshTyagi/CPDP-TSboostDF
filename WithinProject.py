import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.naive_bayes import GaussianNB

dataset = np.asarray(['arc', 'xerces-1.2', 'tomcat', 'xalan-2.4', 'synapse-1.0', 'ant-1.7', 'jedit-3.2'])

def datasetMaker(dataset_name):
    path = 'Data/' + dataset_name + '.csv'
    df = pd.read_csv(path)
    df = df.drop(['name', 'version', 'name.1'], axis=1)
    
    X = np.array(df.drop(['bug'], axis=1), dtype='float32')
    y = np.array(df['bug'], dtype='uint8')
    y[y>0] = 1
    
    return X, y

for i in range(len(dataset)):
    X, y = datasetMaker(dataset[i])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25)
    
    X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    
    clf = GaussianNB()
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print(f1_score(y_test, y_pred))
