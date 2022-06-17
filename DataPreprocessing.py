import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import  SVC
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

dataset = ['ant-1.7', 'arc', 'jedit-3.2', 'synapse-1.0', 'tomcat', 'xalan-2.4', 'xerces-1.2']
columns = ['Dataset', 'HistGradientBoosting', 'GradientBoosting', 'ExtraTreesClassifier', 'BaggingClassifier', 'AdaBoostClassifier']
precision_matrix = pd.DataFrame(columns = columns)
recall_matrix = pd.DataFrame(columns = columns)
mcc_matrix = pd.DataFrame(columns = columns)
f1_score_matrix = pd.DataFrame(columns = columns)

i = 0

for x in dataset:
    path = 'Data/' + x + '.csv'
    acc_values = [x]
    pre_values = [x]
    rec_values = [x]
    f1_values = [x]
    df = pd.read_csv(path)
    df = df.drop(['name', 'version', 'name.1'], axis=1)
    
    features = np.array(df.drop(['bug'], axis=1), dtype='float32')
    labels = np.array(df['bug'], dtype='uint8')
    labels[labels>0] = 1
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(features)
    features = scaler.transform(features)
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)
    
    sm = SMOTE()
    X_train, y_train = sm.fit_resample(X_train, y_train)
    
    clf = HistGradientBoostingClassifier(min_samples_leaf=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc_values.append(matthews_corrcoef(y_test, y_pred))
    f1_values.append(f1_score(y_test, y_pred))
    pre_values.append(precision_score(y_test, y_pred))
    rec_values.append(recall_score(y_test, y_pred))
    
    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc_values.append(matthews_corrcoef(y_test, y_pred))
    f1_values.append(f1_score(y_test, y_pred))
    pre_values.append(precision_score(y_test, y_pred))
    rec_values.append(recall_score(y_test, y_pred))
    
    clf = ExtraTreesClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc_values.append(matthews_corrcoef(y_test, y_pred))
    f1_values.append(f1_score(y_test, y_pred))
    pre_values.append(precision_score(y_test, y_pred))
    rec_values.append(recall_score(y_test, y_pred))
    
    clf = BaggingClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc_values.append(matthews_corrcoef(y_test, y_pred))
    f1_values.append(f1_score(y_test, y_pred))
    pre_values.append(precision_score(y_test, y_pred))
    rec_values.append(recall_score(y_test, y_pred))
    
    clf = AdaBoostClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc_values.append(matthews_corrcoef(y_test, y_pred))
    f1_values.append(f1_score(y_test, y_pred))
    pre_values.append(precision_score(y_test, y_pred))
    rec_values.append(recall_score(y_test, y_pred))
    
    mcc_matrix.loc[i] = acc_values
    precision_matrix.loc[i] = pre_values
    recall_matrix.loc[i] = rec_values
    f1_score_matrix.loc[i] = f1_values
    
    i = i+1
    
mcc_matrix.to_csv('mcc.csv')
precision_matrix.to_csv('precision.csv')
recall_matrix.to_csv('recall.csv')
f1_score_matrix.to_csv('f1.csv')