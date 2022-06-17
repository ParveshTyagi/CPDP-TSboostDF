import pandas as pd
import numpy as np
from sklearn.metrics import  confusion_matrix, f1_score, matthews_corrcoef, recall_score
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB

import warnings

warnings.filterwarnings("ignore")

from da_tool.tca import TCA

from AlgoCPCD import TSboostDF

dataset = np.asarray(['arc', 'xerces-1.2', 'tomcat', 'xalan-2.4', 'synapse-1.0', 'ant-1.7', 'jedit-3.2'])


scores = pd.DataFrame(columns=['Source', 'Target', 'F1', 'MCC', 'G-Mean', 'Balance'])

def GMean_and_Balance(y_true, y_pred):
    CM = confusion_matrix(y_true, y_pred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    
    recall = recall_score(y_true, y_pred)
    
    pf = FP / (FP + TN)
    
    GMean = np.sqrt(recall * (1-pf))
    
    Balance = 1 - (np.sqrt(pf**2 + (1 - recall)**   2) / np.sqrt(2))
    
    return GMean, Balance

def similarityWeight(X_sample, X_target):
    max_elem = np.amax(X_target, axis=0)
    min_elem = np.amin(X_target, axis=0)
    
    similarityValue = []
    
    s_max = 0
    
    for i in range(X_sample.shape[0]):
        s = 0
        for j in range(X_sample.shape[1]):
            if min_elem[j] <= X_sample[i,j] <= max_elem[j]:
                s += 1
        
        s = s/X_sample.shape[1]
        s_max = max(s_max, s)
        similarityValue.append(s)
    
    similarityWeights = np.array(similarityValue)/s_max
    
    return similarityWeights

def dataSelector(X_sample, y_sample, similarityWeights, threshold):
    X = []
    y = []
    for i in range(len(y_sample)):
        if similarityWeights[i] >= threshold:
            X.append(X_sample[i])
            y.append(y_sample[i])
            
    if not np.any(y == 1) or np.count_nonzero(y) < len(y)//20:
        for i in range(len(y_sample)):
            if threshold > similarityWeights[i] >= threshold*0.7:
                X.append(X_sample[i])
                y.append(y_sample[i])
    
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y
    

def datasetMaker(dataset_name):
    path = 'Data/' + dataset_name + '.csv'
    df = pd.read_csv(path)
    df = df.drop(['name', 'version', 'name.1'], axis=1)
    
    X = np.array(df.drop(['bug'], axis=1), dtype='float32')
    y = np.array(df['bug'], dtype='uint8')
    y[y>0] = 1
    
    return X, y

last = 0

for i in range(len(dataset)):
    for j in range(len(dataset)):
        if i == j: continue
        arr = []
        arr.append(dataset[j])  
        arr.append(dataset[i])
        X_sample, y_sample = datasetMaker(dataset[j])
        X_target, y_target = datasetMaker(dataset[i])
                
        
        
        # sw = similarityWeight(X_sample, X_target)
        # X, y = dataSelector(X_sample, y_sample, sw, 0.8)
        
        
        # X, y = X_sample, y_sample
        
        X_sample, X_target, _ = TCA().fit_transform(X_sample, X_target)
        
        # sc = StandardScaler()
        
        # X = sc.fit_transform(X)
        # X_target = sc.fit_transform(X_target)
        
            
        imb = ADASYN()
        X_sample, y_sample = imb.fit_resample(X_sample, y_sample)
        
        
        clf = BaggingClassifier(base_estimator=MLPClassifier())
        clf.fit(X_sample, y_sample)
        
        
        
        # clf = TSboostDF()
        # clf.fit(X_sample, y_sample, X_target, 0.6, 0.2, 10, 0.5)
        
        # clf = GaussianNB()
        # clf = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=20, weights='distance'))
        # clf = AdaBoostClassifier()
        # clf.fit(X_sample,y_sample)
        
        
        # y_pred = train_and_predict(X_sample, y_sample, X_target, clf)
        
        
        # clf1 = KNeighborsClassifier(weights='distance')
        # clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
        # clf3 = GaussianNB()

        # clf = VotingClassifier(
        #     estimators=[('knn', clf1), ('rf', clf2), ('gnb', clf3)],
        #     voting='soft', )
        
        
        # clf.fit(X_sample, y_sample)
        
        y_pred = clf.predict(X_target)
        
        arr.append(f1_score(y_target, y_pred))
        arr.append(matthews_corrcoef(y_target, y_pred))
        GMean, Balance = GMean_and_Balance(y_target, y_pred)
        arr.append(GMean)
        arr.append(Balance)
        
        print(arr)
                        
        scores.loc[last] = arr
        last = last + 1
        
scores.to_csv('Results/Bagging_MLP.csv')
        
        
