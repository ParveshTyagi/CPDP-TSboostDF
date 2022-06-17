from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from tpot import TPOTClassifier
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.metrics import f1_score, make_scorer, matthews_corrcoef
from sklearn.neural_network import MLPClassifier

def datasetMaker(dataset_name):
    path = 'Data/' + dataset_name + '.csv'
    df = pd.read_csv(path)
    df = df.drop(['name', 'version', 'name.1'], axis=1)
    
    X = np.array(df.drop(['bug'], axis=1), dtype='float32')
    y = np.array(df['bug'], dtype='uint8')
    y[y>0] = 1
    
    return X, y

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

X_sample, y_sample = datasetMaker('xerces-1.2')
X_target, y_target = datasetMaker('arc')

sw = similarityWeight(X_sample, X_target)
X, y = dataSelector(X_sample, y_sample, sw, 0.8)
    
imb = ADASYN(random_state=13)
X_train, y_train = imb.fit_resample(X, y)

tpot_config = {  
                'sklearn.naive_bayes.GaussianNB' : {
                    },
                'sklearn.neural_network.MLPClassifier' : {}
}


               
tpot_classifier = TPOTClassifier(generations= 5, population_size=30, offspring_size= 15,
                                 verbosity= 2, early_stop= 12,
                                 config_dict=tpot_config,
                                 scoring='f1',
                                 cv = 4)
tpot_classifier.fit(X,y)

print(f1_score(y_sample, tpot_classifier.predict(X_sample)))


y_target_pred = tpot_classifier.predict(X_target)

print(f1_score(y_target, y_target_pred))
print(matthews_corrcoef(y_target, y_target_pred))
