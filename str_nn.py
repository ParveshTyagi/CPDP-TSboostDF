from cmath import inf
import numpy as np
from sklearn.utils import shuffle
from imblearn.over_sampling import ADASYN, KMeansSMOTE, BorderlineSMOTE

def datasetCreator(X_sample, y_sample, X_target, instances, num):
    X = []
    y = []
    for instance in instances:
        dist_index = []
        for i in range(len(X_sample)):
            dist = np.linalg.norm(X_sample[i] - X_target[instance])
            dist_index.append((dist, i))
        dist_index.sort(reverse=True)
        index = []
        k = num
        while k>0:
            k -= 1
            index.append(dist_index.pop()[1])
        for ind in index:
            X.append(X_sample[ind].copy())
            y.append(y_sample[ind].copy())
    return X,y
    


def train_and_predict(X_sample, y_sample, X_target, classifier, iterations=20):
    X_sample_clean = []
    y_sample_clean = []
    X_sample_defective = []
    y_sample_defective = []
    for i in range(len(y_sample)):
        if y_sample[i] == 0:
            X_sample_clean.append(X_sample[i].copy())
            y_sample_clean.append(y_sample[i].copy())
        else:
            X_sample_defective.append(X_sample[i].copy())
            y_sample_defective.append(y_sample[i].copy())
    X_sample_clean = np.array(X_sample_clean)
    y_sample_clean = np.array(y_sample_clean)
    X_sample_defective = np.array(X_sample_defective)
    y_sample_defective = np.array(y_sample_defective)
    X_train = X_sample.copy()
    y_train = y_sample.copy()
    X_train, y_train = ADASYN().fit_resample(X_train, y_train)
    # X_train = np.array(X_train)
    # y_train = np.array(y_train)
    
    s = len(X_sample)
    t = len(X_target)
    preds = np.empty((len(X_target),0))
    for i in range(iterations):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        clf = classifier
        clf.fit(X_train, y_train)
        y_target_pred = clf.predict(X_target).reshape((len(X_target),1))
        preds = np.append(preds, y_target_pred, axis=1)
        Tc = []
        Td = []
        Tu = []
        for k in range(len(y_target_pred)):
            n0 = 0
            n1 = 0
            for j in range(i+1):
                if preds[k][j] == 0: n0 += 1
                else: n1 += 1
            if n0 > n1: Tc.append(k)
            elif n1 > n0: Td.append(k)
            else: Tu.append(k)
        nc = len(Tc)
        nd = len(Td)
        nu = len(Tu)
        X_train = []
        y_train = []
        # print(nd*100/nc)
        if nd > 0 and nc > 0:
            n4 = np.round(nc/nd)
            if n4 == 0: n4 = 1
            
            X, y = datasetCreator(X_sample_clean, y_sample_clean, X_target, Tc, 1)
            X_train.extend(X)
            y_train.extend(y)
            
            X, y = datasetCreator(X_sample_defective, y_sample_defective, X_target, Td, n4)
            X_train.extend(X)
            y_train.extend(y)
            
            if nu > 0:
                X, y = datasetCreator(X_sample, y_sample, X_target, Tu, 1)
                X_train.extend(X)
                y_train.extend(y)
                X_train = np.array(X_train)
                y_train = np.array(y_train)
            
        else:
            X_train, y_train = datasetCreator(X_sample, y_sample, X_target, np.arange(len(X_target)), 1)
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            X_train, y_train = ADASYN().fit_resample(X_train, y_train)
            
        X_train, y_train = shuffle(X_train, y_train)
    
    return preds[:,-1]
