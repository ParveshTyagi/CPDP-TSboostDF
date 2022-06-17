from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from imblearn.over_sampling import ADASYN
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

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

def DL_Model(activation= 'linear', neurons= 5, optimizer='Adam'):
    model = Sequential()
    model.add(Dense(neurons, input_dim= (X_train.shape[1]), activation= activation))
    model.add(Dense(neurons, activation= activation))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer= optimizer, metrics=['accuracy'])
    return model

X_sample, y_sample = datasetMaker('xerces-1.2')
X_target, y_target = datasetMaker('arc')

sw = similarityWeight(X_sample, X_target)
X, y = dataSelector(X_sample, y_sample, sw, 1)
    
imb = ADASYN(random_state=13)
X_train, y_train = imb.fit_resample(X, y)

# Definying grid parameters
activation = ['softmax', 'relu', 'tanh', 'sigmoid', 'linear']
neurons = [5, 10, 15, 25, 35, 50]
optimizer = ['SGD', 'Adam', 'Adamax']
param_grid = dict(activation = activation, neurons = neurons, optimizer = optimizer)

clf = KerasClassifier(build_fn= DL_Model, epochs= 80, batch_size=40, verbose= 0)

model = GridSearchCV(estimator= clf, param_grid=param_grid, n_jobs=-1)
model.fit(X_train, y_train)

print("Max Accuracy Registred: {} using {}".format(round(model.best_score_,3), 
                                                   model.best_params_))

y_target_pred = model.predict(X_target)

print(f1_score(y_target, y_target_pred))