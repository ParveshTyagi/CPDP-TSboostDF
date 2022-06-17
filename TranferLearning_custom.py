import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import NearMiss
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from skopt import BayesSearchCV
from str_nn import train_and_predict
from tpot import TPOTClassifier

import warnings
warnings.filterwarnings("ignore")

from da_tool.tca import TCA


target = pd.read_csv('Data/tomcat.csv')
source = pd.read_csv('Data/arc.csv')

source.loc[(source.bug>0), 'bug'] = 1
target.loc[(target.bug>0), 'bug'] = 1



X_train = source.drop(['name', 'version', 'name.1','bug'], axis=1)
# st = StandardScaler()
# X_train = st.fit_transform(X_train)
X_test = target.drop(['name', 'version', 'name.1','bug'], axis=1)
# X_test = st.transform(X_test)




y_train = source['bug']
y_test = target['bug']

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

X_train_tca, X_test_tca, _ = TCA().fit_transform(X_train, X_test)

# params = {
#     'var_smoothing': np.logspace(0,-9, num=100)
# }

# clf = RandomizedSearchCV(estimator=GaussianNB(), param_distributions=params, scoring='f1')

# clf = BayesSearchCV(estimator=GaussianNB(), search_spaces= params, scoring='f1')
# clf.fit(X_train_tca, y_train)

clf1 = KNeighborsClassifier(weights='distance')
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()

clf = VotingClassifier(
     estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
     voting='soft')

# prediction = clf.predict(X_test_tca)


prediction = train_and_predict(X_train, y_train, X_test, clf, 50)

print('F1 score: ', f1_score(y_test, prediction))
print('AUC: ', roc_auc_score(y_test, prediction))



