import numpy as np
import random
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from skopt import BayesSearchCV
from tpot import TPOTClassifier


class TSboostDF:
    w = np.asarray([])
    similarityWeights = np.asarray([])
    Classifiers = []
    distances = np.asarray([])

        
    def cumulativeWeight(self, weights):
        p = weights.copy()
        p = np.cumsum(p)
        u = random.random() * p[-1]
        index = 0
        for index in range(len(p)):
            if p[index] > u:
                break
        return index


    def similarityWeight(self, X_sample, X_target):
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



    def BSW(self, X_sample, y_sample, Psize):
        w_0 = []
        w_1 = []
        
        X_final = []
        y_final = []
        
        X_0 = []
        X_1 = []
        
        for i in range(len(y_sample)):
            if y_sample[i] == 1:
                X_1.append(X_sample[i])
                w_1.append(self.w[i])
            else:
                X_0.append(X_sample[i])
                w_0.append(self.w[i])
                
        w_0 = np.asarray(w_0)
        w_1 = np.asarray(w_1)
        
        X_0 = np.asarray(X_0)
        X_1 = np.asarray(X_1)
        
        for i in range(Psize):
            index = self.cumulativeWeight(w_0)
            X_final.append(X_0[index])
            y_final.append(0)
            
        for i in range(Psize):
            index = self.cumulativeWeight(w_1)
            X_final.append(X_1[index])
            y_final.append(1)
        
        X_final = np.asarray(X_final)
        y_final = np.asarray(y_final)  
        
        X_final, y_final = shuffle(X_final, y_final, random_state=22)     
        
        return X_final, y_final 
            

    def createTestData(self, X_sample, y_sample, similarity_theshold, sampling_percentage):
        X_final = []
        y_final = []
        for i in range(len(y_sample)):
            if self.similarityWeights[i] >= similarity_theshold:
                X_final.append(X_sample[i])
                y_final.append(y_sample[i])
        
        X_final = np.asarray(X_final)
        y_final = np.asarray(y_final)
        
        X_test, _, y_test, _ = train_test_split(X_final, y_final, test_size=sampling_percentage, random_state=22)
        
        return X_test, y_test
        

    def errorCalculation(self, clf, X_test, y_test):
        y_pred = clf.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        return 1-score

    def distance(self, X_target, X_train):
        mean_target = X_target.mean(0)
        mean_train = X_train.mean(0)
        
        std_target = X_target.std(0)
        std_train = X_train.std(0)
        
        min_target = X_target.min(0)
        min_train = X_train.min(0)
        
        max_target = X_target.max(0)
        max_train = X_target.max(0)
        
        ans = 0
        
        ans = np.linalg.norm(mean_target - mean_train)
        ans = ans + np.linalg.norm(std_target - std_train)
        ans = ans + np.linalg.norm(min_target - min_train)
        ans = ans + np.linalg.norm(max_target - max_train)
        return ans


    def updateWeight(self, clf, alpha, X_sample, y_sample, similarity_threshold):
        y_predict = clf.predict(X_sample)
        
        for i in range(len(y_sample)):
            v = self.w[i]
            
            z = 0
            
            if self.similarityWeights[i] >= similarity_threshold:
                if y_predict[i] == y_sample[i]:
                    v = v * np.exp(alpha * self.similarityWeights[i])
                else:
                    v = v * np.exp(-alpha * self.similarityWeights[i])
            else:
                if alpha > 0:
                    if y_predict[i] == y_sample[i]:
                        v = v * np.exp(alpha * self.similarityWeights[i])
                    else:
                        v = v * np.exp(-alpha)
                else:
                    if y_predict[i] == y_sample[i]:
                        v = v * np.exp(alpha)
                    else:
                        v = v * np.exp(-alpha * self.similarityWeights[i])
            
            self.w[i] = v
            
            z = z + self.w[i]
        
        for i in range(len(y_sample)):
            self.w[i] = self.w[i]/z
            

    def fit(self, X_sample, y_sample, X_target, similarity_threshold=0.7, sampling_percentage=0.3, loopNum=15, Psize=0.5):
        X_sample = np.asarray(X_sample)
        y_sample = np.asarray(y_sample)
        X_target = np.asarray(X_target)
        
        Psize = int(Psize * len(y_sample))
        
        ClassificationPool = []
        distancePool = []
        self.similarityWeights = self.similarityWeight(X_sample, X_target)
        self.w = np.ones(len(y_sample)) * (1/len(y_sample))
        
        while loopNum > 0:
            X_test, y_test = self.createTestData(X_sample, y_sample, similarity_threshold, sampling_percentage)
            alpha = -1
            
            
            clf1 = KNeighborsClassifier(n_neighbors=10, weights='distance')
            clf2 = MLPClassifier()
            clf3 = GaussianNB()

            clf = VotingClassifier(
                estimators=[('knn', clf1), ('rf', clf2), ('gnb', clf3)],
                voting='soft')
            

            X_train = 0
            
            while alpha < 0:
                X_train, y_train = self.BSW(X_sample, y_sample, Psize)
                clf.fit(X_train, y_train)
                error = self.errorCalculation(clf, X_test, y_test)
                alpha = 0.5 * np.log((1-error)/error)
                self.updateWeight(clf, alpha, X_sample, y_sample, similarity_threshold)
                
            
            dis = self.distance(X_target, X_train)
            ClassificationPool.append(clf)
            distancePool.append(dis)
            loopNum = loopNum-1
            
        
        sum_dist = sum(distancePool)
        self.Classifiers = ClassificationPool
        self.distances = np.asarray(distancePool)/sum_dist
        
        X_test, y_test = self.createTestData(X_sample, y_sample, similarity_threshold, sampling_percentage)
        # self.print_score(X_test, y_test)
        
        
    def predict(self, X_target):
        X_target = np.asarray(X_target)
        
        y_final = np.zeros(shape=X_target.shape[0])
        
        for i in range(len(self.Classifiers)):
            y = self.Classifiers[i].predict(X_target)
            y = y * self.distances[i]
            
            y_final = np.add(y_final, y)
        
        y_final = np.round(y_final)
        
        return y_final
    
    def print_score(self, X_target, y_target):
        X_target = np.asarray(X_target)
        
        y_final = np.zeros(shape=X_target.shape[0])
        
        for i in range(len(self.Classifiers)):
            y = self.Classifiers[i].predict(X_target)
            y = y * self.distances[i]
            
            y_final = np.add(y_final, y)
        
        y_final = np.round(y_final)
        
        print(f1_score(y_target, y_final), matthews_corrcoef(y_target, y_final))
        
    

    
        