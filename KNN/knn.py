import numpy as np
from collections import Counter

class KNN:
    def __init__(self,X,y,k=7):
        self.X = X
        self.y = y
        self.k = k

    @staticmethod
    def distance(x0,x1):
        return np.sqrt(np.sum(np.abs(x0-x1)**2))

    def predict(self,x):
        distance_dict = {idx:self.distance(x,x0) for idx,x0 in enumerate(self.X) }
        neighbor_set = sorted(distance_dict.items(),key=lambda x:x[1])[:self.k]
        cnt = Counter([self.y[i[0]] for i in neighbor_set])
        y_hat = self.y[0]
        for key in cnt:
            if cnt[key] > cnt[y_hat]:
                y_hat = key
        return y_hat

if __name__ == '__main__':

    from sklearn.datasets import load_iris
    from sklearn.metrics import f1_score,roc_auc_score,accuracy_score
    data = load_iris()
    X = data['data']
    y = data['target']
    knn = KNN(X,y,k=10)
    print(knn.predict(X[0]))
    y_hat = [knn.predict(x) for x in X]
    print(accuracy_score(y_hat,y))




