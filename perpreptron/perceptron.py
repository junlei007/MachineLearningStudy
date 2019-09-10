import numpy as np

class Perceptron:
    def __init__(self):
        self.w = None
        self.b = 0

    def fit(self,X,y,learn_rate=0.01,n_generations=50):
        n = len(X[0])
        self.w = np.zeros(n)
        for _ in range(n_generations):
            for x0,y0 in zip(X,y):
                loss = y0 * (np.dot(self.w,x0) + self.b )
                if loss <= 0:
                    self.w += learn_rate*x0*y0
                    self.b += learn_rate*y0

    def predict(self,X):
        return np.sign(np.dot(X,self.w)+self.b)


class PerceptronDuality:
    def __init__(self):
        self.alpha = None
        self.w = None
        self.b = 0

    def fit(self,X,y,learn_rate=0.01,n_generations=50):

        n = len(y)
        self.alpha = np.zeros(n)

        for _ in range(n_generations):
            for i in range(n):

                w = np.dot(self.alpha*y,X)

                if y[i]*(np.dot(w,X[i]) + self.b) <=0:
                    self.alpha[i] += learn_rate
                    self.b += learn_rate*y[i]
        self.w = np.dot(self.alpha*y,X)

    def predict(self,X):
        return np.sign(np.dot(X,self.w)+self.b)



if __name__ == '__main__':
    X = np.array([[3,3],[4,3],[1,1]])
    y = np.array([1,1,-1])

    from sklearn.datasets import load_iris
    from sklearn.metrics import f1_score,roc_auc_score,accuracy_score
    data = load_iris()
    # X = data['data']
    # y = [1 if i == 1 else -1 for i in data['target'] if i ]

    perc = PerceptronDuality()
    perc.fit(X,y,learn_rate=0.01)
    print(perc.w,perc.b)
    y_hat = perc.predict(X)
    print(y_hat)
    print(f1_score(y_hat,y))
    print(accuracy_score(y_hat,y))
