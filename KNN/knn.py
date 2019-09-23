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


class kdNode:
    def __init__(self,data):
        self.data  = data
        self.left  = None
        self.right = None

class kdTree:
    def __init__(self,X):
        self.X = X
        self.start_dim = self.set_root()
        self.root = self.build_tree(self.X,self.start_dim)

    def set_root(self):
        start_dim = 0
        var = np.std(self.X[:,start_dim])
        for col in range(1,self.X.shape[1]):
            col_var = np.std(self.X[:,col])
            if col_var > var:
                start_dim = col
                var = col_var
        return start_dim
    

    def build_tree(self, kdData, dim):
        if len(kdData) == 0:
            return None

        kdData = kdData[kdData[:, dim].argsort()]

        mid_num = kdData.shape[0]//2
        root = kdNode(kdData[mid_num])

        new_dim = 0 if dim == kdData.shape[1]-1 else dim+1

        root.left = self.build_tree(kdData[:mid_num],new_dim)
        root.right = self.build_tree(kdData[mid_num+1:],new_dim)

        return root

    def find_leaf(self,x0):

        start_node = self.root
        start_dim  = self.start_dim
        length     = len(x0)
        while start_node.right or start_node.left:
           if x0[start_dim] <= start_node.data[start_dim]:
               start_node = start_node.left
           else:
               start_node = start_node.right
           start_dim = 0 if start_dim == length-1 else start_dim+1
        return start_node


class kdKNN:
    def __init__(self,X,y,k=7):
        self.X = X
        self.y = y
        self.k = k

    def build_kd_tree(self):
        pass


    @staticmethod
    def distance(x0,x1):
        return np.sqrt(np.sum(np.abs(x0-x1)**2))

    def predict(self,x):
        pass

if __name__ == '__main__':

    # from sklearn.datasets import load_iris
    # from sklearn.metrics import f1_score,roc_auc_score,accuracy_score
    # data = load_iris()
    # X = data['data']
    # y = data['target']
    # knn = KNN(X,y,k=10)
    # print(knn.predict(X[0]))
    # y_hat = [knn.predict(x) for x in X]
    # print(accuracy_score(y_hat,y))
    #
    data = [
        [7,2],
        [5,4],
        [9,6],
        [2,3],
        [4,7],
        [8,1]
    ]
    a = np.array(data)
    kdt = kdTree(a)

    node = kdt.root
    l = [kdt.root]
    while l:
        node = l.pop()
        if node.left:
            l.insert(0,node.left)
        if node.right:
            l.insert(0,node.right)
        print(node.data)

    near_leaf = kdt.find_leaf([6,1])
    print(near_leaf.data)
