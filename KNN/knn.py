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
    def __init__(self,data,dim=None,father=None):
        self.data   = data
        self.left   = None
        self.right  = None
        self.father = father
        self.dim    = dim

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
    

    def build_tree(self, kdData, dim,father=None):
        """
        递归构建kdTree
        :param kdData:构建所需要的数据
        :param dim: 开始维度
        :param father: 节点的父节点
        :return:
        """
        if len(kdData) == 0:
            return None

        idxData = kdData[:, dim].argsort()

        mid_idx = idxData.shape[0]//2
        mid_num = idxData[mid_idx]                      # 取得中间位置的点
        root = kdNode(kdData[mid_num],dim,father)                   # 构建节点

        # print(kdData)
        # print('节点:',root.data)
        # print('dim:',root.dim)
        # print('*'*10)
        new_dim = 0 if dim == kdData.shape[1]-1 else dim+1      # 比较维度更改
                                                                # 递归左右子树
        root.left = self.build_tree(kdData[idxData][:mid_idx],new_dim,root)
        root.right = self.build_tree(kdData[idxData][mid_idx+1:],new_dim,root)

        return root

    def find_close_one(self,x0):
        close_node = self.recall_node(root_node=self.root,x0=x0)
        return close_node

    @staticmethod
    def find_leaf(start_node,x0):
        """
        寻找x0所对应的最小叶节点
        :param start_node: 开始的节点
        :param x0: 寻找的数据
        :return:
        """
        # length  = len(x0)
        start_dim = start_node.dim
        while start_node.right or start_node.left:
            start_dim = start_node.dim
            # print(start_node.data,start_dim,x0)
            if x0[start_dim] <= start_node.data[start_dim]:
                start_node = start_node.left
            else:
                start_node = start_node.right
            # start_dim = 0 if start_dim == length-1 else start_dim+1

        return start_node,start_dim


    def recall_node(self,root_node,x0,near_node=None):
        """
        :param root_node: 从哪个节点开始查找
        :param near_node: 目前最近的节点
        :param start_dim: 开始维度
        :param x0:
        :return:
        """

        leaf_node,leaf_dim= self.find_leaf(root_node,x0)
        dl = np.linalg.norm(np.array(x0)-np.array(leaf_node.data))
        # print(leaf_node.data)
        # 向上回溯
        search_node = leaf_node
        search_dim  = leaf_node.dim
        while search_node.father:
            # 确定当前最近节点和距离
            if not near_node:
                near_node = search_node
                d = dl
            else:
                d = np.linalg.norm(np.array(x0)-np.array(near_node.data))
                # if d < dl:
                #     near_node = search_node
            # print(search_node.data,near_node.data)
            if search_node == root_node:
                break
            last_node = search_node
            # last_dim  = search_dim
            # search_dim = search_node
            search_node = search_node.father
            search_dim = search_node.dim
            d_sn = np.linalg.norm(np.array(x0)-np.array(search_node.data))
            if d_sn < d:
                d = d_sn
                near_node = search_node
                # print(near_node.data,d)

            d_circle = np.abs(x0[search_dim] - search_node.data[search_dim])

            # 如果内交就从内交的点开始继续查找
            if d_circle < d:
                branch_node = search_node.right if search_node.left is last_node else search_node.left
                near_node = self.recall_node(branch_node,x0,near_node)

        return near_node


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
    test_data = [9,3]
    near_leaf = kdt.find_close_one(test_data)
    print(near_leaf.data)
    #
    # print(kdt.find_leaf(kdt.root,kdt.start_dim,[6,1])[0].data)

    a=np.linalg.norm(np.array(test_data)-np.array([9,6]))
    print(a)
    b=np.linalg.norm(np.array(test_data)-np.array([8,1]))
    print(b)
