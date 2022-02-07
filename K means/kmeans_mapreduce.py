# https://stackoverflow.com/questions/31761310/hadoop-streaming-with-python-k-means

# https://github.com/kfragkedaki/hadoop-kmeans


# https://www.cnblogs.com/chaoku/p/3748456.html

# https://stanford.edu/~rezab/classes/cme323/S16/projects_reports/bodoia.pdf

'''
https://www.cnblogs.com/chaoku/p/3748456.html

最近在网上查看用MapReduce实现的Kmeans算法，例子是不错，http://blog.csdn.net/jshayzf/article/details/22739063

但注释太少了，而且参数太多，如果新手学习的话不太好理解。所以自己按照个人的理解写了一个简单的例子并添加了详细的注释。

大致的步骤是：

1，Map每读取一条数据就与中心做对比，求出该条记录对应的中心，然后以中心的ID为Key，该条数据为value将数据输出。

2，利用reduce的归并功能将相同的Key归并到一起，集中与该Key对应的数据，再求出这些数据的平均值，输出平均值。

3，对比reduce求出的平均值与原来的中心，如果不相同，这将清空原中心的数据文件，将reduce的结果写到中心文件中。（中心的值存在一个HDFS的文件中）

     删掉reduce的输出目录以便下次输出。

     继续运行任务。

4，对比reduce求出的平均值与原来的中心，如果相同。则删掉reduce的输出目录，运行一个没有reduce的任务将中心ID与值对应输出。


Do 

- Map
Input is a data point and k centers are broadcasted
Finds the closest center among k centers for the input point

- Reduce
Input is one of k centers and all data points having this center as their closest center. Calculates the new center using data points

Until all of new centers are not changed


input: key-value
  [centroid1, point1]
  [centroid2, point3]
  ...
  [centroid10, point100]

  (mapping ) => 
  [centroid1: p1, p2, p5, p10]
  [centroid2: p3, p4, p6]
  ...

  combine => 
  [centroid1, partialsum1, . . . ]
  [centroid2, partialsum1, . . . ]
  ...

  reduce =>
  [centroid1, centroid1’]
  [centroid2, centroid2’]



Stanford:

Standard k-means
1. Choose k initial means μ1, . . . , μk uniformly at random from the set X. 
2. Foreachpointx∈X,findtheclosestmeanμi andaddxtoasetSi. 
3. For i = 1,...,k, set μi to be the centroid of the points in Si.
4. Repeat steps 2 and 3 until the means have converged.

K-MEANS with MapReduce
1. Choose k initial means μ1, . . . , μk uniformly at random from the set X. 
2. Apply the MapReduce given by k-meansMap and k-meansReduce to X. 
3. Compute the new means μ1 , . . . , μk from the results of the MapReduce. 
4. Broadcast the new means to each machine on the cluster.
5. Repeat steps 2 through 4 until the means have converged.


'''


import numpy as np
from scipy.spatial.distance import cdist 

# def euclidean_distance(x1, x2):
#     return np.sqrt(np.sum((x1 - x2) ** 2))


class KMeans:
    def __init__(self, X, k=2, max_iters=100):
        self.X = X 
        self.k = k 
        self.max_iters = max_iters

    def fit(self):
        '''
        1. random select k points as starting centroids, with no replacement
        2. calculate dist between each point and assign it to cloest centroids
        3. for each group, recalculate the centroids
        4. repeat 2-3 until it converges
        '''

        idx = np.random.choice(len(self.X), self.k, replace=False)
        centroids = self.X[idx, :]

        distances = cdist(self.X, centroids, 'euclidean')
        points = np.argmin(distances, axis=1) # labels

        for step in range(self.max_iters):
            prev_points = points
            centroids = []
            for label in range(self.k):
                sub_group = self.X[points == label]
                centroids.append(sub_group.mean(axis=0)) # mean on all points

            centroids = np.vstack(centroids)
            distances = cdist(self.X, centroids, 'euclidean')
            points = np.argmin(distances, axis=1) # labels
            if np.array_equal(prev_points, points):
                print('early stop at step {}'.format(step))
                break

        print('final labels:', points)
        # print('final centroids', centroids)
        self.centroids = centroids

        return points

    def predict(self, x):
        distances = cdist(x, self.centroids, 'euclidean')
        labels = np.argmin(distances, axis=1)
        print('predicted labels:', labels)
        return labels




X = np.array([[1, 2], [1, 4], [1, 0], [1, 2], [10, 5],[10, 2], [10, 4], [10, 0]])
kmeans = KMeans(X)
kmeans.fit()

kmeans.predict([[0, 0], [12, 3]])



# from sklearn.datasets import load_digits
# data = load_digits().data
# kmeans2 = KMeans(data, 10)

# label = kmeans2.fit()
# print(len(label))

