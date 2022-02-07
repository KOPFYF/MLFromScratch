#Loading the required modules
# https://www.askpython.com/python/examples/k-means-clustering-from-scratch
# https://medium.com/@rishit.dagli/build-k-means-from-scratch-in-python-e46bf68aa875
 
import numpy as np
from scipy.spatial.distance import cdist 
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
 
#Defining our function 
def kmeans(x, k, no_of_iterations):
    '''
    X: multidimensional data
    k: number of clusters
    max_iterations: number of repetitions before clusters are established
    
    Steps:
    1. Convert data to numpy aray
    2. Pick indices of k random point without replacement
    3. Find class (P) of each data point using euclidean distance
    4. Stop when max_iteration are reached of P matrix doesn't change
    
    Return:
    np.array: containg class of each data point
    '''
    print('x.shape:', x.shape) # (1797, 2), 1797 points in 2d dimension

    idx = np.random.choice(len(x), k, replace=False)
    print('idx.shape:', idx.shape) # (10,), pick 10 points as centroids

    # Randomly choosing Centroids 
    centroids = x[idx, :] #Step 1
    print('centroids.shape:', centroids.shape) # (10, 2), 10 centroids points in 2d dimension
     
    # finding the distance between centroids and all the data points
    # cdist: Compute distance between each pair of the two collections of inputs
    distances = cdist(x, centroids ,'euclidean') #Step 2
    print('distances.shape:', distances.shape) # (1797, 10), 1797 points cross 10 centroids
     
    #Centroid with the minimum Distance
    points = np.array([np.argmin(i) for i in distances]) #Step 3
    print('points.shape:', points.shape) # (1797,) # centroid id for each point (1797 points)
     
    #Repeating the above steps for a defined number of iterations
    #Step 4
    for _ in range(no_of_iterations): 
        centroids = []
        for idx in range(k): # centroid id from 0 to k - 1
            #Updating Centroids by taking mean of Cluster it belongs to
            temp_cent = x[points == idx].mean(axis=0) 
            centroids.append(temp_cent)
 
        # stack arrays in sequence vertically 
        centroids = np.vstack(centroids) #Updated Centroids 
         
        distances = cdist(x, centroids ,'euclidean')
        points = np.array([np.argmin(i) for i in distances])
         
    return points 
 
 
#Load Data
data = load_digits().data
print('data.shape:', data.shape) # (1797, 64)

pca = PCA(2)
  
#Transform the data
df = pca.fit_transform(data)
 
#Applying our function
label = kmeans(df, 10, 1000)
 
#Visualize the results
 
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
plt.legend()
plt.show()