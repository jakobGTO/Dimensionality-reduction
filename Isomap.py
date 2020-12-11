import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.graph import graph_shortest_path

df = pd.read_csv('zoo.data',names=['animal name','hair','feathers','eggs','milk',
                                    'airborne','aquatic','predator','toothed',
                                    'backbone','breathes','venomous','fins',
                                    'legs','tail','domestic','catsize','type'])

features = ['hair','feathers','eggs','milk',
            'airborne','aquatic','predator','toothed',
            'backbone','breathes','venomous','fins',
            'legs','tail','domestic','catsize']

x = df.loc[:,features]
target = df.loc[:,'type']

# Compute distance matrix
def euc_distance(x,y):
    return np.sqrt(sum((x-y)**2))

# extended distance matrix, keeping only the k nearest neighbours
def d_mat(data,k):
    distance_matrix = np.array([[euc_distance(x, y) for y in data] for x in data])

    knn_mat = np.zeros_like(distance_matrix)
    temp = np.argsort(distance_matrix, axis=1)[:, 1:k+1]
    for i,j in enumerate(temp):
        knn_mat[i,j] = distance_matrix[i,j]
    print(knn_mat)
    return knn_mat

def MDS_func(d_mat, output_dims = 2):
    D = (-0.5 * d_mat**2)

    # get row and colmeans for centering
    row_mean = np.mat(np.mean(D,axis=1))
    col_mean = np.mat(np.mean(D,axis=0))

    # Double centering trick
    S = np.array(D - np.transpose(row_mean) - col_mean + np.mean(D))

    [U, Lambda, temp] = np.linalg.svd(S)

    Y = U * np.sqrt(Lambda)

    return Y[:,0:output_dims]

def isomap(data,output_dim,k):
    distance_matrix = d_mat(data,k)
    
    graph = graph_shortest_path(distance_matrix,directed=False,method='FW')
    return MDS_func(graph,output_dim)

isomap_output = isomap(data=x.values,output_dim=2,k=24)

# Plot mds for 2 dims
target_names = [1,2,3,4,5,6,7]
colors = ['r', 'g', 'b', 'c','m','y','k']
plt.rcParams['figure.figsize'] = [7, 7]
plt.rc('font', size=14)
for i in np.unique(target.values):
    subset = isomap_output[target.values == i]
    xaxis = [row[0] for row in subset]
    yaxis = [row[1] for row in subset]
    plt.scatter(xaxis,yaxis,c=colors[i-1],label=target_names[i-1])

plt.legend()
plt.show()

