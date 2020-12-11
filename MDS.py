import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso

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

#Standardize numerical feature
x.loc[:,'legs'] = (x.loc[:,'legs'] - x.loc[:,'legs'].mean()) / x.loc[:,'legs'].std()

# Infer feature importance
model = Lasso(alpha=0.033)
model.fit(x.values,target.values)

# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# evaluate model
scores = cross_val_score(model, x.values, target.values, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
scores = np.abs(scores)
print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
model.coef_

#MDS for important features
df = pd.read_csv('zoo.data',names=['animal name','hair','feathers','eggs','milk',
                                    'airborne','aquatic','predator','toothed',
                                    'backbone','breathes','venomous','fins',
                                    'legs','tail','domestic','catsize','type'])

features = ['hair','feathers','milk',
            'airborne','aquatic',
            'backbone','breathes',
            'legs','tail','catsize']

x = df.loc[:,features]

target = df.loc[:,'type']

# Compute distance matrix
def euc_distance(x,y):
    return np.sqrt(sum((x-y)**2))

def d_mat(data):
    return np.array([[euc_distance(x, y) for y in data] for x in data])

distance_matrix = d_mat(x.values)

def MDS_func(d, output_dims = 2):
    D = (-0.5 * d**2)

    # get row and colmeans for centering
    row_mean = np.mat(np.mean(D,axis=1))
    col_mean = np.mat(np.mean(D,axis=0))

    # Double centering trick
    S = (np.array(D - np.transpose(row_mean) - col_mean + np.mean(D)))

    # decomposition
    [U, Lambda, temp] = np.linalg.svd(S)

    Y = U * np.sqrt(Lambda)

    return Y[:,0:output_dims], Lambda

X,Lambda = MDS_func(distance_matrix,output_dims=2)

#Plot log of singular values

#plt.plot(np.log(Lambda[0:10]),'bo')
#plt.show()

# Plot mds for 2 dims
target_names = [1,2,3,4,5,6,7]
colors = ['r', 'g', 'b', 'c','m','y','k']
plt.rcParams['figure.figsize'] = [7, 7]
plt.rc('font', size=14)
for i in np.unique(target.values):
    subset = X[target.values == i]

    xaxis = [row[0] for row in subset]
    yaxis = [row[1] for row in subset]
    plt.scatter(xaxis,yaxis,c=colors[i-1],label=target_names[i-1])

plt.legend()
plt.show()
