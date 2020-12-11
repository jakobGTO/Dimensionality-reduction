import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#read data
df = pd.read_csv('zoo.data',names=['animal name','hair','feathers','eggs','milk',
                                    'airborne','aquatic','predator','toothed',
                                    'backbone','breathes','venomous','fins',
                                    'legs','tail','domestic','catsize','type'])
features = ['hair','feathers','eggs','milk',
            'airborne','aquatic','predator','toothed',
            'backbone','breathes','venomous','fins',
            'legs','tail','domestic','catsize']

x = df.loc[:,features]
y = df.loc[:,'type']

#Standardize numerical feature
x = x - x.mean()
x.loc[:,'legs'] = x.loc[:,'legs'] / x.loc[:,'legs'].std()

#Apply pca
pca = PCA()
principalComponents = pca.fit_transform(x)

#Plot log of singular values

#plt.plot(np.log(pca.singular_values_),'bo')
#plt.show()

#Reapply with 5 components
pca = PCA(n_components=5)
principalComponents = pca.fit_transform(x)

#variance % explained
sum(pca.explained_variance_ratio_)

loadingsDf = pd.DataFrame(pca.components_.T, columns = ['PCA1','PCA2','PCA3','PCA4','PCA5'], index=list(x.columns.values))
print(loadingsDf.to_latex(index=True))
#pca DF
principalDf = pd.DataFrame(data = principalComponents, columns = ['PCA1','PCA2','PCA3','PCA4','PCA5'])

finalDf = pd.concat([principalDf, y], axis=1)

#Plot components with color for type, plot every combination of components

#Recreation of plots
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('PCA', fontsize = 20)
targets = [1,2,3,4,5,6,7]
colors = ['r', 'g', 'b', 'c','m','y','k']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['type'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'PCA1']
               , finalDf.loc[indicesToKeep, 'PCA2']
               , c = color
               , s = 50)
ax.legend(targets)
fig.show()

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 3', fontsize = 15)
ax.set_title('PCA', fontsize = 20)
targets = [1,2,3,4,5,6,7]
colors = ['r', 'g', 'b', 'c','m','y','k']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['type'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'PCA1']
               , finalDf.loc[indicesToKeep, 'PCA3']
               , c = color
               , s = 50)
ax.legend(targets)
fig.show()

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 4', fontsize = 15)
ax.set_title('PCA', fontsize = 20)
targets = [1,2,3,4,5,6,7]
colors = ['r', 'g', 'b', 'c','m','y','k']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['type'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'PCA1']
               , finalDf.loc[indicesToKeep, 'PCA4']
               , c = color
               , s = 50)
ax.legend(targets)
fig.show()

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 5', fontsize = 15)
ax.set_title('PCA', fontsize = 20)
targets = [1,2,3,4,5,6,7]
colors = ['r', 'g', 'b', 'c','m','y','k']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['type'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'PCA1']
               , finalDf.loc[indicesToKeep, 'PCA5']
               , c = color
               , s = 50)
ax.legend(targets)
fig.show()

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 2', fontsize = 15)
ax.set_ylabel('Principal Component 3', fontsize = 15)
ax.set_title('PCA', fontsize = 20)
targets = [1,2,3,4,5,6,7]
colors = ['r', 'g', 'b', 'c','m','y','k']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['type'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'PCA2']
               , finalDf.loc[indicesToKeep, 'PCA3']
               , c = color
               , s = 50)
ax.legend(targets)
fig.show()

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 2', fontsize = 15)
ax.set_ylabel('Principal Component 4', fontsize = 15)
ax.set_title('PCA', fontsize = 20)
targets = [1,2,3,4,5,6,7]
colors = ['r', 'g', 'b', 'c','m','y','k']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['type'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'PCA2']
               , finalDf.loc[indicesToKeep, 'PCA4']
               , c = color
               , s = 50)
ax.legend(targets)
fig.show()

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 2', fontsize = 15)
ax.set_ylabel('Principal Component 5', fontsize = 15)
ax.set_title('PCA', fontsize = 20)
targets = [1,2,3,4,5,6,7]
colors = ['r', 'g', 'b', 'c','m','y','k']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['type'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'PCA2']
               , finalDf.loc[indicesToKeep, 'PCA5']
               , c = color
               , s = 50)
ax.legend(targets)
fig.show()

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 3', fontsize = 15)
ax.set_ylabel('Principal Component 4', fontsize = 15)
ax.set_title('PCA', fontsize = 20)
targets = [1,2,3,4,5,6,7]
colors = ['r', 'g', 'b', 'c','m','y','k']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['type'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'PCA3']
               , finalDf.loc[indicesToKeep, 'PCA4']
               , c = color
               , s = 50)
ax.legend(targets)
fig.show()

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 3', fontsize = 15)
ax.set_ylabel('Principal Component 5', fontsize = 15)
ax.set_title('PCA', fontsize = 20)
targets = [1,2,3,4,5,6,7]
colors = ['r', 'g', 'b', 'c','m','y','k']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['type'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'PCA3']
               , finalDf.loc[indicesToKeep, 'PCA5']
               , c = color
               , s = 50)
ax.legend(targets)
fig.show()

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 4', fontsize = 15)
ax.set_ylabel('Principal Component 5', fontsize = 15)
ax.set_title('PCA', fontsize = 20)
targets = [1,2,3,4,5,6,7]
colors = ['r', 'g', 'b', 'c','m','y','k']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['type'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'PCA4']
               , finalDf.loc[indicesToKeep, 'PCA5']
               , c = color
               , s = 50)
ax.legend(targets)
fig.show()
