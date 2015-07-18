print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color
import urllib

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import AffinityPropagation
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import pandas as pd
import os.path


np.random.seed(42)

num_stocks = 500
max_clusters = 5
col1 = 1 # 1 = % change, 2 = variation, 3 = adjusted price
col2 = 2 # "
folder = "finaldata"
onePointPerCompany = True # Create One point using Mean
chartTitle = 'K-means clustering (5 Years, Daily)\n Centroids are marked with white cross (5 clusters)'
xaxisLabel = '% Change'
yaxisLabel = 'Variation'
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .001     # point in the mesh [x_min, m_max]x[y_min, y_max].

test2 = None
f = open('sp500-symbol-list.txt')
i = 1
for line in f.readlines():
	line = line.replace('\n','')
	if not os.path.exists(folder + "/" + line + ".csv"): # check if file exists
	    print folder + "/" + line + ".csv Not Found"
	    continue
        if test2 is None: # setup for first file
            test = pd.read_csv(folder + "/" + line + ".csv")
            test2 = np.array(test)
            target = np.array([test2[0, 4]]) # add ticker to array
            test2 = test2[:, [col1,col2]] # get only col1 and col2
            l = [[np.mean(test2, axis=0)]] # add mean to list for plotting
            if onePointPerCompany:
                test2 = [np.mean(test2, axis=0)] # calculate mean
            continue
	if i >= num_stocks: # for restricting a number of stocks
		break
	temp = pd.read_csv(folder + "/" + line + ".csv") # read file
	temp = np.array(temp)
        target = np.concatenate((target, [temp[0, 4]]), axis=0) # add ticker to target list
	temp2 = temp[:, [col1,col2]] # get only col1 and col2
	l.append([np.mean(temp2, axis=0)]) # add mean to list for plotting
	if onePointPerCompany:
	   test2 = np.concatenate((test2, [np.mean(temp2, axis=0)]), axis=0) # calculate mean
	else:
	   test2 = np.concatenate((test2, temp2), axis=0) # concat each file into the array
	i+=1

print target

data = test2
#n_samples, n_features = data.shape
n_digits = max_clusters #len(np.unique(target))
#labels = target

# run kmeans
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
meanshift = MeanShift()
affprop = AffinityPropagation()
kmeans.fit(data)

print data.shape


# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')
# plot data used for calculation
plt.plot(data[:, 0], data[:, 1], 'k.', markersize = 2, color='k')
# plot one point per stock
j = 0
while True:
    if j >= i:
        break
    for c in color.cnames:
        if j >= i:
            break
        datacluster = np.array(l.pop());
        plt.plot(datacluster[:, 0], datacluster[:, 1], 'k.', markersize=5, color='k')
        # uncomment for labels
        #plt.text(datacluster[:, 0], datacluster[:, 1], target[target.shape[0]-j-1])
        j += 1
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title(chartTitle)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel(xaxisLabel)
plt.ylabel(yaxisLabel)

f.close()

plt.show()