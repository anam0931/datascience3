print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(42)

digits = load_digits()
data = scale(digits.data)


n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))
#######################################################################
import numpy as np
#from sklearn import metrics

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

###############################################################################

reduced_data = PCA(n_components=2).fit_transform(data)

###########################################################################
#printing agglomerative clustering
from sklearn.cluster import AgglomerativeClustering

clusterAgglo = AgglomerativeClustering(n_clusters=10, affinity='euclidean', linkage='ward')
y=clusterAgglo.fit_predict(reduced_data)

print(clusterAgglo.labels_)
print("purity score is : ",purity_score(labels,y))
plt.scatter(reduced_data[:,0],reduced_data[:,1], c=clusterAgglo.labels_, cmap='rainbow')
plt.show()
#################################################################################
#printing debscan clustering

from sklearn.cluster import DBSCAN
'''
clustering = DBSCAN(eps=0.95, min_samples=1)
clustering.fit_predict(reduced_data)
print(clustering.labels_)

plt.scatter(reduced_data[:,0], reduced_data[:,1], c=clustering.labels_, cmap='rainbow')
plt.show()
'''

EPS=[0.05, 0.5, 0.95]
MIN_SAMPLE=[1, 10, 30, 50]

for i in EPS:
    for j in MIN_SAMPLE:
        clusterDBS = DBSCAN(eps=i, min_samples=j)
        pred=clusterDBS.fit_predict(reduced_data)
        print(clusterDBS.labels_)
        print("for eps and min_samples: ",i,j)
        print("purity score is : ", purity_score(labels, pred))
        plt.scatter(reduced_data[:,0], reduced_data[:,1], c=clusterDBS.labels_, cmap='rainbow')
        #plt.text(0,0,r'$\ep=i,\ \sigma=15$')
        plt.show()


#####################################################################################




#####################################################################################

################################################################################
from scipy.spatial.distance import cdist
def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
    labels = kmeans.fit_predict(X)

    # plot the input data
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)

    # plot the representation of the KMeans model
    centers = kmeans.cluster_centers_
    radii = [cdist(X[labels == i], [center]).max()
             for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))
        plt.show()

kmeans = KMeans(init='k-means++', n_clusters=10)
kmeans.fit(reduced_data)
y11=kmeans.predict(reduced_data)
kmeans = KMeans(n_clusters=i, random_state=0)
#plot_kmeans(kmeans, reduced_data, n_clusters=10)
print("--------------" ,purity_score(labels, y11))
######################################################################














