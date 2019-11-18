print(__doc__)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
#####################################################################################

import numpy as np
from sklearn import metrics

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

###############################################################################

# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(data)

	from sklearn.mixture import GMM
gmm = GMM(n_components=4).fit(reduced_data)
labels1 = gmm.predict(reduced_data)
#plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, s=40, cmap='viridis');


gmm1 = GMM(n_components=10).fit(reduced_data)
y=gmm1.predict(reduced_data)
print('-----',(purity_score(labels, y)))
###########################################################################
from matplotlib.patches import Ellipse
'''
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
   
                          angle, **kwargs))'''


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    #w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covars_, gmm.weights_):
        #draw_ellipse(pos, covar, alpha=w * w_factor)
        plt.show()
list =[2,3,5,8,10,12,15,17]
for i in list:


    gmm = GMM(n_components=i, random_state=42)
    print("for i : " , i)
    plot_gmm(gmm, reduced_data)
###########################################################################
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

l=[]

for n_cluster in list:
    kmeans = KMeans(n_clusters=n_cluster).fit(reduced_data)
    label = kmeans.labels_
    sil_coeff = silhouette_score(reduced_data, label, metric='euclidean')
    #pprint("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
    l.append(sil_coeff)
plt.plot(list, l)
plt.show()
