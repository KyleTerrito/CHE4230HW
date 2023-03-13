import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
#DR Methods
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
#import umap
import pacmap
#import trimap
#Clustering Methods
from sklearn.cluster import KMeans, DBSCAN, MeanShift, Birch, estimate_bandwidth
#import HDBSCAN
from scipy.spatial.distance import pdist, squareform

#DATA PREPROCESSING
#Load Data
df = pd.read_excel('C:\\Users\kterri3\Documents\Git Repositories\hw\CHE4230HW\che4230_hw\Project Stuff\RawData.xlsx')
    # print(df)
# Fill empty data with mean values
df.describe()
df.shape
df.isnull().sum().sum()
df_mean = df.fillna(df.mean(), inplace= True)
df.isnull().sum().sum()
df_drop_dup = df.drop_duplicates()
df.fillna(0)
# Train data
x_train, x_test = train_test_split(df, test_size = 0.2, random_state=42)
x_train2, x_val = train_test_split(x_train, test_size = 0.2, random_state=42)
    #print(len(x_train2))
    #print(len(x_test))
    #print(len(x_val))
scaler = StandardScaler().fit(x_train2)
df_scaled = scaler.transform(x_train2)
df_scaled

#PCA with Clustering
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)
plt.scatter(df_pca[:,0], df_pca[:,1])
plt.title("Principal Component Analysis")
#PCA and KMEANS
pca_kmean_model = KMeans(n_clusters = 11)
pca_kmean_model.fit(df_pca)
pca_kmean_labels = pca_kmean_model.labels_
centroids = pca_kmean_model.cluster_centers_
print(pca_kmean_labels)
    # colors for plotting
colors = ['blue', 'red', 'green', 'orange', 'black', 'brown','teal','silver', 'yellow','purple','pink']
    # assign a color to each features (note that we are using features as target)
features_colors = [ colors[pca_kmean_labels[i]] for i in range(len(df_pca)) ]
plt.scatter(df_pca[:,0], df_pca[:,1],c=features_colors,marker='o')
plt.title("PCA and KMeans")
plt.show()

#tSNE with Clustering
tsne = TSNE(n_components = 2)
df_tsne = tsne.fit_transform(df_scaled)
plt.scatter(df_tsne[:,0], df_tsne[:,1])
plt.title("tSNE")
plt.show()
#tSNE and KMEANS
tsne_kmean_model = KMeans(n_clusters = 11)
tsne_kmean_model.fit(df_tsne)
tsne_kmean_labels = tsne_kmean_model.labels_
centroids = tsne_kmean_model.cluster_centers_
print(tsne_kmean_labels)
    # colors for plotting
colors = ['blue', 'red', 'green', 'orange', 'black', 'brown','teal','silver', 'yellow','purple','pink']
    # assign a color to each features (note that we are using features as target)
features_colors = [ colors[tsne_kmean_labels[i]] for i in range(len(df_tsne)) ]
plt.scatter(df_tsne[:,0], df_tsne[:,1],c=features_colors,marker='o')
plt.title("tSNE and KMeans")
plt.show()

#ICA and Clustering
ica = FastICA(n_components = 2)
df_ica = ica.fit_transform(df_scaled)
plt.scatter(df_ica[:,0], df_ica[:,1])
plt.title("ICA")
plt.show()
#ICA and KMEANS
ica_kmean_model = KMeans(n_clusters = 11)
ica_kmean_model.fit(df_tsne)
ica_kmean_labels = ica_kmean_model.labels_
centroids = ica_kmean_model.cluster_centers_
print(ica_kmean_labels)
    # colors for plotting
colors = ['blue', 'red', 'green', 'orange', 'black', 'brown','teal','silver', 'yellow','purple','pink']
    # assign a color to each features (note that we are using features as target)
features_colors = [ colors[ica_kmean_labels[i]] for i in range(len(df_ica)) ]
plt.scatter(df_ica[:,0], df_ica[:,1],c=features_colors,marker='o')
plt.title("ICA and KMeans")
plt.show()

#PACMAP and Clustering
pacmap = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0)
df_pacmap = pacmap.fit_transform(df_scaled)
plt.scatter(df_pacmap[:,0], df_pacmap[:,1])
plt.title("PACMAP")
plt.show()
#PACMAP and KMEANS
pacmap_kmean_model = KMeans(n_clusters = 5)
pacmap_kmean_model.fit(df_pacmap)
pacmap_kmean_labels = pacmap_kmean_model.labels_
centroids = pacmap_kmean_model.cluster_centers_
print(pacmap_kmean_labels)
    # colors for plotting
colors = ['blue', 'red', 'green', 'orange', 'black', 'brown','teal','silver', 'yellow','purple','pink']
    # assign a color to each features (note that we are using features as target)
features_colors = [ colors[pacmap_kmean_labels[i]] for i in range(len(df_pacmap)) ]
plt.scatter(df_pacmap[:,0], df_pacmap[:,1],c=features_colors,marker='o')
plt.title("PaCMAP and KMeans")
plt.show()
#PACMAP and DBSCAN
pacmap_dbscan_model = DBSCAN()
pacmap_dbscan_model.fit(df_pacmap)
pacmap_dbscan_labels = pacmap_dbscan_model.labels_
    #centroids = pacmap_dbscan_model.cluster_centers_
print(pacmap_dbscan_labels)
    # colors for plotting
colors = ['blue', 'red', 'green', 'orange', 'black', 'brown','teal','silver', 'yellow','purple','pink']
    # assign a color to each features (note that we are using features as target)
features_colors = [ colors[pacmap_dbscan_labels[i]] for i in range(len(df_pacmap)) ]
plt.scatter(df_pacmap[:,0], df_pacmap[:,1],c=features_colors,marker='o')
plt.title("PaCMAP and DBSCAN")
plt.show()