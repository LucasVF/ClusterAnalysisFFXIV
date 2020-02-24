import ClusterUtil
import sklearn.preprocessing
import pyclustering

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from pyclustering.cluster.somsc import somsc
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.fcm import fcm

def getCluster(arrayNormalizedCharacters,scaler,clusterName):
	if(clusterName == 'K-means'):
		return kMeans(arrayNormalizedCharacters,scaler,3)
	if(clusterName == 'WARD'):
		return hierarchicalWard(arrayNormalizedCharacters,scaler, 3)
	if(clusterName == 'Spectral'):
		return spectral(arrayNormalizedCharacters,scaler, 3)
	if(clusterName == 'DBSCAN'):
		return dbscan(arrayNormalizedCharacters,scaler)
	if(clusterName == 'BANG'):
		return bang(arrayNormalizedCharacters,scaler)
	if(clusterName == 'SOM'):
		return som(arrayNormalizedCharacters,scaler,3)
	if(clusterName == 'Fuzzy C-Means'):
		return cMeans(arrayNormalizedCharacters,scaler,3)

def kMeans(arrayNormalizedCharacters,scaler, k):
	
	kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
	clustering = kmeans.fit(arrayNormalizedCharacters)
	
	#clusteredCharacters = []
	clustersResult = []
	
	for i in range(0,k):
		clustersResult.append([])
	
	interator = 0
	for playerCluster in clustering.labels_:
		clustersResult[playerCluster].append(arrayNormalizedCharacters[interator])
		#clusteredCharacters.append(list(arrayNormalizedCharacters[interator])+ [playerCluster])
		interator = interator + 1
	
	for i in range(0,k):
		clustersResult[i] = scaler.inverse_transform(clustersResult[i])
	
	#ClusterUtil.plotParallel(clusteredCharacters,'K-means')
	
	return (clustersResult,clustering.labels_)
	
def hierarchicalWard(arrayNormalizedCharacters,scaler, k):
	
	ward = AgglomerativeClustering(n_clusters=k,distance_threshold=None)
	clustering = ward.fit(arrayNormalizedCharacters)
	
	#clusteredCharacters = []
	clustersResult = []
	
	for i in range(0,k):
		clustersResult.append([])
	
	interator = 0
	for playerCluster in clustering.labels_:
		clustersResult[playerCluster].append(arrayNormalizedCharacters[interator])
		#clusteredCharacters.append(list(arrayNormalizedCharacters[interator])+ [playerCluster])
		interator = interator + 1
	
	for i in range(0,k):
		clustersResult[i] = scaler.inverse_transform(clustersResult[i])
	
	#ClusterUtil.plotParallel(clusteredCharacters,'WARD')
	
	return (clustersResult,clustering.labels_)
	
	
	
def spectral(arrayNormalizedCharacters,scaler, k):
	
	clustering = SpectralClustering(n_clusters=k,assign_labels="discretize",random_state=0).fit(arrayNormalizedCharacters)

	#clusteredCharacters = []
	clustersResult = []
	
	for i in range(0,k):
		clustersResult.append([])
	
	interator = 0
	for playerCluster in clustering.labels_:
		clustersResult[playerCluster].append(arrayNormalizedCharacters[interator])
		#clusteredCharacters.append(list(arrayNormalizedCharacters[interator])+ [playerCluster])
		interator = interator + 1
	
	for i in range(0,k):
		clustersResult[i] = scaler.inverse_transform(clustersResult[i])
	
	#ClusterUtil.plotParallel(clusteredCharacters,'Spectral')
	
	return (clustersResult,clustering.labels_)

def dbscan(arrayNormalizedCharacters,scaler):
	
	clustering = DBSCAN(min_samples=24).fit(arrayNormalizedCharacters)

	#clusteredCharacters = []
	clustersResult = []
	noise = 0
	
	k = max(clustering.labels_)
	
	for i in range(0,k+1):
		clustersResult.append([])
	
	interator = 0
	for playerCluster in clustering.labels_:
		if(playerCluster != -1):
			
			clustersResult[playerCluster].append(arrayNormalizedCharacters[interator])
			#clusteredCharacters.append(list(arrayNormalizedCharacters[interator])+ [playerCluster])
		else:
			noise = noise + 1
		interator = interator + 1
	
	print("Número de Pontos classificados como RUÍDO: ", noise,"\n")
	
	for i in range(0,k):
		clustersResult[i] = scaler.inverse_transform(clustersResult[i])
	
	#ClusterUtil.plotParallel(clusteredCharacters,'OPTICS')
	
	return (clustersResult,clustering.labels_)


def cMeans(arrayNormalizedCharacters,scaler,k):
	#from pyclustering.cluster import cluster_visualizer
	
	
	# load list of points for cluster analysis
	data = arrayNormalizedCharacters
	# initialize
	initial_centers = kmeans_plusplus_initializer(data, k, kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE).initialize()
	# create instance of Fuzzy C-Means algorithm
	fcm_instance = fcm(data, initial_centers)
	# run cluster analysis and obtain results
	fcm_instance.process()
	clusters = fcm_instance.get_clusters()
	centers = fcm_instance.get_centers()
	
	clustersResult = []
	k = len(clusters)
	labels = []
	
	for i in range(0,k):
		clustersResult.append([])	
	
	i=0
	for player in arrayNormalizedCharacters:
		j=0
		for cluster in clusters:
			if(i in cluster):
				clustersResult[j].append(arrayNormalizedCharacters[i])
				labels.append(j)
			j= j+1
		i=i+1
	
	for i in range(0,k):
		clustersResult[i] = scaler.inverse_transform(clustersResult[i])
	
	return (clustersResult,labels)
	

def som(arrayNormalizedCharacters,scaler,k):
	
	#from pyclustering.cluster import cluster_visualizer
	
	# Load list of points for cluster analysis
	data = arrayNormalizedCharacters
	# Create instance of SOM-SC algorithm to allocated two clusters
	somsc_instance = somsc(data, k)
	# Run cluster analysis and obtain results
	somsc_instance.process()
	clusters = somsc_instance.get_clusters()
	# Visualize clustering results.
	#visualizer = cluster_visualizer()
	#visualizer.append_clusters(clusters, data)
	#visualizer.show()
	
	clustersResult = []
	k = len(clusters)
	labels = []
	
	for i in range(0,k):
		clustersResult.append([])	
	
	i=0
	for player in arrayNormalizedCharacters:
		j=0
		for cluster in clusters:
			if(i in cluster):
				clustersResult[j].append(arrayNormalizedCharacters[i])
				labels.append(j)
			j= j+1
		i=i+1
	
	for i in range(0,k):
		clustersResult[i] = scaler.inverse_transform(clustersResult[i])
	
	return (clustersResult,labels)
	

def bang(arrayNormalizedCharacters,scaler):
	
	from pyclustering.cluster.bang import bang, bang_visualizer
	
	# Read data three dimensional data.
	data = arrayNormalizedCharacters
	# Prepare algorithm's parameters.
	levels = 11
	# Create instance of BANG algorithm.
	bang_instance = bang(data, levels)
	bang_instance.process()
	# Obtain clustering results.
	clusters = bang_instance.get_clusters()
	#dendrogram = bang_instance.get_dendrogram()
	# Visualize BANG clustering results.
	#bang_visualizer.show_dendrogram(dendrogram)
	
	clustersResult = []
	k = len(clusters)
	labels = []
	
	for i in range(0,k):
		clustersResult.append([])	
	
	i=0
	for player in arrayNormalizedCharacters:
		j=0
		for cluster in clusters:
			if(i in cluster):
				clustersResult[j].append(arrayNormalizedCharacters[i])
				labels.append(j)
			j= j+1
		i=i+1
	
	for i in range(0,k):
		clustersResult[i] = scaler.inverse_transform(clustersResult[i])
	
	return (clustersResult,labels)