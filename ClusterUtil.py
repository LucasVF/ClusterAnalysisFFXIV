#Plotting With Parallel Coordinates
import seaborn as sns
import pandas as pd
import Constants
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import sklearn.preprocessing
import DataReduction

from DataReduction import ClusterCharacter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pandas.plotting import parallel_coordinates

from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
from math import isnan
from pyclustertend import hopkins
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

def plotParallel(clusteredCharacters,clusterName):
	cols =Constants.NAME_COLUMNS
	df = pd.DataFrame(clusteredCharacters,columns=Constants.NAME_COLUMNS+['Cluster'])
	
	# Scaling attribute values to avoid few outiers
	subset_df = df[cols]

	
	ss = StandardScaler()

	scaled_df = ss.fit_transform(subset_df)
	scaled_df = pd.DataFrame(scaled_df, columns=cols)
	final_df = pd.concat([scaled_df, df['Cluster']], axis=1)
	final_df.head()

	# plot parallel coordinates
	
	pc = parallel_coordinates(final_df, 'Cluster', color=('#FFE888', '#FF9999','#3433FF'))
	plt.title('Parallel Coordinates for '+clusterName+' ('+Constants.PLAYERBASE_SIZE+' Characters)')
	plt.show()
    
def adjustParameters(plt):
	plt.rcParams['figure.figsize'] = (20,10)
	plt.rcParams['figure.subplot.right'] = 0.98
	plt.rcParams['axes.labelsize'] = 20
	plt.rcParams['xtick.labelsize'] = 20
	plt.rcParams['ytick.labelsize'] = 20
	plt.rcParams['axes.labelsize'] = 20
	plt.rcParams['axes.titlesize'] = 20
    
def plotHeatMap(arrayNormalizedCharacters,idName):
#Using Pearson Correlation
	#Creating of HeatMap to Evaluate Correlation
	dataFrame = pd.DataFrame(arrayNormalizedCharacters,columns=Constants.NAME_COLUMNS)

	adjustParameters(plt)

	plt.figure()
	cor = dataFrame.corr()
	ax = sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
	ax.set_ylim(Constants.NUMBER_ATRIBUTES, 0)
	plt.tight_layout()
	plt.savefig('HeatMap'+str(idName)+'.png')
	
	
def plotElbow(arrayNormalizedCharacters,idName):
	#Elbow Method to determine k Test	
	wcss = []
	adjustParameters(plt)
	plt.rcParams['figure.figsize'] = (10,10)
	for i in range(1, 11):
		kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
		kmeans.fit(arrayNormalizedCharacters)
		wcss.append(kmeans.inertia_)
		
	plt.figure()
	plt.plot(range(1, 11), wcss)
	plt.title('Elbow Method')
	plt.xlabel('Number of clusters')
	plt.ylabel('WCSS')
	plt.tight_layout()
	plt.savefig('Elbow'+str(idName)+'.png')
	
def getDescriptions(input,clusterName,month,year):
	for i in range(0,len(input)):
		getDescription(input[i],"Cluster "+str(i+1)+" criado pela Clusterização "+clusterName,month,year)
	
def getDescription(input,desc,month,year):
	sumCharactersT = np.array(input,dtype='int64')

	sns.set_style('darkgrid')
	sns.set(font_scale=1.5)

	df = pd.DataFrame(sumCharactersT,columns=Constants.RESUMED_COLUMNS)
	describedDf = df.describe()

	describedDf.update(describedDf['FCom'].apply(lambda x: str(round(float(x), 4))))
	describedDf.update(describedDf['PMExp'].apply(lambda x: str(int(x))))
	describedDf.update(describedDf['PRExp'].apply(lambda x: str(int(x))))
	describedDf.update(describedDf['TExp'].apply(lambda x: str(int(x))))
	describedDf.update(describedDf['MRExp'].apply(lambda x: str(int(x))))
	describedDf.update(describedDf['HExp'].apply(lambda x: str(int(x))))
	describedDf.update(describedDf['ArcExp'].apply(lambda x: str(int(x))))
	describedDf.update(describedDf['CrExp'].apply(lambda x: str(int(x))))
	describedDf.update(describedDf['GaExp'].apply(lambda x: str(int(x))))
	describedDf.update(describedDf['mmCPTime'].apply(lambda x: str(round(float(x), 3))))
	describedDf.update(describedDf['PreHw'].apply(lambda x: str(round(float(x), 4))))
	describedDf.update(describedDf['Mrrg'].apply(lambda x: str(round(float(x), 4))))
	describedDf.update(describedDf['Hldn'].apply(lambda x: str(round(float(x), 4))))
	describedDf.update(describedDf['BTC'].apply(lambda x: str(int(x))))    

	cols = ['Estatistica','FCom','PMExp','PRExp','TExp','MRExp','HExp','ArcExp','CrExp','GaExp','mmCPTime','PreHw','Hldn','Mrrg','BTC']

	import plotly.offline as offline

	offline.plot({'data': [go.Table(columnwidth = 50,
		header=dict(values=cols,
					fill_color='paleturquoise',
					align='left'),
		cells=dict(values=describedDf.reset_index().values.T,
					fill_color='lavender',
					align='left')
	)],'layout': {'title': "Descrição do "+desc+" do mes "+month+"/"+year}},filename=desc+month+year+'.html')
 
def hopkins(input):

	sumCharactersT = np.array(input,dtype='int64')
	X = pd.DataFrame(sumCharactersT,columns=Constants.NAME_COLUMNS)
    
	d = X.shape[1]
	#d = len(vars) # columns
	n = len(X) # rows
	m = int(0.1 * n) # heuristic from article [1]
	nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
	 
	rand_X = sample(range(0, n, 1), m)
	 
	ujd = []
	wjd = []
	for j in range(0, m):
		u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
		ujd.append(u_dist[0][1])
		w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
		wjd.append(w_dist[0][1])
 
	H = sum(ujd) / (sum(ujd) + sum(wjd))
	if isnan(H):
		H = H + 0
	return H
	
def getNormalizedAndScaler(tupleCharactersAtributes):
	scaler = MinMaxScaler()
	scaler.fit(tupleCharactersAtributes)
	normalizedCharacters = scaler.transform(tupleCharactersAtributes)
	return (normalizedCharacters,scaler)
	
	
def getTupleCharactersAtributes(clusterCharacters,resumed):
	tupleCharactersAtributes = []

	for clusterCharacter in clusterCharacters:	
		if(resumed):
			tupleCharacterAtributes = DataReduction.getTuple(clusterCharacter,Constants.HIGH_CORRELATED_COLUMNS)
		else:
			tupleCharacterAtributes = DataReduction.getTuple(clusterCharacter,['race','gender','free_company','realm','grand_company','legacy_player','physicalMeeleExp','physicalRangeExp','tankExp','magicRangeExp','healerExp','arcanistExp','craftExp','gatherExp','minimalCertainPaidTime','hildibrand','preArr', 'preHw', 'physicalObject', 'marriage', 'beastTribesCompleted'])

		tupleCharactersAtributes.append(tupleCharacterAtributes)
		
	return tupleCharactersAtributes
	
def printSilhouette(arrayNormalizedCharacters,labels, name):
	if(max(labels)==0):
		print("Não foi possível calcular o valor da Silhueta para {} pois apenas 1 cluster foi formado\n".format(name))
	else:
		df = pd.DataFrame(arrayNormalizedCharacters,columns=Constants.HIGH_CORRELATED_COLUMNS)
		score = silhouette_score (df,labels, metric='euclidean')
		print ("\nO valor de Silhueta para {} é {}\n".format(name, score))
		
def printDavies(arrayNormalizedCharacters,labels, name):
	if(max(labels)==0):
		print("Não foi possível calcular o valor do índice de Davies-Bouldin para {} pois apenas 1 cluster foi formado\n".format(name))
	else:
		df = pd.DataFrame(arrayNormalizedCharacters,columns=Constants.HIGH_CORRELATED_COLUMNS)
		score = davies_bouldin_score(df, labels)
		print ("O valor do índice de Davies-Bouldin para {} é: {}\n".format(name, score))
		
def printCalinski(arrayNormalizedCharacters,labels, name):
	if(max(labels)==0):
		print("Não foi possível calcular o valor do índice de Calinski-Harabasz para {} pois apenas 1 cluster foi formado\n".format(name))
	else:
		df = pd.DataFrame(arrayNormalizedCharacters,columns=Constants.HIGH_CORRELATED_COLUMNS)
		score = calinski_harabasz_score(df, labels)
		print ("O valor do índice de Calinski-Harabasz para {} é: {}\n".format(name, score))

def printEvaluations(arrayNormalizedCharacters,labels, name):
	nClusters = max(labels)+1
	print("Numero Total de Clusters: {}".format(nClusters))
	if(nClusters==1):
		print("Não foi possível avaliar {} pois apenas 1 cluster foi formado\n".format(name))
	else:
		printSilhouette(arrayNormalizedCharacters,labels, name)
		printCalinski(arrayNormalizedCharacters,labels, name)
		printDavies(arrayNormalizedCharacters,labels, name)		
		
def plotKnnDistance(X):
	nbrs = NearestNeighbors(n_neighbors=len(X)).fit(X)
	distances, indices = nbrs.kneighbors(X)
	
	lastColumn = []
	for d in distances:
		lastColumn.append(d[len(d)-1])
	
	lastColumn.sort(reverse = True)
	plt.plot(lastColumn)
	plt.show()