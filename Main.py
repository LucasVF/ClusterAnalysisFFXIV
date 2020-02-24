import DBCon
import DataReduction
import sklearn.preprocessing
import numpy as np
import matplotlib.pyplot as plt
import Constants
import ClusterUtil
import pandas as pd
import Cluster

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import FactorAnalysis
from DataReduction import ClusterCharacter
from pprint import pprint
from datetime import datetime
'''
ChosenAlgorithm = 3

0 = 'K-means'
1 = 'WARD'
2 = 'Spectral'
3 = 'DBSCAN'
4 = 'BANG'
5 = 'SOM'
6 = 'Fuzzy C-Means']
'''

'''
ClusterUtil.getDescription(tupleCharactersAtributes,'Total')
for i in range(7):
	print("================================================================Applying Clustering: ",Constants.CLUSTERS[ChosenAlgorithm],"\n")
	beginTime = time.time()
	#ClusterUtil.plotKnnDistance(arrayNormalizedCharacters)

	(clustersResult,labels) = Cluster.getCluster(arrayNormalizedCharacters,scaler,Constants.CLUSTERS[i])

	timeElapsed = time.time() - beginTime
	print("\nTempo de Pesquisa: ", "{:.5f}".format(timeElapsed)," Seconds")
		
	print("\n================================================================Clustering Applied\n")

	#Analisar caracteristicas de cada Cluster, media, variancia, extremos, etc
	ClusterUtil.getDescriptions(clustersResult,Constants.CLUSTERS[i])

	print("================================================================Statistical Analysis Completed\n")

	#Avaliar qualidade do Cluster
	ClusterUtil.printEvaluations(arrayNormalizedCharacters,labels,Constants.CLUSTERS[i])

	print("================================================================Evaluation Completed\n")
'''

def prepareClusterization(month,year,nullCheck,preProcessing):
    Constants.MONTH = month
    Constants.YEAR = year
    if(preProcessing):
        print("\n\n================================================================Análise para Mes: {} Ano: {}\n".format(Constants.MONTH,Constants.YEAR))

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Hora de Inicio = ", current_time)

    characters = DBCon.getCharacters(nullCheck)
    clusterCharacters = DataReduction.getClusterCharacters(characters)
    #print('Colunas: physicalMeeleExp,physicalRangeExp,tankExp,magicRangeExp,healerExp,arcanistExp,craftExp,gatherExp,minimalCPTime,hildibrand, PreHw,  Marriage\n')	
    tupleCharactersAtributes = ClusterUtil.getTupleCharactersAtributes(clusterCharacters,not preProcessing)


    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Hora da Finalização da Redução de Dimensão = ", current_time)

    print("\n================================================================Redução Completa\n")

    if(preProcessing):
        #Evaluate Cluster Tendency
        print("Estatística de Hopkins: ",ClusterUtil.hopkins(tupleCharactersAtributes),"\n")

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Hora da Finalização da Análise da Estatistica de Hopkins = ", current_time)
        print("\n================================================================Avaliação de Tendência para Clusterização Completa\n")
        
    #Normalizes Data
    (normalizedCharacters,scaler) = ClusterUtil.getNormalizedAndScaler(tupleCharactersAtributes)
    arrayNormalizedCharacters = np.array(normalizedCharacters)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Hora da Finalização da Normalização = ", current_time)
    print("\n================================================================Normalização Completa\n")
    if(preProcessing):
    #Using Pearson Correlation
        ClusterUtil.plotHeatMap(arrayNormalizedCharacters,str(Constants.MONTH)+str(Constants.YEAR))
    
        print("\n================================================================Mapa de Calor Criado\n")

        ClusterUtil.plotElbow(arrayNormalizedCharacters,str(Constants.MONTH)+str(Constants.YEAR))
    
        print("\n================================================================Gráfico do Método do Cotovelo Criado\n")
    
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Hora de Fim = ", current_time)
        print("\n")
    else:
        ClusterUtil.getDescription(tupleCharactersAtributes,'Total',month,year)
        return (arrayNormalizedCharacters,scaler)
    
def clusterize(month,year):
	Constants.MONTH = month
	Constants.YEAR = year
	print("\n\n================================================================Clusterização para Mes: {} Ano: {}\n".format(Constants.MONTH,Constants.YEAR))
    
	(arrayNormalizedCharacters,scaler) = prepareClusterization(month,year,False,False)

	for i in range(7):
		print("================================================================Aplicando Algoritmo de Clusterização: ",Constants.CLUSTERS[i],"\n")		

		(clustersResult,labels) = Cluster.getCluster(arrayNormalizedCharacters,scaler,Constants.CLUSTERS[i])

		now = datetime.now()
		current_time = now.strftime("%H:%M:%S")
		print("Hora de Finalização da Clusterização = ", current_time)
		
		print("\n================================================================Clusterização Aplicada\n")

	    #Analisar caracteristicas de cada Cluster, media, variancia, extremos, etc
		ClusterUtil.getDescriptions(clustersResult,Constants.CLUSTERS[i],Constants.MONTH,Constants.YEAR)

		print("\n================================================================Análise Estatística Aplicada\n")

	    #Avaliar qualidade do Cluster
		ClusterUtil.printEvaluations(arrayNormalizedCharacters,labels,Constants.CLUSTERS[i])

		print("\n================================================================Avaliação Completa\n")
	now = datetime.now()
	current_time = now.strftime("%H:%M:%S")
	print("Hora de Fim = ", current_time)
	print("\n")

'''
prepareClusterization('6','18',True,True)
prepareClusterization('6','19',True,True)
prepareClusterization('10','17',True,True)
prepareClusterization('10','18',True,True)
prepareClusterization('2','18',True,True)
prepareClusterization('2','19',True,True)
'''

clusterize('6','18')
clusterize('6','19')
clusterize('10','17')
clusterize('10','18')
clusterize('2','18')
clusterize('2','19')