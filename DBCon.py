import mysql
import mysql.connector
import time
from datetime import datetime
import Constants

from mysql.connector import Error

def getCharacters(missing):
	try:
		connection = mysql.connector.connect(host='127.0.0.1',
											 database='ffxivcensus',
											 user='lucasfernandes',
											 password='Megaluc18',
											 auth_plugin='mysql_native_password')
		print("\nConexão MySQL esta aberta")
		
		if(missing):
 			checkMissingValues(connection)
		cursor = connection.cursor()
		print("\nFazendo Pesquisa no Banco de Dados...\n")
		cursor.execute(Constants.QUERYSTORM)
		records = cursor.fetchall()
		print("Mes: {} Ano: {}\n".format(Constants.MONTH,Constants.YEAR))
		print("Número Total de Personagens coletados: ", cursor.rowcount)
		now = datetime.now()
		current_time = now.strftime("%H:%M:%S")
		print("\nHora da Finalização da Pesquisa = ", current_time)
	except Error as e:
		print("Error reading data from MySQL table", e)
	finally:
		if (connection.is_connected()):
			connection.close()
			cursor.close()
			print("\nConexão MySQL está fechada\n")
			print("================================================================Dados Coletados\n")
			if(records != -1):
				return records
			else:
				print("ERROR: Data has recovered -1 characters")
				exit()


def checkMissingValues(connection):
	cursor = connection.cursor()
	try:
		print("\n================================================================Verificando existência de Valores Ausentes\n")		
		cursor.execute(Constants.NULL_QUERY_STORM)
		missingValues = cursor.fetchall()
		now = datetime.now()
		current_time = now.strftime("%H:%M:%S")
		print("Hora da Finalização da Checagem = ", current_time)
	except Error as e:
		print("\nError reading data from MySQL table", e)
	finally:
		if missingValues[0][0] > 0:
			print("\nERROR: Data has null values, please complete missing data")
		else:
			print("\nResultado: Não foi encontrado Valores Ausentes\n")
		print("================================================================Valores Ausentes Verificados\n")
				
	