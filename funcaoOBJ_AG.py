# coding: utf-8
import math
import numpy as np
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import csv
from sklearn.metrics import mean_squared_error

def func_obj(x):
	days = 30
	sirSj = list()
	sirS = list()
	sirI = list()
	sirR = list()
	Igerado = list()
	Rgerado = list()
	# Ready CSV
	arq = open("casos_sj2.csv")
	sirSjCsv = csv.DictReader(arq,fieldnames = ["S","R","I"])
	g = nx.erdos_renyi_graph(n, 0.001)
	i=0
	for row in sirSjCsv:
		sirSj.insert(i, { "S": int(row['S']), "I": int(row['I']), "R" : int(row['R'])})
		sirS.append(int(row['S']))
		sirI.append(int(row['I']))
		sirR.append(int(row['R']))
		i+=1

	# Model selection
	SIRModel = ep.SIRModel(g)
	# Model Configuration
	cfg = mc.Configuration()
	cfg.add_model_parameter('beta', 0.3)
	cfg.add_model_parameter('gamma',0.05)
	cfg.add_model_parameter("fraction_infected", 0.008)
	SIRModel.set_initial_status(cfg)

	SIRModel.reset()

	cfg.add_model_parameter('beta', x[0])
	cfg.add_model_parameter('gamma', x[1])
	cfg.add_model_parameter("fraction_infected", 0.08)
	SIRModel.set_initial_status(cfg)
	iterations = SIRModel.iteration_bunch(days)
	print(iterations)
	a = 0
	Igerado.clear()
	Rgerado.clear()
	matriz_gerada = np.zeros((days,3), dtype = np.int)

	for v in iterations:
		matriz_gerada[a][0] = v['node_count'][0]
		matriz_gerada[a][1] = v['node_count'][1]
		matriz_gerada[a][2] = v['node_count'][2]
		Igerado.append(v['node_count'][1])
		Rgerado.append(v['node_count'][2])
		a = a + 1
	print(iterations)
	mseI = mean_squared_error(sirI, Igerado)
	mseR = mean_squared_error(sirR, Rgerado)
	rmseI = math.sqrt(mseI)
	rmseR = math.sqrt(mseR)
	f = (rmseI + rmseR)/2
	return f