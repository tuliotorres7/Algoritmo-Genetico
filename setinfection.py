import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import csv
import numpy as np
import math
from geneticalgorithm import geneticalgorithm as ga
from sklearn.metrics import mean_squared_error

days =1
n = 500
beta = 0.3
gamma = 0.3
g = nx.erdos_renyi_graph(n, 0.5)
a = int()

# Model selection
SIRModel = ep.SIRModel(g)
# Model Configuration
cfg = mc.Configuration()
cfg.add_model_parameter('beta', beta)
cfg.add_model_parameter('gamma', gamma)
cfg.add_model_parameter("fraction_infected", 0.3)
SIRModel.set_initial_status(cfg)



# Ready CSV
arq = open("casos_sj.csv")
sirSjCsv = csv.DictReader(arq,fieldnames = ["S","R","I"])

matriz_gerada = list()
sirSj = list()
sirS = list()
sirI = list()
sirR = list()
Igerado = list()
Rgerado = list()
i = 0

for row in sirSjCsv:
    sirSj.insert(i, { "S": int(row['S']), "I": int(row['I']), "R" : int(row['R'])})
    sirS.append(int(row['S']))
    sirI.append(int(row['I']))
    sirR.append(int(row['R']))
    i+=1

print(sirSj)
data = sirSj

print(data[0]['I'])

varbound=np.array([[0,1]]*2)

def fitness(x):
    global a
    a = a + 1
    cfg.add_model_parameter('beta', x[0])
    cfg.add_model_parameter('gamma', x[1])
    #SIRModel.set_initial_status(cfg)
    iterations = SIRModel.iteration_bunch(days)
    print(iterations)
    matriz_gerada.insert(a, iterations[0]['node_count'])
    Igerado.clear()
    Rgerado.clear()
    for row in matriz_gerada:
        Igerado.append(row[1])
        Rgerado.append(row[2])

    print(matriz_gerada)
    print(Igerado)
    print(iterations)
    #print(individual)

    mseI = mean_squared_error(sirI, Igerado)
    mseR = mean_squared_error(sirR, Rgerado)
    
    rmseI = math.sqrt(mseI)

    rmseR = math.sqrt(mseR)
    fitness = (rmseI + rmseR) / 2    
    return fitness


GaModel= ga(function= fitness,dimension=2,variable_type='real',variable_boundaries=varbound)

GaModel.run()
