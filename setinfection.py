import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import csv
from pyeasyga.pyeasyga import GeneticAlgorithm
from numpy import linalg as LA

days =1
n = 10
beta = 0.3
gamma = 0.3
g = nx.erdos_renyi_graph(n, 0.5)
a = int()

# Model selection
model = ep.SIRModel(g)
# Model Configuration
cfg = mc.Configuration()
cfg.add_model_parameter('beta', beta)
cfg.add_model_parameter('gamma', gamma)
cfg.add_model_parameter("fraction_infected", 0.3)
model.set_initial_status(cfg)



# Simulation execution
iterations = model.iteration_bunch(days)

print(iterations)
matriz_gerada = list()
i=0

while(i < len(iterations)):
    matriz_gerada.insert(i, iterations[i]['node_count'])
    i += 1

# Ready CSV
arq = open("casos_sj.csv")
sirSjCsv = csv.DictReader(arq,fieldnames = ["R","I"])
sirSj = list()
i = 0

for row in sirSjCsv:
    sirSj.insert(i, { "I": row['I'], "R" : row['R']})
    i+=1

print(sirSj)
data = sirSj



ga = GeneticAlgorithm(data)


def fitness (individual, data):
    fitness = 0
    global a
    a= a+1

    print(data['R'])
    print(individual)
    print(LA.norm(individual[0:a]))
    print(a)
    return fitness

ga.fitness_function = fitness
ga.run()
#print(ga.best_individual())
    
