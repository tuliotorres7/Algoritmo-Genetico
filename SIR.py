import csv
import math
import matplotlib.pyplot as plt
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import networkx as nx
import numpy as np
import random
from sklearn.metrics import mean_squared_error
from statistics import mean


class Individuo:
    def __init__(self, x, nBits, dimensao):
        self.x = []
        self.x = x
        self.nBits = nBits
        self.ndim = [dimensao]


def cria_populacao_inicial(nPop, populacao, nBits, dimensao):
    for i in range(nPop):
        ind = Individuo(0, nBits, dimensao)
        ind.x = []
        for k in range(nBits * dimensao):
            ind.x.append(random.randint(0, 1))
        populacao.append(ind)
    return populacao


def avalia_fitness(minNumA, maxNumA, minNumB, maxNumB,  nBits, populacao, grafo, days):
    vetorParametros = []
    fit = []
    aux = []
    for ind in populacao:
        a = "".join(map(str, ind.x[0:nBits]))
        dec = int(a, 2)
        aux.append(minNumA + (((maxNumA - minNumA) / ((2.0 ** nBits) - 1)) * dec))
        a = "".join(map(str, ind.x[nBits:nBits * 2]))
        dec = int(a, 2)
        aux.append(minNumB + (((maxNumB - minNumB) / ((2.0 ** nBits) - 1)) * dec))
        vetorParametros.append(aux)
        aux = []

    for x in vetorParametros:
        fit.append(func_obj(x, grafo, days))
    return fit


def torneio(nPop, fit, populacao):
    vpais = []
    pv = 0.9
    i = 0
    while i < nPop:
        p1 = random.randint(0, nPop - 1)
        p2 = random.randint(0, nPop - 1)
        while p1 == p2:
            p2 = random.randint(0, nPop - 1)
        r = random.randint(0, 1)
        if fit[p2] > fit[p1]:
            vencedor = p1
            if r > pv:
                vencedor = p2
        else:
            vencedor = p2
        if r > pv:
            vencedor = p1
        vpais.append(populacao[vencedor])
        i = i + 1
    return vpais


def cruzamento(pais, populacao, taxaCuzamento, dimensao, nBits):
    i = 0
    novaPopulacao = []
    while i < len(populacao):
        rand = random.random()
        corte = random.randint(1, nBits * dimensao - 1)
        if rand < taxaCuzamento:
            filho1 = pais[i].x[0:corte] + pais[i + 1].x[corte:nBits * 2]
            filho2 = pais[i + 1].x[0:corte] + pais[i].x[corte:nBits * 2]
            novaPopulacao.append(Individuo(filho1, nBits, dimensao))
            novaPopulacao.append(Individuo(filho2, nBits, dimensao))
            i = i + 2
        else:
            novaPopulacao.append(populacao[i])
            i = i + 1
    return novaPopulacao


def mutacao(populacao, taxaMutacao, dimensao, nBits):
    for ind in populacao:
        for j in range(0, nBits * dimensao - 1):
            rand = random.random()
            if rand < taxaMutacao:
                if ind.x[j] == 1:
                    ind.x[j] = 0
                else:
                    ind.x[j] = 1
    return populacao


def elitismo(populacaoMutada, populacao, vetFitness):
    novaPopulacao = populacaoMutada
    # o Melhor fitnes que permanece, nao os melhores
    novaPopulacao[10] = populacao[melhor_fitness(vetFitness)]
    return novaPopulacao


def melhor_fitness(vFitness):
    menor = 99999
    iMenor = 0
    for i in range(len(vFitness)):
        if vFitness[i] < menor:
            menor = vFitness[i]
            iMenor = i
    return iMenor


def func_obj(x, grafo, days):
    sirSj = list()
    sirS = list()
    sirI = list()
    sirR = list()
    Igerado = list()
    Rgerado = list()
    # Ready CSV
    arq = open("casos_sj3.csv")
    sirSjCsv = csv.DictReader(arq, fieldnames=["S", "I", "R"])
    i = 0
    for row in sirSjCsv:
        sirSj.insert(i, {"S": int(row['S']), "I": int(row['I']), "R": int(row['R'])})
        sirS.append(int(row['S']))
        sirI.append(int(row['I']))
        sirR.append(int(row['R']))
        i += 1

    # Model selection
    SIRModel = ep.SIRModel(grafo)
    # Model Configuration
    cfg = mc.Configuration()
    cfg.add_model_parameter('beta', 0.3)
    cfg.add_model_parameter('gamma', 0.05)
    cfg.add_model_parameter("fraction_infected", 0.0006)
    SIRModel.set_initial_status(cfg)

    SIRModel.reset()

    cfg.add_model_parameter('beta', x[0])
    cfg.add_model_parameter('gamma', x[1])
    cfg.add_model_parameter("fraction_infected", 0.0006)
    SIRModel.set_initial_status(cfg)
    iterations = SIRModel.iteration_bunch(days)
    # print(iterations)
    a = 0
    Igerado.clear()
    Rgerado.clear()
    matriz_gerada = np.zeros((days, 3), dtype=int)

    for v in iterations:
        matriz_gerada[a][0] = v['node_count'][0]
        matriz_gerada[a][1] = v['node_count'][1]
        matriz_gerada[a][2] = v['node_count'][2]
        Igerado.append(v['node_count'][1])
        Rgerado.append(v['node_count'][2])
        a = a + 1
    # print(iterations)

    mseI = mean_squared_error(sirI, Igerado)
    mseR = mean_squared_error(sirR, Rgerado)

    np_mseI = np.square(np.subtract(sirI, Igerado)).mean()
    np_mseR = np.square(np.subtract(sirR, Rgerado)).mean()

    print(f'sirR : {sirR}')
    print(f'Rgerado: {Rgerado}')

    print(f'mseI : {mseI}')
    print(f'mseR : {mseR}')

    print(f'mseI do NP : {np_mseI}')
    print(f'mseR do NP: {np_mseR}')

    rmseI = math.sqrt(mseI)
    rmseR = math.sqrt(mseR)
    f = (rmseI + rmseR) / 2
    return f

def main():
    days = 30
    n = 50000
    nPop = 50000
    nGer = 10
    taxaCruza = 1
    taxaMuta = 0.1
    nBits = 6
    minNumA = 0
    maxNumA = 0.3
    minNumB = 0
    maxNumB = 0.3
    dimensao = 2
    populacao = []
    # pais = []
    # nElitismo = []
    vetMin = []
    vetMax = []
    # vetIndexMin = []
    grafo = nx.erdos_renyi_graph(n, 0.001)
    populacao = cria_populacao_inicial(nPop, populacao, nBits, dimensao)
    g = 0
    while g in range(nGer):
        fitness = avalia_fitness(minNumA, maxNumA,minNumB, maxNumB, nBits, populacao, grafo, days)
        pais = torneio(nPop, fitness, populacao)
        populacaoCruzada = cruzamento(pais, populacao, taxaCruza, dimensao, nBits)
        populacaoMutada = mutacao(populacaoCruzada, taxaMuta, dimensao, nBits)
        populacaoFinal = elitismo(populacaoMutada, populacao, fitness)
        populacao = populacaoFinal
        vetMin.append(min(fitness))
        vetMax.append(max(fitness))
        # vetIndexMin.append(index(min(fitness)))
    aux = []
    for x in range(1, nGer + 1):
        aux.append(x)
    plt.plot(vetMin)
    plt.plot(vetMax)
    res = [mean(values) for values in zip(vetMin, vetMax)]
    plt.plot(res)
    plt.title("Max, Min , Média por Geração")
    plt.show()
    # print(populacao)

if __name__ == '__main__':
    main()