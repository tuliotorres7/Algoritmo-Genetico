import random
import math
import numpy as np
from funcao_AG import func_obj
import matplotlib.pyplot as plt
from statistics import mean
#from matplotlib import pyplot as plt

nPop = 30
nGer = 30
taxaCruza = 1
taxaMuta = 0.1
nBits = 20
maxNum = 1
minNum = -1 
dimensao = 2
populacao = []
pais=[]
nElitismo = []
vetMin = []
vetMax = []
vetIndexMin = []

class individuo:
    def __init__(self,x,nBits,dimensao):
        self.x = []
        self.x = x
        self.nBits = nBits 
        self.ndim = [dimensao]


def criaPopulacaoInicial(nPop,populacao,nBits):
    for i in range(nPop):
        ind = individuo(0,nBits,2)
        ind.x = []
        for k in range(nBits*2):
            ind.x.append(random.randint(0,1))    
        populacao.append(ind)
    #populacao[len(populacao)-1].x = [1,1,1,0,0,0,1,1,1,0,0,0]
    return populacao

def avaliaFitness(minNum,maxNum,nBits,populacao):
    vetorParametros = []
    fit = []
    aux = []
    a = str()
    for ind in populacao:
        #a = str(ind.x[0:nBits]).strip('[]')
        #print(a)
        a= "".join(map(str,ind.x[0:nBits]))
        dec = int(a,2)
        aux.append((minNum)+(((maxNum - minNum)/((2.0**nBits)-1))*dec))
        a=  "".join(map(str,ind.x[nBits:nBits*2]))
        dec = int(a,2)
        aux.append((minNum)+(((maxNum - minNum)/((2.0**nBits)-1))*dec))
        vetorParametros.append(aux)
        aux = []
    for x in vetorParametros:
        fit.append(func_obj(x))
    return fit

def torneio(nPop,fit,populacao):
    vpais= []
    pv = 0.9
    i = 0
    vencedor = 0
    while( i < nPop ):
        p1 = random.randint(0,nPop-1)
        p2 = random.randint(0,nPop-1)
        while(p1==p2):
            p2 = random.randint(0,nPop-1)
        r = random.randint(0,1)
        if(fit[p2]>fit[p1]):
            vencedor = p1
            if(r>pv):
                vencedor = p2
        else:
            vencedor = p2
        if(r>pv):
            vencedor = p1
        vpais.append(populacao[vencedor])
        i=i + 1
    return vpais
def cruzamento(pais,populacao,taxaCuzamento,dimensao,nBits):
    i = 0
    novaPopulacao = []
    while(i<len(populacao)):
        rand = random.random()
        corte = random.randint(1,nBits*dimensao-1)
        if rand < taxaCuzamento:
            filho1 = pais[i].x[0:corte] + pais[i+1].x[corte:nBits*2]
            filho2 = pais[i+1].x[0:corte] + pais[i].x[corte:nBits*2]
            novaPopulacao.append(individuo(filho1,nBits,dimensao))
            novaPopulacao.append(individuo(filho2,nBits,dimensao))
            i= i + 2
        else:
            novaPopulacao.append(populacao[i])
            i=i+1
    return novaPopulacao

def mutacao(populacao,taxaMutacao,dimensao,nBits):
    for ind in populacao:
        for j in range(0,nBits*dimensao-1):
            rand = random.random()
            if rand < taxaMutacao:
                if ind.x[j] == 1:
                    ind.x[j] = 0
                else:
                    ind.x[j] = 1  
    return populacao

def elitismo(populacaoMutada,populacao,vetFitness):
    novaPopulacao = populacaoMutada
    #o Melhor fitnes que permanece, nao os melhores
    novaPopulacao[10] = populacao[melhorFitness(vetFitness)]
    return novaPopulacao


def melhorFitness(vFitness):
    menor = 99999
    iMenor = 0
    for i in range(len(vFitness)):
        if vFitness[i] < menor:
            menor = vFitness[i]
            iMenor = i
    return iMenor

    

populacao = criaPopulacaoInicial(nPop,populacao,nBits)
g=0
while(g<nGer):
    fitness = avaliaFitness(minNum,maxNum,nBits,populacao)
    pais = torneio(nPop,fitness,populacao)
    populacaoCruzada = cruzamento(pais,populacao,taxaCruza,dimensao,nBits)
    populacaoMutada = mutacao(populacaoCruzada,taxaMuta,dimensao,nBits)
    populacaoFinal = elitismo(populacaoMutada,populacao, fitness)
    populacao = populacaoFinal
    vetMin.append(min(fitness))
    vetMax.append(max(fitness))
    #vetIndexMin.append(index(min(fitness)))
    g=g+1
aux = []
for x in range(1,nGer+1):
    aux.append(x)
plt.plot(vetMin)
plt.plot(vetMax)
res = [mean(values) for values in zip(vetMin,vetMax)]
plt.plot(res)
plt.title("Max, Min , Média por Geração")
plt.show()
#print(populacao)
     
