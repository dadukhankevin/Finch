"""
This example generates real words because the fitness function incentives text that has higher positive
or negative sentiment at all (and so is made up of real words)
"""
import string

from Finch.FinchGA import generic
from textblob import TextBlob
from Finch.FinchGA import Environments, GenePools
from Finch.FinchGA import Layers as l
from Finch.FinchGA import EvolveRates as r

def fitness(sentence):
    real = "".join(sentence).split(" ")
    points = 0
    for word in real:
        points += abs(TextBlob(word).polarity)
    return points
pool = GenePools.GenePool(list("qwertyuiopasdfghjklzxcvbnm     "), fitness, mx=100, mn=0.001)
r.graph(r.Rates(40,.00001).slow, 200*400)
env = Environments.SequentialEnvironment(layers=[
    l.GenerateData(pool, population=50, array_length=20),
    l.SortFitness(),
    l.Mutate(pool, select_percent=40, likelihood=r.Rates(40,.00001).slow),
    l.NarrowGRN(pool, delay=20, method="outer", amount=2, reward=.3, penalty=.99, mn=.1, mx=40, every=1),
    l.UpdateWeights(pool),
    l.SortFitness(),
    l.Parents(pool, gene_size=4, family_size=10, percent=100, every=4, method="best", amount=4), # parents the best ones
    l.KeepLength(100), #keeps a low population
])
env.compile(epochs=800,every=10, fitness=fitness, stop_threshold=2)
hist, data = env.simulate_env()
info = env.display_history()
print("best percent: "+str(env.best))
print("best individual: "+str(env.best_ind.chromosome.get_raw()))
print(info)
print(pool.weights)
env.plot()
