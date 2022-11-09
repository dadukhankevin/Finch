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
        points += abs(TextBlob(word).polarity)*len(word)
    return points
pool = GenePools.GenePool(list("qwertyuiopasdfghjklzxcvbnm     "), fitness, mx=100, mn=0)
env = Environments.SequentialEnvironment(layers=[
    l.GenerateData(pool, population=200, array_length=20, end=2),
    l.SortFitness(),
    l.Mutate(pool, select_percent=20, likelihood=40),
    l.NarrowGRN(pool, delay=0, method="outer", amount=1, reward=.7, penalty=.99, mn=.1, mx=10, every=1, end = 20),
    l.UpdateWeights(pool, every=1, end = 20),
    l.SortFitness(),
    l.RemoveDuplicatesFromTop(amount=2),
    l.Parents(pool, gene_size=5, family_size=2, percent=50, every=1, method="best", amount=20), # parents the best ones
    l.KeepLength(100), #keeps a low population
])
env.compile(epochs=160,every=10, fitness=fitness, stop_threshold=50)
hist, data = env.simulate_env()
info = env.display_history()
print("best percent: "+str(env.best))
print("best individual: "+str(env.best_ind.chromosome.get_raw()))
print(info)
print(pool.weights)
env.plot()
