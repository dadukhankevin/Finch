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
pool = GenePools.GenePool(list("qwertyuiopasdfghjklzxcvbnm  "), fitness, mx=1, mn=.001, max_fitness=10)
env = Environments.SequentialEnvironment(layers=[
    l.GenerateData(pool, population=20, array_length=10, end=2),
    l.SortFitness(),
    #l.Mutate(pool, select_percent=20, likelihood=40),
    l.FastMutateTop(pool, amount=1, individual_mutation_amount=5),
    l.NarrowGRN(pool, delay=0, method="outer", amount=1, reward=.0001, penalty=.001, every=1),
    l.UpdateWeights(pool, every=1, end = 20),
    l.SortFitness(),
    #l.RemoveDuplicatesFromTop(amount=2),
    l.OverPoweredMutation(pool=pool, iterations=20, index=-1, fitness_function=fitness),
    l.Duplicate(3),
    l.Parents(pool, gene_size=3, family_size=2, percent=50, every=1, method="best", amount=4), # parents the best ones
    l.KeepLength(10), #keeps a low population
])
env.compile(epochs=1000,every=500, fitness=fitness, stop_threshold=50)
hist, data = env.simulate_env()
info = env.display_history()
print("best percent: "+str(env.best))
print("best individual: "+str(env.best_ind.genes))
print(info)
print(pool.weights)
env.plot()
