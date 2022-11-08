# A Keras style GA genetic algorithm library
from Finch.FinchGA.GenePools import GenePool
from Finch.FinchGA import Layers
import matplotlib.pyplot as plt
from Finch.FinchGA.Environments import *
import numpy as np


def fit(backpack):
    weight = 0
    value = 0
    for item in backpack:
        weight += float(item[2])  # Adds to the weight of the backpack
        value += float(item[1])  # Adds to the overall value of the backpack
    if weight > 15:  # If weight is above 15 fitness is 0
        return 0
    else:
        return value  # otherwise return the sum of the importance of each item in teh backpack

# In the format [name, weight, value] all of these have little bearing on reality.
backpack = np.array(
    [["apple", 1, .1], ["phone", 8, 6], ["lighter", 1, .1], ["Book", .1, 2], ["compass", 2, .4], ["flashlight", 1, 6],
     ["water", 5, 9], ["passport", 7, .5]])

pool = GenePool(backpack, fit, replacement=False)  # TO avoid duplicates "replacement" must be false

env = SequentialEnvironment(layers=[
    Layers.GenerateData(pool, population=10, array_length=2, delay=0), # Generates data
    Layers.SortFitness(), # Sorts individuals by fitness
    Layers.NarrowGRN(pool, delay=1, method="best", amount=1, reward=.05, penalty=.05, mn=.1, mx=100, every=1), # Calculates new weights
    Layers.UpdateWeights(pool, every=1, end=200), # Updates likelihood of specific
    Layers.Parents(pool, gene_size=1, family_size=1, delay=0, every=4, method="best"), #Parents random individuals together
    Layers.Mutate(pool, delay=0, select_percent=100, likelihood=40), #mutates 40% ish of all of the individuals
    Layers.SortFitness(), # Sorts individuals by fitness
    Layers.KeepLength(100), # Keeps the population under 100

])

env.compile(epochs=40, fitness=fit, every=1, stop_threshold=20) #stop when value > 18
_, hist = env.simulate_env()
print(pool.weights) # relative weights of each gene
plt.plot(hist)
plt.show() # Graph our progress