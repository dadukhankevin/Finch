# A Keras style GA genetic algorithm library
from Finch.FinchGA.GenePools import GenePool
from Finch.FinchGA import Layers
from Finch.FinchGA.Environments import *
from Finch.FinchGA.FitnessFunctions import ValueWeightFunction
import numpy as np

fitness = ValueWeightFunction(maxweight=15) #max wieght of our backpack is 15 weight units. Feel free to make your own fitness function whenever.
# In the format [name, value, weight] all of these have little bearing on reality.
backpack = np.array(
    [["apple", .1, 1], ["phone", 6, 2], ["lighter", .5, .1], ["Book", 3, 33], ["compass", .5, .01], ["flashlight", 1, 4],
     ["water", 10, 6], ["passport", 7, .5], ["computer", 11, 15], ["clothes", 10, 2], ["glasses", 3, .1], ["covid", -100, 0], ["pillow", 1.4, 1]])

pool = GenePool(backpack, fitness.func, replacement=False)  # TO avoid duplicates "replacement" must be false
n = 0
def info(data=None):
    global n
    n += 1
    return data
env = SequentialEnvironment(layers=[
    Layers.GenerateData(pool, population=20, array_length=4, delay=0), # Generates 20 individuals and then at least 10
    Layers.SortFitness(), # Sorts individuals by fitness
    Layers.NarrowGRN(pool, delay=1, method="fast", amount=1, reward=.6, penalty=.99, every=1), # Calculates new weights
    Layers.UpdateWeights(pool, every=1, end=200), # Updates likelihood of specific
    Layers.Parents(pool, gene_size=1, family_size=4, delay=0, every=4, method="best", amount=4), #Parents random individuals together
    Layers.Mutate(pool, delay=0, select_percent=100, likelihood=20), #mutates 40% ish of all of the individuals
    Layers.SortFitness(), # Sorts individuals by fitness
    Layers.RemoveDuplicatesFromTop(amount=2),

    Layers.KeepLength(10), # Keeps the population under 10, allows the GenerateData layer to generate 10 new individuals
    Layers.Function(info)
])

env.compile(epochs=100, fitness=fitness.func, every=1, stop_threshold=33) #stop when value > 18
_, hist = env.simulate_env()
print(pool.weights) # relative weights of each gene
print("best: ", env.best_ind.genes)
plt.plot(hist)
plt.show() # Graph our progress