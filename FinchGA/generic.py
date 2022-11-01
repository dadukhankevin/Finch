import numpy as np
from Finch.FinchGA.EvolveRates import *


def to_genes(data):
    return np.array([Gene(i) for i in data])


class Fittness:
    def __init__(self, funcs, thresh=[]):
        self.funcs = iter(funcs)
        self.thresholds = iter(thresh)
        self.funct = next(self.funcs)
        self.thresh = next(self.thresholds)

    def func(self, individual):
        points, individual = self.funct(individual)
        if points >= self.thresh:
            try:
                self.funct = next(self.funcs)
                self.thresh = next(self.thresholds)
            except StopIteration:
                pass
        return points, individual


class Equation:
    def __init__(self, equation, *args):
        self.equation = equation


class GenePool:
    def __init__(self, data):
        self.data = data
        self.genes = to_genes(data)
        self.weights = np.asarray([gene.weight for gene in self.genes])

    def gen_data(self, length):
        data = np.random.choice(self.genes, length, p=self.weights / self.weights.sum())
        return data

    def update(self):
        self.weights = np.asarray([gene.weight for gene in self.genes])


class Chromosome:
    def __init__(self, genes):
        self.genes = genes
        self.data = np.asarray([gene.gene for gene in genes])


class Gene:
    def __init__(self, gene, weight=1):
        self.gene = gene  # the actual value: int str tuple list object function...
        self.weight = weight

    def penalize(self, percent=.1):
        # decreases likelihood of this gene being chosen
        if not callable(percent):
            percent = Rates(percent, 0).constant  # will always return percent
        self.weight = self.weight * (1 - percent())  # modify weight

    def reward(self, percent):
        # increases likelihood of this gene being chosen
        if not callable(percent):
            percent = Rates(percent, 0).constant  # will always return percent
        self.weight = self.weight * (1 + percent())  # modify weight


# returns numpy array of genes
class Fuzzy:
    def __init__(self, w):
        pass


class Individual:
    def __init__(self, data, fitness_func, fitness=0):
        self.age = 0
        self.genes = Chromosome(data)
        self.fitness = fitness
        self.fitness_func = fitness_func

    def fit(self, factor=1):
        """Factor: values closer to 0 favor the earlier fitness while values closer to 1 favors the new fitness """
        self.fitness = ((1-factor)*self.fitness_func) + (factor*self.fitness_func(self.genes.data))
    def set_genes(self, data):
        self.genes = Chromosome(data)




pool = GenePool([1,2,3,4,5])
pool.genes[1].reward(.1)
pool.update()
print(Chromosome(pool.gen_data(20)).data)