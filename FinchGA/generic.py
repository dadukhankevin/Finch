import numpy
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
        """
        :param data: The raw data to be turned into GenePool of list * Gene
        """
        self.data = data
        self.genes = to_genes(data)  # converts data to genes
        self.weights = np.asarray([gene.weight for gene in self.genes])  # calculates initial weights

    def gen_data(self, data, population, length):
        """
        :param data: The already existing data
        :param population: The wanted population
        :param length: The amount of genes within each individual.
        :return: New data with old data
        """
        while data.count < population:  # TODO: verify this logic is best
            data = np.append(data, np.random.choice(self.genes, length, p=self.weights / self.weights.sum()))
        return data

    def update(self):
        self.weights = np.asarray([gene.weight for gene in self.genes])  # updates weights


class Chromosome:
    def __init__(self, genes):
        self.genes = genes


class Gene:
    def __init__(self, gene, weight=1):
        self.gene = gene  # the actual value: int str tuple list object function...
        self.weight = weight

    def penalize(self, percent=.1):
        """
        :param percent: Percent to decrease the weight
        :return: None
        """
        # decreases likelihood of this gene being chosen
        if not callable(percent):
            percent = Rates(percent, 0).constant  # will always return percent
        self.weight = self.weight * (1 - percent())  # modify weight

    def reward(self, percent):
        """
        :param percent: The percent to increase the weight
        :return: none
        """
        # increases likelihood of this gene being chosen
        if not callable(percent):
            percent = Rates(percent, 0).constant  # will always return percent
        self.weight = self.weight * (1 + percent())  # modify weight


# returns numpy array of genes
class Fuzzy:
    def __init__(self, w):
        pass


class Individual:
    def __init__(self, data, fitness_func, fitness=0, calculate_on_start=False):
        """
        :param data: The raw data to turn into Individual of Chromosome of List Gene
        :param fitness_func: The fitness function
        :param fitness: The default fitness
        :param calculate_on_start: Should the fitness be calculated on start? If so, fitness=0 is ignored
        """
        self.age = 0  # age of the Individual in epochs?
        if type(data) == list:
            data = numpy.asarray([Gene(i) for i in data])  # format data into a list of Genes
        self.fitness = fitness
        if calculate_on_start:
            self.fitness = fitness_func(data)
        self.genes = Chromosome(data)  # just a list of genes really, for now
        self.fitness_func = fitness_func

    def fit(self, factor=1):
        """
        :param factor: values closer to 0 favor the earlier fitness while values closer to 1 favors the new fitness
        """
        self.fitness = ((1 - factor) * self.fitness) + (factor * self.fitness_func(self.genes.data))
        return self.fitness

    def set_genes(self, data):
        """:param data: new data"""
        self.genes = Chromosome(data)

    @staticmethod
    def sorting_key(individual):
        """For use with sorted(x, key=Individual.sorting_key) to sort by fitness. However, numpy should be faster"""
        return individual.fitness, individual




class Generation:
    def __init__(self, individuals):
        """

        :param individuals: List of individuals
        """
        self.individuals = individuals

    def sort(self):  # TODO: find better way to do this
        sort = np.argsort([i.fitness for i in self.individuals])
        self.individuals = self.individuals[sort]
        return self.individuals


pool = GenePool([1, 2, 3, 4, 5])
pool.genes[1].reward(.1)
pool.update()
print(Chromosome(pool.gen_data(20)).data)
