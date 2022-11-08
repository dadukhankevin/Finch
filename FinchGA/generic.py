import math
import random

import numpy
import numpy as np
from Finch.FinchGA.EvolveRates import *
import random as r


class Fittness:
    def __init__(self, funcs, thresh=[]):
        """
        :param funcs: The fitness functions
        :param thresh: The threshold of fitness before you want to move on to the next function
        """
        self.funcs = iter(funcs)
        self.thresholds = iter(thresh)
        self.funct = next(self.funcs)
        self.thresh = next(self.thresholds)

    def func(self, individual):
        points, individual = self.funct(individual)
        if points >= self.thresh:  # gets then next function
            try:
                self.funct = next(self.funcs)
                self.thresh = next(self.thresholds)
            except StopIteration:
                pass
        return points, individual


class Equation:
    def __init__(self, equation, *args):  # TODO: make this class
        """
        :param equation:
        :param args:
        """
        self.equation = equation


class Chromosome:
    def __init__(self, genes):
        """
        This is currently not useful in any way but it may eventually be utilized.
        :param genes: the genes in the chromosome
        """
        self.genes = genes

    def get_raw(self):  # gets the raw value of each gene
        return numpy.array([i.gene for i in self.genes])

    def set_raw(self, data):  # converts raw values to genes
        self.genes = [Gene(i) for i in data]


class Gene:
    def __init__(self, gene, weight=1):
        """
        :param gene: The raw value of the gene
        :param weight: How much this gene appears in the best individuals
        """
        self.gene = gene  # the actual value: int str tuple list object function...
        self.weight = weight


# returns numpy array of chromosome
class Fuzzy:
    def __init__(self, w):
        """
        TODO: Make it so that individuals that are similar are assigned the same fitness. Helps with optimization
        :param w:
        """
        pass


class Individual:

    def __init__(self, pool, ar, fitness_func, fitness=0, mutation_function=None):
        """
        :param pool: the gene pool we want to use
        :param ar: The raw data to turn into Individual of Chromosome of List Gene
        :param fitness_func: The fitness function
        :param fitness: The default fitness
        :param calculate_on_start: Should the fitness be calculated on start? If so, fitness=0 is ignored
        """
        self.age = 0  # age of the Individual in epochs?
        if type(ar) == list or type(ar) == numpy.array:
            ar = numpy.asarray([pool.to_gene(i) for i in ar])  # format data into a list of Genes
        self.fitness = fitness
        if mutation_function == None:
            self.mfunction = self.default_mutate
        self.chromosome = Chromosome(ar)  # just a list of chromosome really, for now
        self.fitness_func = fitness_func

    def default_mutate(self, pool, percent):
        # gene = np.random.choice(self.chromosome.genes)
        # gene = self.chromosome.genes
        # input(self.chromosome.genes)
        for i in range(len(self.chromosome.genes)):
            # input(gene.gene)
            if r.randint(0, 100) < percent():
                newgene = pool.rand(index=i - 1)  # Use this pool if the pool is actually a typed gene pool
                if np.all(self.chromosome.get_raw() != newgene.gene) or pool.replacement: # Keeps only unique genes
                    self.chromosome.genes[i - 1] = newgene # TODO: replace this whole thing with np.unique() which is faster

    def mutate(self, pool, select, percent):
        """
        :param pool: The gene pool
        :param select: The Ods of selecting this individual
        :param percent: The percent of genes within this individual to mutate (if selected)
        :return:
        """
        if r.randint(0, 100) < select():
            self.mfunction(pool, percent) #calls the mutate function
            self.fit(1) # calls the fitness function with 100% bias to its new fitness

    def fit(self, factor=1):
        """
        :param factor: values closer to 0 favor the earlier fitness while values closer to 1 favors the new fitness
        """
        self.fitness = ((1 - factor) * self.fitness) + (
                    factor * self.fitness_func(self.chromosome.get_raw()))  # to prevent division by zero
        return self.fitness

    def get_genes(self): # returns all genes
        return np.array([i.gene for i in self.chromosome.genes])

    def set_genes(self, data):
        """:param data: new data"""
        self.chromosome = Chromosome(data) # sets the chromosome genes

    @staticmethod
    def sorting_key(individual):
        """For use with sorted(x, key=Individual.sorting_key) to sort by fitness. However, numpy should be faster"""
        return individual.fitness, individual


class Generation:
    def __init__(self, individuals, fitnessfunc=None):
        """
        :param individuals: List of individuals
        """

        self.individuals = individuals

    def sort(self):  # TODO: find bett

        # er way to do this
        sort = np.argsort([i.fitness for i in self.individuals])
        self.individuals = self.individuals[sort]
        return self.individuals

    def fit_all(self, factor):
        for i in self.individuals:
            i.fit(factor=factor)

    def add(self, lst):
        self.individuals = np.append(self.individuals, lst)

# pool = GenePool([1, 2, 3, 4, 5])
# pool.chromosome[1].reward(.1)
# pool.update()
# print(Chromosome(pool.gen_data(20)).data)
