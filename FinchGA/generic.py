import copy
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
    def __init__(self, vars, equation, desired):  # TODO: make this class
        """
        :param vars: A list containing the variable names: ["x", "y"] must correspond to the index of the elements in an Individual
        :param args: The equation string to be evaluated. "x*y" or "import math\n math.pow(x, y)".
        """
        self.vars = vars
        self.equation = equation
        self.desired = desired
    def evaluate(self, individual):
        local_string = self.equation
        local_iter = iter(self.vars)
        for i in individual:
            local_string = local_string.replace(next(local_iter), str(i))
        try:
            return eval(local_string)
        except ZeroDivisionError and OverflowError:
            return self.desired*-1

class Gene:
    def __init__(self, gene, weight=1):
        """
        :param gene: The raw value of the gene
        :param weight: How much this gene appears in the best individuals
        """
        self.gene = gene  # the actual value: int str tuple list object function...
        self.weight = weight


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
        :param ar: The raw data to turn into Individual of List Gene
        :param fitness_func: The fitness function
        :param fitness: The default fitness
        :param calculate_on_start: Should the fitness be calculated on start? If so, fitness=0 is ignored
        """
        self.age = 0  # age of the Individual in epochs?
        # format data into a list of Genes
        self.fitness = fitness
        if mutation_function == None:
            self.mfunction = self.default_mutate
        self.genes = ar
        self.fitness_func = fitness_func

    def default_mutate(self, pool, percent):
        # gene = np.random.choice(self.genes)
        # gene = self.genes
        # input(self.genes)
        these_ones = np.random.choice(len(self.genes), size = int(len(self.genes) * int(percent())))
        if pool.replacement == True:
            for i in these_ones:
                self.genes[i] = pool.rand()
        else:

            for i in range(len(self.genes)):
                # input(gene.gene)

                if r.randint(0, 100) < percent():
                    newgene = pool.rand(index=i - 1)  # Use this pool if the pool is actually a typed gene pool
                    if np.all(self.genes != newgene) or pool.replacement: # Keeps only unique genes
                        self.genes[i - 1] = newgene # TODO: replace this whole thing with np.unique() which is faster

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
                    factor * self.fitness_func(self.genes))  # to prevent division by zero
        return self.fitness

    def get_genes(self): # returns all genes
        return self.genes

    def set_genes(self, data):
        """:param data: new data"""
        self.genes = data # sets the genes

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
        self.individuals = np.append(lst,self.individuals)

# pool = GenePool([1, 2, 3, 4, 5])

