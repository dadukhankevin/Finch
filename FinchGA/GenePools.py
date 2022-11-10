import random
import math
import numpy as np
from Finch.FinchGA.generic import Gene, Individual
import Finch.FinchGA.EvolveRates as er

class GenePool:
    def __init__(self, data, fitness_func, mx=1, mn=0, replacement=True):
        """
        :param data: The "vocabulary" to make into genes
        :param fitness_func: The fitness function
        :param mx: The minimum weight
        :param mn: The maximum weight
        """
        self.fitnes_func = fitness_func
        self.raw = np.array(data)
        self.genes = to_genes(data)  # converts data to chromosome
        self.weights = np.asarray([gene.weight for gene in self.genes])
        self.mx = er.make_constant_rate(mx)
        self.mn = er.make_constant_rate(mn)
        self.replacement = replacement

    def to_gene(self, i):
        """
        :param i: Raw data
        :return: Object of Gene class
        """
        return self.date[np.where(self.genes.gene == i)][0]


    def gen_data(self, gen, population, length):
        """
        :param data: The already existing data
        :param population: The wanted population
        :param length: The amount of chromosome within each individual.
        :return: New data with old data
        """
        while len(gen.individuals) < population:
            ind = Individual(self,
                             ar=self.raw[np.random.choice(len(self.raw), p=self.weights / self.weights.sum(), replace=self.replacement, size=length)],
                             fitness_func=self.fitnes_func) #Creates the new individual
            ind.fit(1) # completely recalculates the fitness
            gen.add(ind)
    def set_all_weights(self, value):
        for i in range(len(self.weights)-1):
            self.weights[i] = value
    def update(self):
        """
        Updates model weights
        :return: None
        """

        self.weights = np.asarray([gene.weight for gene in self.genes])  # updates weights

        self.weights = np.nan_to_num(self.weights,nan = 0) # get rid of nan values

    def rand(self, index=None):
        """
        Generates a new random gene
        :return: New Gene
        """
        r = random.choices(self.raw, weights=self.weights/sum(self.weights), k=1)[0]
        return r
    def get_weight(self, raw):
        return self.genes[np.where(self.raw == raw)[0]]

class TypedGenePool:
    def __init__(self, pools=[]):
        """
        :param pools: The pools to be included in the mega gene pool
        """
        self.pools = pools
        self.replacement = None
    def rand(self, index):
        """
        :param index: The pool to edit
        :return: Random token from that pool
        """
        return self.pools[index].rand()

    def update(self):
        """
        :return: Updates the weights in all the gene pools
        """
        for i in self.pools:
            i.update()

    def gen_data(self, gen, population, length):
        """
        :param gen:
        :param population:
        :param length: has to equal the amount of GenePools
        :return:
        """
        assert length == len(self.pools), "The length specified must match the number of gene pools!"
        for i in range(population):
            addable = []
            for i in self.pools:

                addable = np.append(addable, i.rand(self.pools.index(i)))
            ind = Individual(self,
                             ar=addable,
                             fitness_func=i.fitnes_func)  # TODO: verify this logic is best
            gen.add(ind)




def to_genes(data):
    """
    :param data: Raw vocabulary
    :return:
    """
    return np.array([Gene(i) for i in data])
