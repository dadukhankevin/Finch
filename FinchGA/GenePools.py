import random
import math
import numpy as np
from generic import Gene, Individual


class GenePool:
    def __init__(self, data, fitness_func, mx=math.inf, mn=-math.inf):
        """
        :param data: The "vocabulary" to make into genes
        :param fitness_func: The fitness function
        :param mx: The minimum weight
        :param mn: The maximum weight
        """
        self.data = data
        self.fitnes_func = fitness_func
        self.genes = to_genes(data)  # converts data to chromosome
        self.weights = np.asarray([gene.weight for gene in self.genes])  # calculates initial weights
        self.mx = mx
        self.mn = mn

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
                             ar=np.random.choice(self.genes, length, p=self.weights / self.weights.sum(), replace=True),
                             fitness_func=self.fitnes_func)  # TODO: verify this logic is best
            gen.add(ind)

    def update(self):
        """
        Updates model weights
        :return: None
        """
        self.weights = np.asarray([min(max(gene.weight, self.mn), self.mx) for gene in self.genes])  # updates weights

    def rand(self):
        """
        Generates a new random gene
        :return: New Gene
        """
        r = random.choice(np.random.choice(self.genes, 1, p=self.weights / self.weights.sum()))
        return r


def to_genes(data):
    return np.array([Gene(i) for i in data])
