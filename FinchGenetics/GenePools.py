import random
import math
import numpy as np
from Finch.FinchGenetics.Genetics import Individual
from Finch.FinchGenetics.Rates import make_callable
def mul_tup(tup):
    base = 1
    for i in tup:
        base = base*i
    return base
class Pool:
    def __init__(self):
        pass

    def gen_data(self, data, amount):
        pass


class GenePool(Pool):
    def __init__(self, data, fitness_func, mx=1, mn=0, replacement=True, max_fitness=1, shape=None, treat_sublists_as_genes=False):
        """
        :param data: The "vocabulary" to make into genes
        :param fitness_func: The fitness function
        :param mx: The minimum weight
        :param mn: The maximum weight
        """
        self.shape = shape
        super().__init__()
        self.fitnes_func = fitness_func
        self.raw = np.array(data)
        self.weights = np.ones(len(self.raw))
        self.mx = make_callable(mx)
        self.mn = make_callable(mn)
        self.replacement = replacement
        self.max_fitness = max_fitness
        self.directional_weights = 1
        self.treat_sublists_as_genes = treat_sublists_as_genes



    def gen_data(self, population, population_count):
        """
        :param data: The already existing data
        :param population: The wanted population_count
        :param length: The amount of chromosome within each individual.
        :return: New data with old data
        """
        while len(population) < population_count:
            p = np.nan_to_num(self.weights / self.weights.sum(), nan=0)
            if self.treat_sublists_as_genes:
                data = self.raw[np.random.choice(len(self.raw), p=p, replace=self.replacement, size=self.shape[0])]
            else:
                data = self.raw[
                    np.random.choice(len(self.raw), p=p, replace=self.replacement, size=mul_tup(self.shape))]
            ind = Individual(self, data, self.fitnes_func)  # Creates the new individual
            ind.fit(1)  # completely recalculates the fitness
            population = np.append([ind], population)
        return population
    def set_all_weights(self, value):
        for i in range(len(self.weights) - 1):
            self.weights[i] = value

    def update(self):
        """
        Updates model weights
        :return: None
        """

        self.weights = np.asarray([gene.weight for gene in self.raw])  # updates weights

        self.weights = np.nan_to_num(self.weights, nan=0)  # get rid of nan values

    def rand(self, index=None):
        """
        Generates a new random gene
        :return: New Gene
        """
        r = random.choices(self.raw, weights=self.weights / sum(self.weights), k=1)[0]
        return r


    def rand_many(self, amount):

        r = np.random.choice(len(self.raw), p=self.weights / sum(self.weights), size=amount, replace=self.replacement)
        r = self.raw[r]
        return r


class TypedGenePool(Pool):
    def __init__(self, pools=[]):
        """
        :param pools: The pools to be included in the mega gene pool
        """
        super().__init__()
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
            gen.append(ind)


class FloatPool(Pool):
    def __init__(self, min, max, fitfunc, shape,treat_sublists_as_genes=False,initialization="midpoint"):
        super().__init__()
        self.min = min
        self.max = max
        self.shape = shape
        self.fitness_func = fitfunc
        self.replacement = True
        self.weights = [1]
        self.directional_weights = 1
        self.initialization = initialization
        self.midpoint = (min + max) / 2
        self.first = True
        self.treat_sublists_as_genes = treat_sublists_as_genes

    def rand(self, index):
        """
        :param index: The pool to edit
        :return: Random token from that pool
        """
        return np.random.uniform(self.min, self.max, 1)[0]

    def gen_data(self, population, population_count):
        """
        :param data: The already existing data
        :param population_count: The wanted population
        :param length: The amount of chromosome within each individual.
        :return: New data with old data
        """
        data = np.zeros(mul_tup(self.shape))
        if self.initialization == "midpoint":
            data.fill(self.midpoint)
        while len(population) < population_count:
            ind = Individual(self,
                             ar=data,
                             fitness_func=self.fitness_func)  # Creates the new individual
            ind.fit(1)  # completely recalculates the fitness
            population = np.append(ind, population)
        return population
    def get_one(self, shape):
        return self.gen_data([], 1, shape)
    def get_weight(self, raw):
        return 1  # float pools don't have weights yet

    def rand_many(self, amount):
        r = np.random.uniform(self.min, self.max, amount)
        return r



