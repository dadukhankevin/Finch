import copy
import logging
import math

import numpy as np

from Finch.FinchGenetics.Genetics import Individual
from Finch.FinchGenetics.Layers import Layer
from Finch.FinchGenetics.Rates import make_callable, make_switcher
import random


class MutatePercent(Layer):
    def __init__(self, percent=.5, delay=0, end=math.inf, every=1, iterations=1, change_amount=1, fit_factor=1):
        self.fit_factor = fit_factor
        self.percent = make_callable(percent)
        self.change_amount = make_switcher(change_amount)
        super().__init__(delay=delay, end=end, every=every, iterations=iterations, native_run=self.run)

    def mutate(self, individual, array):
        n = self.change_amount()
        old = individual.fitness
        these = random.choices(list(range(len(array))), k=int(len(array) * self.percent()))
        these = np.asarray(these)
        array[these] += n
        new = individual.fit(self.fit_factor)

        if new >= old:
            pass
        else:
            array[these] -= 2 * n

    def run(self, individuals: np.ndarray) -> np.ndarray:
        """
        :param individuals:
        :return:
        """
        for ind in individuals:
            genes = ind.genes
            shape = genes.shape
            genes = genes.flatten()
            self.mutate(ind, genes)
            ind.genes = genes

        return individuals


class OPMutation(Layer):
    def __init__(self, pool, fitness_function, delay=0, every=1, iterations=1, end=math.inf, method="random", amount=1,
                 genes=1, constant=1, use_constant=False):
        super().__init__(every, delay, iterations, native_run=self.run, end=end)
        self.genes = make_callable(genes)
        self.pool = pool
        self.method = method
        self.amount = make_callable(amount)
        self.fitness_function = fitness_function
        self.constant = make_switcher(constant)
        self.use_constant = use_constant

    def random(self, individuals):
        selected = random.choices(individuals, k=self.amount())
        size = self.genes()

        for individual in selected:
            f = individual.raw_fit()
            individual.genes = individual.genes.reshape(self.pool.shape)
            old_genes = copy.deepcopy(individual.genes)
            gene_indicies = np.random.choice(len(individual.genes), size=size)
            if self.use_constant:
                individual.genes[gene_indicies] += self.constant()
            else:
                selected_genes = self.pool.rand_many(amount=size)
                individual.genes[gene_indicies] = selected_genes
            individual.genes = individual.genes.flatten()
            new_f = individual.raw_fit()
            if f > new_f:
                individual.genes = old_genes.flatten()
            individual.fit(1)

        return individuals
    def run(self, data):
        return self.random(data)


class OverPoweredMutation(Layer):
    def __init__(self, pool, iters, index, fitness_function, range_rate=1, method="smartint", delay=0, every=1,
                 end=math.inf):
        super().__init__(delay=delay, every=every, end=end, native_run=self.native_run, iterations=1)
        self.pool = pool
        self.iterations = make_callable(iters)
        self.index = index
        self.fitness_function = fitness_function
        self.method = method
        self.rand_range = make_callable(range_rate)
        logging.warning("Using OverPoweredMutation will override any custom fitness factor you set for your specified "
                        "index.")
        self.least_mutated = None

    def complete_random(self, data):
        individual = data[self.index]
        fitness = individual.fit(1)
        l = len(individual.genes)
        for i in range(self.iterations()):
            new = Individual(individual.pool, copy.deepcopy(individual.genes), fitness_func=individual.fitness_func)
            new.genes[random.randint(0, l - 1)] = self.pool.rand(index=1)
            newf = new.fit(1)
            if newf > fitness:
                data[self.index] = new
                fitness = newf
                individual = data[self.index]
        return data

    def smart(self, data):
        individual = data[self.index]
        fitness = individual.fit(1)
        individual.genes = individual.genes.reshape(self.pool.shape)
        l = len(individual.genes)
        if self.least_mutated is None:
            self.least_mutated = np.ones(l)
        indexes = random.choices(range(0, l), weights=self.least_mutated, k=self.iterations())
        for i in indexes:
            new = Individual(individual.pool, copy.deepcopy(individual.genes), fitness_func=individual.fitness_func)

            new.genes[i] += random.uniform(-self.rand_range(), self.rand_range())
            self.least_mutated[i] *= .96  # TODO: make this a parameter
            newf = new.fit(1)
            if newf > fitness:
                data[self.index] = new
                fitness = newf
                individual = data[self.index]
        return data

    def smartint(self, data):
        individual = data[self.index]
        fitness = individual.fit(1)
        l = len(individual.genes)
        for i in range(self.iterations()):
            new = Individual(individual.pool, copy.deepcopy(individual.genes), fitness_func=individual.fitness_func)

            new.genes[random.randint(0, l - 1)] += random.randint(-int(self.rand_range()), int(self.rand_range()))
            newf = new.fit(1)
            if newf > fitness:
                data[self.index] = new
                fitness = newf
                individual = data[self.index]
        return data

    def native_run(self, data):
        if self.method == "random":
            data = self.complete_random(data)
        if self.method == "smart":
            data = self.smart(data)
        if self.method == "smartint":
            data = self.smartint(data)
        return data

