import random
import random as r
from Finch.FinchGA.EvolveRates import *
from Finch.FinchGA.generic import *
import numpy as np
from difflib import SequenceMatcher


class Layer:
    def __init__(self, delay, native_run):
        """
        :param delay: Delay until used, until then it will simply return what it is given
        :param native_run: The run function to be used from other classes
        """
        self.delay = delay
        self.native_run = native_run

    def run(self, data, func):
        """

        :param data: The Generation type to be modified
        :param func: Any relevant function that is needed in the native_run
        :return: Whatever native_function returns
        """
        if self.delay <= 0:
            new = self.native_run(data, func)
            data = new
            return data
        else:
            self.delay -= 1
            return data


class GenerateData(Layer):
    def __init__(self, gene_pool, population, array_length, delay):
        """
        :param gene_pool: The gene pool to be used
        :param population: The population to be generated
        :param array_length:
        """
        super().__init__(delay=delay, native_run=self.native_run)  # the Layer class
        self.gene_pool = gene_pool
        self.population = population
        self.array_len = array_length

    def native_run(self, data, func):
        self.gene_pool.gen_data(data, self.population, self.array_len)
        return data.individuals


class NarrowGRN(Layer):  # Narrow Gene Regulatory Network. Promotes good chromosome (not individuals).
    def __init__(self, gene_pool, method="outer", amount=10, delay=0, reward=0.01, penalty=0.01):
        """
        :param gene_pool: The gene_pool to modify
        :param method: Can also be "all" defines how to calculate new weights. "all" recalculate
        all of them, "outer" will penalize the lowest fitness ones and reward the highest fitness. "best" will reward
        the best. "worst" will penalize the worst chromosome.
        :param amount: The amount of individuals to look at. Only relevant when the method is not "all".
        :param delay: The delay
        :param reward: The percentage to increase the weight of a gene
        :param penalty: Like reward
        """
        super().__init__(delay, self.native_run)
        self.amount = amount
        self.reward = reward
        self.penalty = penalty
        if not callable(reward):
            self.reward = Rates(reward, 0).constant  # the reward will remain the same
        if not callable(penalty):
            self.penalty = Rates(penalty, 0).constant  # the penalty will remain the same

        if not callable(amount):
            self.amount = Rates(amount, 0).constant  # the amount will remain the same
        self.gene_pool = gene_pool
        self.method = method

    def best(self, data):
        """
        :param data: The data
        :return: data
        """

        best = data.individuals[-int(self.amount()):]  # Most

        for individual in best:
            for gene in individual.chromosome.genes:
                gene.reward(self.reward())
        return data

    def worst(self, data):

        worst = data.individuals[0: int(self.amount())]  # Least fit
        for individual in worst:
            for gene in individual.chromosome.genes:

                gene.penalize(self.penalty())


        return data

    def outer(self, data):
        """
        :param data: The data
        :return: data
        """

        data = self.best(data)
        data = self.worst(data)
        return data

    def alld(self, data):
        n = 0
        for gene in data:
            pass  # TODO: implement this

    def native_run(self, data, func):

        """
        :param data: T
        :param func: this does nothing
        :return:
        """
        if self.method == "all":
            self.alld(data)
        if self.method == "worst":
            self.worst(data)
        if self.method == "best":
            self.best(data)
        if self.method == "outer":
            self.outer(data)

        return data


class CalcFitness(Layer):
    def __init__(self, delay=0):
        super().__init__(delay=delay, native_run=self.native_run)

    @staticmethod
    def native_run(data, func):
        data.fit_all(1)
        data.individuals = np.array(data.sort())
        return data


class Duplicate(Layer):
    def __init__(self, clones=1, delay=0):
        super().__init__(delay, self.native_run)
        self.clones = 1

    def native_run(self, data, func):
        for i in range(self.clones):
            data.add(data.individuals[-1])
        return data


class Mutate(Layer):
    def __init__(self, pool, delay=0):
        super().__init__(delay, self.native_run)
        self.pool = pool

    def native_run(self, data, func):
        for i in data.individuals:
            i.mfunction(self.pool)
        return data


class SequentialEnvironment():
    def __init__(self, layers=[]):
        self.layers = layers
        self.data = Generation([])

    def simulate_env(self, epochs, fitness=None, every=1):
        history= []
        for i in range(epochs):
            for d in self.layers:
                d.run(self.data, fitness)
            if i % every == 0:
                print(self.data.individuals[-1].fitness, self.data.individuals[-1].chromosome.get_raw())
            history.append(self.data.individuals[-1].fitness)
        return history

class Kill(Layer):
    def __init__(self, percent, delay=0):
        super().__init__(delay=delay, native_run=self.native_run)
        self.name = "kill"
        self.percent = percent

        if not callable(percent):
            self.percent = Rates(percent, 0).constant
        self.now = 0

    def native_run(self, data, func):
        data.individuals = data.individuals[int(len(data.individuals) * self.percent()):-1]
        return data


class UpdateWeights(Layer):
    def __init__(self, pool, delay=0):
        super().__init__(delay=delay, native_run=self.native_run)
        self.pool = pool

    def native_run(self, data, func):
        self.pool.update()
        return data


class Function(Layer):
    def __init__(self, fun, delay=0):
        super().__init__(delay=delay, native_run=self.native_run)
        self.func = fun

    def native_run(self, data, func):
        self.func(data=data)
