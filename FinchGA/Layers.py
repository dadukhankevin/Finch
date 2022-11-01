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
            return new
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

    def native_run(self, data):
        data = self.gene_pool.gen_data(data, self.population, self.array_len)
        return data


class NarrowGRN(Layer):  # Narrow Gene Regulatory Network. Promotes good genes (not individuals).
    def __init__(self, gene_pool, method="outer", amount=1, delay=0, reward=0.01, penalty=0.01):
        """
        :param gene_pool: The gene_pool to modify
        :param method: Can also be "all" defines how to calculate new weights. "all" recalculate
        all of them, "outer" will penalize the lowest fitness ones and reward the highest fitness. "best" will reward
        the best. "worst" will penalize the worst genes.
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
        best = data[-int(self.amount()):]  # Most fit
        for gene in best:
            gene.reward(self.reward())

    def worst(self, data):
        worst = data[0: int(self.amount())]  # Least fit
        for gene in worst:
            gene.penalize(self.penalty())

    def outer(self, data):
        """
        :param data: The data
        :return: data
        """
        self.best(data)
        self.worst(data)

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
class Duplicate(Layer):
    def __init__(self,clones=1, delay=0):
        super().__init__(delay, self.native_run)


