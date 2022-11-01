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
    def __init__(self, gene_pool, method="outer", amount=1):
        """
        :param gene_pool: The gene_pool to modify
        :param method: Can also be "all" defines how to calculate new weights. "all" recalculate
        all of them, "outer" will penalize the lowest fitness ones and reward the highest fitness.
        :param amount: The amount of individuals to look at. Only relevant when the method is not "all".
        """
        self.amount = amount
        self.gene_pool = gene_pool
        self.method = method
