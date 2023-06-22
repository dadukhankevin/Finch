import numpy as np


class Individual:
    def __init__(self, genes, fitness_function):
        self.genes = genes
        self.fitness = 0
        self.fitness_function = fitness_function

    def fit(self):
        self.fitness = self.fitness_function()

